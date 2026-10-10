"""Finite, hash-bound H3 client for an exclusively owned retained server.

No GCP mutation, server launch or blind resubmission. An uncertain POST or lost
task stops the run for reconciliation. VM shutdown is owned by the caller.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import requests

if not __debug__:
    raise SystemExit("Optimized Python disables production safety gates")

SERVER = "http://127.0.0.1:30010"



def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def validate_manifest(path, maximum):
    plan = json.loads(path.read_text())
    assert plan["route"] == "owned-h100-h3" and plan["generationApiAllowed"] is False
    assert isinstance(plan["deadlineAt"], (int, float))
    items = plan["items"]
    assert 1 <= len(items) <= maximum <= 32
    assert len({i["id"] for i in items}) == len(items)
    for item in items:
        assert re.fullmatch(r"[A-Za-z0-9_-]+", item["id"])
        assert item["prompt"].strip() and 4 <= item["seconds"] <= 15
        assert item["short_edge"] == 768 and item["steps"] in (20, 30, 50)
        assert (item.get("expectedWidth", 1344), item.get("expectedHeight", 768)) in ((1344, 768), (768, 1344))
        assert not item.get("last_frame")
        assert 0 < item["expectedFrames"] <= 400
        assert abs(item["expectedFrames"] / 24 - item["seconds"]) < 0.75
        for name, digest in [("still", "sourceSha256"), ("last_frame_still", "lastFrameSha256")]:
            if name not in item:
                continue
            assert Path(item[name]).name == item[name]
            assert sha(path.parent / "stills" / item[name]) == item[digest]
    return plan


class StopRun(RuntimeError):
    pass


class RejectedTake(RuntimeError):
    pass


def requires_server_recovery(result):
    """A responding HTTP frontend does not prove GPU ranks survived an OOM."""
    message = json.dumps(result.get('error', ''), ensure_ascii=False).lower()
    return 'out of memory' in message or 'outofmemoryerror' in message


def verify_frame_contract(video, item):
    actual = int(video['nb_read_frames'])
    expected = item['expectedFrames']
    if actual != expected:
        raise StopRun(
            f'Frame contract mismatch: expected {expected}, observed {actual}. '
            'Downloaded take retained; reconcile duration contract before any regeneration.'
        )


def run(manifest_path, out, maximum):
    plan = validate_manifest(manifest_path, maximum)
    out.mkdir(parents=True, exist_ok=True)
    runtime_value = os.environ.get("AGENTIC_H3_ROOT")
    if not runtime_value or not Path(runtime_value).is_absolute():
        raise StopRun("Set AGENTIC_H3_ROOT to the qualified runtime directory")
    lock = open(Path(runtime_value).resolve() / "finite-client.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = (process / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(arg.decode(errors="replace").endswith(("/sgl_worker.py", "/req.py", "/chain_batches.sh"))
               for arg in argv):
            raise StopRun("Another known H3 client owns work: " + process.name)
    state_path = out / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        assert state["manifestSha256"] == sha(manifest_path)
        assert state["clientSha256"] == sha(__file__), "Changed client: reconcile explicitly"
    else:
        now = time.time()
        state = dict(manifestSha256=sha(manifest_path), clientSha256=sha(__file__),
                     startedAt=now, deadlineAt=min(now + 7000, plan["deadlineAt"]), jobs={})
        atomic_json(state_path, state)

    def save():
        atomic_json(state_path, state)

    def remaining():
        left = state["deadlineAt"] - time.time()
        if left <= 0:
            raise StopRun("Persisted finite run deadline reached")
        return left

    def req(method, path, **kwargs):
        return requests.request(method, SERVER + path,
                                timeout=(min(5, remaining()), min(30, remaining())), **kwargs)

    def health():
        try:
            req("GET", "/health").raise_for_status()
        except Exception as exc:
            raise StopRun("Server health failed") from exc

    def inspect(path, item):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
             "-of", "json", str(path)], check=True, capture_output=True, text=True,
            timeout=min(90, remaining()))
        probe = json.loads(result.stdout)
        streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
        assert len(streams) == 1
        v = streams[0]
        assert (v["width"], v["height"]) == (item.get("expectedWidth", 1344), item.get("expectedHeight", 768))
        assert v["r_frame_rate"] == "24/1"
        verify_frame_contract(v, item)
        assert abs(float(v["duration"]) - item["expectedFrames"] / 24) < 0.002
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            capture_output=True, check=True, timeout=min(120, remaining()))
        assert not result.stderr
        return probe

    try:
        health()
        for item in plan["items"]:
            remaining()
            job = state["jobs"].setdefault(item["id"], {"attempts": []})
            if job.get("status") == "complete":
                assert sha(out / job["output"]) == job["sha256"]
                continue
            while job.get("status") != "complete":
                remaining()
                attempts = job["attempts"]
                attempt = attempts[-1] if attempts else None
                if attempt and attempt.get("requiresServerRecovery"):
                    raise StopRun("Persisted GPU failure; explicit server recovery required")
                if attempt and attempt["status"] in ("posting", "uncertain-submit"):
                    raise StopRun("Uncertain POST; reconcile task before resubmitting")
                if not attempt or attempt["status"] == "failed":
                    if len(attempts) >= 2:
                        job["status"] = "failed"
                        save()
                        break
                    health()
                    for name, digest in [("still", "sourceSha256"), ("last_frame_still", "lastFrameSha256")]:
                        if name in item and sha(manifest_path.parent / "stills" / item[name]) != item[digest]:
                            raise StopRun("Staged input changed before submission")
                    conditions = [{"type": "image",
                                   "uri": (manifest_path.parent / "stills" / item["still"]).as_uri(),
                                   "role": "keyframe", "frame_index": 0}]
                    if item.get("last_frame_still"):
                        conditions.append({"type": "image",
                            "uri": (manifest_path.parent / "stills" / item["last_frame_still"]).as_uri(),
                            "role": "keyframe", "frame_index": -1})
                    body = dict(model="MiniMaxAI/MiniMax-H3", task="fl2va", prompt=item["prompt"],
                                seconds=item["seconds"], conditions=conditions,
                                target=dict(short_edge=768, aspect_ratio=item.get("aspect", "auto"),
                                            duration_seconds=item["seconds"]),
                                num_outputs_per_prompt=1, num_inference_steps=item["steps"],
                                flow_shift=12.0, audio_flow_shift=3.0,
                                seed=item["seed"] + len(attempts))
                    attempt = dict(number=len(attempts) + 1, status="posting",
                                   startedAt=time.time(), request=body)
                    attempts.append(attempt)
                    save()
                    try:
                        response = req("POST", "/v1/videos", json=body)
                        response.raise_for_status()
                        task_id = response.json()["id"]
                    except Exception as exc:
                        attempt.update(status="uncertain-submit", error=repr(exc))
                        save()
                        raise StopRun("POST outcome uncertain") from exc
                    attempt.update(status="pending", taskId=task_id)
                    save()
                try:
                    while "completedResponse" not in attempt:
                        remaining()
                        response = req("GET", "/v1/videos/" + attempt["taskId"])
                        if response.status_code == 404:
                            raise StopRun("Task lost; explicit reconciliation required")
                        response.raise_for_status()
                        result = response.json()
                        if result.get("status") in ("completed", "succeeded", "done"):
                            attempt["completedResponse"] = result
                            save()
                            break
                        if result.get("status") in ("failed", "error", "cancelled"):
                            if requires_server_recovery(result):
                                attempt.update(status="failed", completedResponse=result,
                                               error=json.dumps(result),
                                               requiresServerRecovery=True)
                                save()
                                raise StopRun("GPU out of memory; explicit server recovery required")
                            raise RejectedTake(json.dumps(result)[:1500])
                        if time.time() > attempt["startedAt"] + 120 + 6 * item["steps"]:
                            raise StopRun("Task timeout; no duplicate work submitted")
                        health()
                        time.sleep(min(2, remaining()))
                    native = out / (item["id"] + ".attempt" + str(attempt["number"]) + ".mp4")
                    if not (attempt.get("downloadSha256") and native.exists()
                            and sha(native) == attempt["downloadSha256"]):
                        partial = native.with_suffix(".mp4.partial")
                        with req("GET", "/v1/videos/" + attempt["taskId"] + "/content",
                                 stream=True) as response:
                            response.raise_for_status()
                            with partial.open("wb") as stream:
                                for chunk in response.iter_content(1024 * 1024):
                                    remaining()
                                    stream.write(chunk)
                        partial.replace(native)
                        attempt["downloadSha256"] = sha(native)
                        save()
                    try:
                        probe = inspect(native, item)
                    except StopRun:
                        raise
                    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                        raise StopRun("QC infrastructure failed; downloaded take retained") from exc
                    except Exception as exc:
                        raise RejectedTake("Technical verification: " + repr(exc)) from exc
                    attempt.update(status="complete", probe=probe)
                    job.update(status="complete", output=native.name, sha256=sha(native),
                               sourceSha256=item["sourceSha256"],
                               lastFrameSha256=item.get("lastFrameSha256"),
                               audioPolicy="Raw provenance only; -an at production ingest",
                               creativeVerdict="pending-independent-review")
                    save()
                    print(json.dumps(dict(id=item["id"], status="complete",
                                          seconds=probe["format"]["duration"])), flush=True)
                except RejectedTake as exc:
                    attempt.update(status="failed", error=str(exc))
                    save()
    except Exception as exc:
        state.update(stoppedAt=time.time(), stopReason=repr(exc))
        save()
        raise
    state.update(finishedAt=time.time(),
                 completeCount=sum(j.get("status") == "complete" for j in state["jobs"].values()))
    save()
    print(json.dumps(dict(complete=state["completeCount"], expected=len(plan["items"]))), flush=True)
    return 0 if state["completeCount"] == len(plan["items"]) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.manifest.resolve(), args.out.resolve(), args.max_jobs))
