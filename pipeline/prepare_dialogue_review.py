#!/usr/bin/env python3
"""Source-bound dialogue review sheets, speech windows and sheet-observation plans.

No paid API calls. A sheet is evidence for review, never an acceptance record.
Sheets render on the qualified GPU host. --preview draws the same sheets anywhere
ffmpeg runs, names them *-preview-not-final, records the real host and ffmpeg
build, and writes no observation manifest. When the local ffmpeg has no
drawtext filter or no font is available, clip IDs and timestamps are not burned
in; the frame ordinals and times stay in each proof record.
--measure-speech-windows runs anywhere ffmpeg runs. Each measurement is a new speech-windows-attempt-NNN.json:
conservative, hash-bound speech windows. A decisions row for
edit/admit_review_decisions.py names an attempt and item as speechWindowFrom;
admission checks the item's sourceSha256 against the take and copies the
window into the review record. Earlier attempts stay; speech-windows.json indexes them,
so a second run with transcript evidence, other thresholds or another host adds
an attempt instead of refusing. With
--observation-prompt and --observation-max-credits it also writes
sheet-observation-manifest.json for observe_media_finite.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common

WORKER_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
PREVIEW = "-preview-not-final"

if __package__:
    from ._common import adopt_legacy_record, atomic_json, attempt_records, write_attempt_json
else:
    from _common import adopt_legacy_record, atomic_json, attempt_records, write_attempt_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Immutable review input changed: " + str(path))
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def select_jobs(plan, selected_ids, expected):
    """An explicit completed subset retains the original execution binding."""
    rows = plan["jobs"]
    known = {row["id"] for row in rows}
    if len(known) != len(rows):
        raise ValueError("Duplicate job IDs in execution manifest")
    if selected_ids is not None:
        requested = selected_ids.split(",")
        if len(set(requested)) != len(requested) or not set(requested) <= known:
            raise ValueError("Selected IDs are duplicate or absent from execution manifest")
        rows = [row for row in rows if row["id"] in requested]
    if len(rows) != expected or not 1 <= expected <= 160:
        raise ValueError("Exact bounded job count required")
    return rows


def timestamp_filter(font, label=""):
    """The drawtext stage that burns each frame's clip ID and time, or '' without a font."""
    if font is None:
        return ""
    if any(c in str(font) for c in "':,[]\\;"):
        raise ValueError("Unsafe font path for an ffmpeg filter: " + str(font))
    return ("drawtext=fontfile=" + str(font) + ":"
            "text='" + label + "%{pts\\:hms}':fontsize=18:fontcolor=white:box=1:"
            "boxcolor=black@0.75:x=4:y=4,")


def verified_jobs(plan, manifest, root, here, selected):
    """Each selected job's video, digest and sidecar, refusing any changed binding."""
    jobs = []
    for row in selected:
        source = (root / row["source"]).resolve()
        source.relative_to(root)
        prompt = (root / row["prompt"]).resolve()
        prompt.relative_to(here)
        video = (root / row["outputStem"]).with_suffix(".mp4").resolve()
        video.relative_to(here)
        meta_path = video.with_suffix(".vid.json")
        meta = json.loads(meta_path.read_text())
        digest = sha(video)
        if (sha(source) != row["sourceSha256"] or sha(prompt) != row["promptSha256"]
                or meta.get("outputSha256") != digest
                or meta.get("sourceSha256") != row["sourceSha256"]
                or meta.get("executionManifestSha256") != sha(manifest)):
            raise ValueError("Source/output provenance changed: " + row["id"])
        jobs.append((row, video, digest, meta_path))
    return jobs


def _silences(path, noise_db, minimum_silence):
    done = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
                           "-af", f"silencedetect=noise={noise_db}dB:d={minimum_silence}",
                           "-f", "null", "-"], capture_output=True, text=True, check=True,
                          timeout=300)
    starts = [float(x) for x in re.findall(r"silence_start: (-?[0-9.]+)", done.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[0-9.]+)", done.stderr)]
    return list(zip(starts, ends + [None] * (len(starts) - len(ends))))


def speech_window(duration, silences, handle, transcript_span=None):
    """A conservative speech window from detected silences, or the whole clip.

    Energy detection hears ambience as speech and soft words as silence, so the
    window gets at least ``handle`` seconds either side, never excludes a
    transcript span, and falls back to the full clip when it would be empty.
    """
    start, end = 0.0, duration
    for silence_start, silence_end in silences:
        if silence_start <= 0.0 + 1e-6 and silence_end is not None:
            start = max(start, silence_end)
        if silence_end is None or silence_end >= duration - 1e-3:
            end = min(end, max(silence_start, 0.0))
    detected = None if end <= start else {"start": round(start, 6), "end": round(end, 6)}
    if detected is None:
        return None, {"start": 0.0, "end": round(duration, 6)}, True
    if transcript_span:
        start = min(start, float(transcript_span["start"]))
        end = max(end, float(transcript_span["end"]))
    window = {"start": round(max(0.0, start - handle), 6),
              "end": round(min(duration, end + handle), 6)}
    return detected, window, False


def measure_speech_windows(manifest, root, jobs, *, noise_db, minimum_silence, handle,
                           transcripts=None):
    """Write one offline, hash-bound speech-windows attempt beside the manifest.

    This is measurement, not admission: a decisions row names this attempt
    and item as speechWindowFrom, and admit_review_decisions.py copies that
    window, after checking the item's hashes against the take, into the
    record the timeline reads as speechWindow. A run whose
    document equals the latest attempt adds nothing; any other run adds an
    attempt and never replaces one. speech-windows.json is the attempt index.
    """
    here = manifest.parent
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                            check=True).stdout.splitlines()[0]
    spans = {}
    if transcripts is not None:
        evidence = json.loads(transcripts.read_text(encoding="utf-8"))
        spans = {row["id"]: row for row in evidence.get("items", [])
                 if row.get("transcriptSpan")}
    items = []
    for row, video, digest, meta_path in jobs:
        probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams",
                                           "-show_format", "-of", "json", str(video)],
                                          capture_output=True, text=True, check=True,
                                          timeout=120).stdout)
        if not any(s.get("codec_type") == "audio" for s in probe.get("streams", [])):
            raise ValueError("Speech windows need the clip's audio stream: " + row["id"])
        duration = float(probe["format"]["duration"])
        span = spans.get(row["id"])
        if span and span.get("sourceSha256") != digest:
            raise ValueError("Transcript evidence describes other bytes: " + row["id"])
        detected, window, fallback = speech_window(
            duration, _silences(video, noise_db, minimum_silence), handle,
            span["transcriptSpan"] if span else None)
        items.append({"id": row["id"], "source": str(video.relative_to(here)),
                      "sourceSha256": digest, "sidecarSha256": sha(meta_path),
                      "durationSeconds": round(duration, 6), "detected": detected,
                      "speechWindow": window, "fallbackToFullClip": fallback,
                      "transcriptSpan": span["transcriptSpan"] if span else None})
    doc = {"schemaVersion": 1, "kind": "dialogue-speech-windows",
           "manifestSha256": sha(manifest),
           "transcriptEvidenceSha256": sha(transcripts) if transcripts else None,
           "method": "ffmpeg-silencedetect",
           "thresholds": {"noiseDb": noise_db, "minimumSilenceSeconds": minimum_silence,
                          "minimumHandleSeconds": handle},
           "ffmpeg": ffmpeg, "computeHost": platform.node(),
           "policy": ("Energy-based measurement; complete speech outranks a tight trim. A window "
                      "is never narrower than a supplied transcript span and falls back to the "
                      "full clip when no speech is found. Admission copies a chosen window."),
           "admission": "not implied", "items": items}
    # A single speech-windows.json from before the attempt files is kept as one.
    adopt_legacy_record(here, "speech-windows")
    attempts = attempt_records(here, "speech-windows")
    if attempts and attempts[-1]["record"] == doc:
        target = here / attempts[-1]["path"]
    else:
        target = write_attempt_json(here, "speech-windows", doc)
    atomic_json(here / "speech-windows.json", {
        "kind": "dialogue-speech-windows-index", "manifestSha256": doc["manifestSha256"],
        "latest": target.name, "attempts": attempt_records(here, "speech-windows")})
    return target, doc


def observation_manifest(here, root, plan, evidence, prompt_file, max_credits):
    """A sheet-observation plan in the shape observe_media_finite.py runs.

    The prompt is this project's own file; nothing here writes a rubric.
    Sheets are images, the only part the observation lane sends today.
    """
    here, root = here.resolve(), root.resolve()
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Observation prompt file is empty")
    jobs = []
    for item in evidence:
        if not re.fullmatch(r"[a-z0-9-]+", item["id"]):
            raise ValueError("Observation job IDs must be lowercase letters, digits and hyphens: "
                             + item["id"])
        sheet = (here / item["sheet"]).resolve()
        jobs.append({"id": item["id"], "prompt": prompt,
                     "media": [{"path": sheet.relative_to(root).as_posix(),
                                "sha256": item["sheetSha256"]}]})
    return {"episode": plan["episode"], "projectRoot": os.path.relpath(root, here),
            "financialApproval": False, "maxCredits": max_credits,
            "promptFile": prompt_file.resolve().relative_to(root).as_posix(),
            "promptSha256": sha(prompt_file),
            "purpose": "Blind description of review sheets; advisory only", "jobs": jobs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--repo-root", "--project", dest="project", type=Path, required=True,
                    help="project root; the manifest, stills and clips must sit inside it")
    ap.add_argument("--max-jobs", type=int, required=True)
    ap.add_argument("--ids", help="Explicit comma-separated completed subset; never changes source manifest")
    windows = ap.add_argument_group("offline speech windows (ffmpeg only, any host)")
    windows.add_argument("--measure-speech-windows", action="store_true",
                         help="add a speech-windows attempt and stop; no sheets, no worker needed")
    windows.add_argument("--noise-db", type=float, default=-35.0)
    windows.add_argument("--minimum-silence", type=float, default=0.30)
    windows.add_argument("--minimum-handle", type=float, default=0.25)
    windows.add_argument("--transcript-evidence", type=Path,
                         help="measure_dialogue_speech.py output; windows never cut its spans")
    ap.add_argument("--preview", action="store_true",
                    help="Draw sheets anywhere ffmpeg runs; outputs are labelled preview-not-final")
    ap.add_argument("--font", type=Path, help="Timestamp font; otherwise the profile fonts.review, then the worker DejaVuSans")
    observing = ap.add_argument_group("sheet observation plan (optional)")
    observing.add_argument("--observation-prompt", type=Path,
                           help="this project's own describe-only prompt file")
    observing.add_argument("--observation-max-credits", type=float,
                           help="ceiling the observation approval record will bind")
    a = ap.parse_args(argv)
    if (a.observation_prompt is None) != (a.observation_max_credits is None):
        ap.error("--observation-prompt and --observation-max-credits go together")
    if a.preview and a.observation_prompt is not None:
        ap.error("--preview writes no observation plan")
    if not a.measure_speech_windows:
        # Sheets: the worker gate comes before any project file is read.
        host = None
        if a.preview:
            font = _common.review_font(a.font) or (WORKER_FONT if WORKER_FONT.is_file() else None)
            if "drawtext" not in _common.ffmpeg_capabilities()["filters"]:
                font = None
            host = _common.preview_host_record(font)
        else:
            if platform.system() != "Linux":
                raise SystemExit("Heavy media preparation requires the qualified GPU host")
            subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, check=True, timeout=30)
            font = _common.review_font(a.font) or WORKER_FONT
        suffix = PREVIEW if a.preview else ""
    root = a.project.resolve()
    manifest = a.manifest.resolve()
    here = manifest.parent
    here.relative_to(root)
    plan = json.loads(manifest.read_text())
    selected = select_jobs(plan, a.ids, a.max_jobs)
    if a.measure_speech_windows:
        target, doc = measure_speech_windows(
            manifest, root, verified_jobs(plan, manifest, root, here, selected),
            noise_db=a.noise_db, minimum_silence=a.minimum_silence, handle=a.minimum_handle,
            transcripts=a.transcript_evidence)
        print(json.dumps({"speechWindows": str(target),
                          "index": str(here / "speech-windows.json"), "items": len(doc["items"]),
                          "fallbacks": sum(x["fallbackToFullClip"] for x in doc["items"]),
                          "networkCalls": 0}))
        return
    review = here / "dense-review"
    review.mkdir(exist_ok=True)
    jobs = verified_jobs(plan, manifest, root, here, selected)

    def one(job):
        row, video, digest, meta_path = job
        probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames",
            "-show_streams", "-show_format", "-of", "json", str(video)],
            check=True, capture_output=True, text=True, timeout=120).stdout)
        v = next(s for s in probe["streams"] if s["codec_type"] == "video")
        if v["r_frame_rate"] != "24/1":
            raise ValueError("Review sampling requires known 24fps input")
        count = int(v["nb_read_frames"])
        frames = sorted(set(range(0, count, 12)) | {count - 1})
        if len(frames) > 36:
            raise ValueError("Review sheet would omit frames; increase its explicit grid")
        sheet = review / (row["id"] + "-2fps" + suffix + ".jpg")
        proof = sheet.with_suffix(".json")
        entry = dict(id=row["id"], source=str(video.relative_to(here)),
                     sourceSha256=digest, sidecarSha256=sha(meta_path),
                     frameOrdinals=frames, timestamps=[round(n/24, 6) for n in frames],
                     sheet=str(sheet.relative_to(here)), speaker=row["speaker"],
                     hindi=row["hindi"], probe=probe,
                     reviewScope="Every 12th frame plus final frame; not continuous playback or approval")
        if a.preview:
            entry.update(preview=True, finalApproved=False, burnedTimestamps=font is not None)
        if sheet.exists() != proof.exists():
            raise ValueError("Incomplete review pair; reconcile before overwriting")
        if sheet.exists():
            old = json.loads(proof.read_text())
            if old.get("sourceSha256") != digest or old.get("sheetSha256") != sha(sheet):
                raise ValueError("Existing review evidence changed")
            return old
        rows = (len(frames) + 4) // 5
        # Five 400 px columns: a 2000 px sheet with each frame's clip ID and time.
        label = (row["id"] + " " if re.fullmatch(r"[A-Za-z0-9_.-]+", row["id"]) else "")
        vf = ("select='not(mod(n,12))+eq(n," + str(count-1) + ")',scale=400:225,"
              + timestamp_filter(font, label) + "tile=5x" + str(rows))
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(video), "-vf", vf,
                        "-frames:v", "1", "-q:v", "2", str(sheet)], check=True, timeout=120)
        entry["sheetSha256"] = sha(sheet)
        save(proof, entry)
        return entry

    with ThreadPoolExecutor(max_workers=3) as pool:
        evidence = list(pool.map(one, jobs))
    if a.preview:
        _common.atomic_json(here / ("dense-review" + PREVIEW + ".json"), {
            "manifestSha256": sha(manifest), "episode": plan["episode"], "items": evidence,
            "acceptance": "not implied", "kind": "dialogue-sheets-preview-not-final", "hostRecord": host})
        print(json.dumps({"jobs": len(jobs), "sheets": len(evidence), "networkCalls": 0, "preview": True}))
        return
    save(here / "dense-review-index.json", {"manifestSha256": sha(manifest),
        "episode": plan["episode"], "items": evidence, "acceptance": "not implied"})
    if a.observation_prompt is not None:
        save(here / "sheet-observation-manifest.json", observation_manifest(
            here, root, plan, evidence, a.observation_prompt, a.observation_max_credits))
    print(json.dumps({"jobs": len(jobs), "sheets": len(evidence), "networkCalls": 0}))


if __name__ == "__main__":
    main()
