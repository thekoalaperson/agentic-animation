#!/usr/bin/env python3
"""Dry-run-first, finite Kie/Suno score execution and recovery.

This lane never queries or mutates a wallet and has no top-up operation.  A
generation POST is possible only with --submit, an authorized immutable
manifest, and caller-supplied request/credit ceilings.  Each movement receives
at most one POST.  Ambiguous POSTs and provider failures stop the run and are
never regenerated automatically.  --poll is a separate, idempotent recovery
operation for already-persisted task IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

if __package__:
    from ._common import (atomic_json as _atomic_json,
                          atomic_new_bytes as _atomic_bytes_new, env_key,
                          sha256_file as _sha_file)
    from .paid_approval import PaidApprovalError, validate_paid_approval
else:
    from _common import (atomic_json as _atomic_json,
                         atomic_new_bytes as _atomic_bytes_new, env_key,
                         sha256_file as _sha_file)
    from paid_approval import PaidApprovalError, validate_paid_approval


API_ROOT = "https://api.kie.ai/api/v1/"
STATE_VERSION = 1
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
SUCCESS = {"SUCCESS", "CALLBACK_EXCEPTION"}
# https://docs.kie.ai/suno-api/get-music-details — partial success still needs
# polling until both included tracks are ready.
IN_PROGRESS = {"PENDING", "WAITING", "QUEUING", "GENERATING", "CREATE_TASK",
               "TEXT_SUCCESS", "FIRST_SUCCESS"}


class PlanError(ValueError):
    pass


class UncertainRequest(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _now() -> float:
    return time.time()


def _persist(path: Path, state: dict[str, Any]) -> None:
    state["updatedAt"] = _now()
    _atomic_json(path, state)


def _relative_child(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PlanError(f"{label} must be a non-empty relative path")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise PlanError(f"{label} must stay below the execution manifest")
    return (root / rel).resolve()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _verified_binding(manifest_path: Path, doc: dict[str, Any],
                      path_key: str, sha_key: str) -> dict[str, str] | None:
    raw_path, raw_sha = doc.get(path_key), doc.get(sha_key)
    if raw_path is None and raw_sha is None:
        return None
    if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_sha, str):
        raise PlanError(f"{path_key} and {sha_key} must be supplied together")
    rel = Path(raw_path)
    if rel.is_absolute():
        raise PlanError(f"{path_key} must be relative")
    mode = doc.get("provenancePathBase")
    if mode == "project":
        value = doc.get("projectRoot")
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise PlanError("project provenance requires a relative projectRoot")
        base = (manifest_path.parent / value).resolve()
        if not base.is_dir():
            raise PlanError("project provenance root is missing")
    elif mode == "manifest":
        base = manifest_path.parent
    else:
        raise PlanError(
            "provenancePathBase must be 'project' or 'manifest' when provenance is declared"
        )
    path = (base / rel).resolve()
    if not _is_below(path, base) or not path.is_file():
        raise PlanError(f"{path_key} is missing or escapes its declared base: {path}")
    actual = _sha_file(path)
    if actual != raw_sha.lower():
        raise PlanError(f"{path_key} changed after the score manifest was frozen")
    return {"kind": path_key, "path": str(path), "sha256": actual}


def load_plan(manifest_path: Path, max_requests: int,
              max_credits: int) -> dict[str, Any]:
    if max_requests <= 0 or max_credits <= 0:
        raise PlanError("--max-requests and --max-credits must be positive")
    manifest_path = manifest_path.resolve()
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read execution manifest: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise PlanError("execution manifest version must be 1")
    if not isinstance(doc.get("financialApproval"), bool):
        raise PlanError("financialApproval must be an explicit boolean")
    if not isinstance(doc.get("currentPriceVerified"), bool):
        raise PlanError("currentPriceVerified must be an explicit boolean")
    root = manifest_path.parent
    source_path = _relative_child(root, doc.get("sourceDraft"), "sourceDraft")
    if not source_path.is_file():
        raise PlanError(f"source draft is missing: {source_path}")
    source_sha = _sha_file(source_path)
    if source_sha != doc.get("sourceDraftSha256"):
        raise PlanError("source draft changed after execution manifest was frozen")
    production_binding = _verified_binding(
        manifest_path, doc, "productionOverrides", "productionOverridesSha256"
    )
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"source draft is invalid JSON: {exc}") from exc
    cues = doc.get("cues")
    if not isinstance(cues, list) or not cues:
        raise PlanError("cues must be a non-empty array")
    source_common = source.get("commonPayload")
    source_cues = source.get("cues")
    if not isinstance(source_common, dict) or not isinstance(source_cues, list):
        raise PlanError("source draft lacks commonPayload or cues")
    materialized_source = [
        {**cue, "payload": {**source_common, **cue.get("payload", {})}}
        for cue in source_cues
    ]
    if cues != materialized_source:
        raise PlanError("frozen cues are not byte-semantically equal to source draft cues")
    ids: set[str] = set()
    for cue in cues:
        if not isinstance(cue, dict) or set(cue) != {"id", "shots", "emotionalArc", "payload"}:
            raise PlanError("each cue must freeze id, shots, emotionalArc and payload")
        cue_id = cue.get("id")
        if not isinstance(cue_id, str) or not SAFE_ID.fullmatch(cue_id) or cue_id in ids:
            raise PlanError(f"duplicate or malformed cue id: {cue_id}")
        ids.add(cue_id)
        payload = cue.get("payload")
        required = {"model", "customMode", "instrumental", "prompt", "title",
                    "duration", "callBackUrl", "negativeTags", "style"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise PlanError(f"{cue_id}: payload fields must be exactly {sorted(required)}")
        if payload["model"] not in {"V5_5", "V6"} or payload["customMode"] is not True:
            raise PlanError(f"{cue_id}: only V5_5 or V6 custom mode is allowed")
        if payload["instrumental"] is not True or payload["prompt"] != "":
            raise PlanError(f"{cue_id}: score must be instrumental with empty lyrics prompt")
        if not isinstance(payload["duration"], int) or not 10 <= payload["duration"] <= 360:
            raise PlanError(f"{cue_id}: duration must be an integer from 10 to 360")
        if not isinstance(payload["style"], str) or not 0 < len(payload["style"]) <= 1000:
            raise PlanError(f"{cue_id}: style must contain at most 1000 characters")
        if not isinstance(payload["title"], str) or not 0 < len(payload["title"]) <= 80:
            raise PlanError(f"{cue_id}: title must contain at most 80 characters")
        if not isinstance(payload["negativeTags"], str) or not payload["negativeTags"].strip():
            raise PlanError(f"{cue_id}: negativeTags must be non-empty")
    credits_each = doc.get("creditsPerRequest")
    manifest_requests = doc.get("maxRequests")
    manifest_credits = doc.get("maxCredits")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
               for x in (credits_each, manifest_requests, manifest_credits)):
        raise PlanError("creditsPerRequest/maxRequests/maxCredits must be positive integers")
    request_count = len(cues)
    expected_credits = request_count * credits_each
    if request_count > manifest_requests or expected_credits > manifest_credits:
        raise PlanError("frozen cue count exceeds manifest ceiling")
    if request_count > max_requests or expected_credits > max_credits:
        raise PlanError(
            f"plan needs {request_count} requests/{expected_credits} credits, exceeding "
            f"explicit CLI caps {max_requests}/{max_credits}")
    if max_requests > manifest_requests or max_credits > manifest_credits:
        raise PlanError("CLI caps may not exceed the immutable manifest ceilings")
    output_root = _relative_child(root, doc.get("outputDirectory"), "outputDirectory")
    expected_candidates = doc.get("expectedCandidatesPerRequest")
    if (not isinstance(expected_candidates, int) or isinstance(expected_candidates, bool)
            or expected_candidates <= 0 or expected_candidates > 8):
        raise PlanError("expectedCandidatesPerRequest must be an integer from 1 to 8")
    fingerprint = _sha_file(manifest_path)
    return {"path": str(manifest_path), "sha256": fingerprint, "doc": doc,
            "sourceSha256": source_sha, "cues": cues, "outputRoot": str(output_root),
            "productionBinding": production_binding,
            "expectedCandidatesPerRequest": expected_candidates,
            "requestCount": request_count, "expectedCredits": expected_credits,
            "maxRequests": max_requests, "maxCredits": max_credits}


def _state_for(plan: dict[str, Any]) -> dict[str, Any]:
    return {"version": STATE_VERSION, "executionManifest": plan["path"],
            "executionManifestSha256": plan["sha256"],
            "approvalAdmissionSha256": plan.get("approvalAdmissionSha256"),
            "approvalAuthorizationId": plan.get("approvalAuthorizationId"),
            "sourceDraftSha256": plan["sourceSha256"],
            "productionOverridesSha256": (
                plan["productionBinding"]["sha256"] if plan["productionBinding"] else None
            ),
            "maxRequests": plan["maxRequests"], "maxCredits": plan["maxCredits"],
            "expectedCredits": plan["expectedCredits"], "createdAt": _now(),
            "updatedAt": _now(), "walletQueried": False, "topUpAttempted": False,
            "items": {cue["id"]: {"status": "PLANNED", "postAttempts": 0,
                                       "requestSha256": _sha_bytes(_canonical(cue["payload"]))}
                      for cue in plan["cues"]}}


def _load_state(plan: dict[str, Any], path: Path, create: bool) -> dict[str, Any]:
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(f"cannot read durable state: {exc}") from exc
    elif create:
        state = _state_for(plan)
        _atomic_json(path, state)
    else:
        raise PlanError(f"durable state does not exist: {path}")
    expected = {x["id"] for x in plan["cues"]}
    if state.get("version") != STATE_VERSION or set(state.get("items", {})) != expected:
        raise PlanError("durable state schema/item set mismatch")
    checks = {"executionManifestSha256": plan["sha256"],
              "approvalAdmissionSha256": plan.get("approvalAdmissionSha256"),
              "approvalAuthorizationId": plan.get("approvalAuthorizationId"),
              "sourceDraftSha256": plan["sourceSha256"],
              "productionOverridesSha256": (
                  plan["productionBinding"]["sha256"] if plan["productionBinding"] else None
              ),
              "maxRequests": plan["maxRequests"], "maxCredits": plan["maxCredits"]}
    for key, value in checks.items():
        if state.get(key) != value:
            raise PlanError(f"durable state mismatch: {key}")
    return state


def _request_json(method: str, url: str, key: str,
                  payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=_canonical(payload) if payload is not None else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "User-Agent": "agentic-animation-finite-score/1"}, method=method)
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read())
    if result.get("code") != 200 or not isinstance(result.get("data"), dict):
        raise RuntimeError("provider returned a non-success envelope")
    return result["data"]


def _create(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if payload["model"] == "V6":
        names = {"customMode": "custom_mode", "negativeTags": "negative_tags"}
        body = {"model": "ai-music-api/generate", "callBackUrl": payload["callBackUrl"],
                "input": {names.get(k, k): v for k, v in payload.items() if k != "callBackUrl"}}
        return _request_json("POST", API_ROOT + "jobs/createTask", key, body)
    return _request_json("POST", API_ROOT + "generate", key, payload)


def _record(task_id: str, key: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"taskId": task_id})
    return _request_json("GET", API_ROOT + "generate/record-info?" + query, key)


def _record_v6(task_id: str, key: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"taskId": task_id})
    raw = _request_json("GET", API_ROOT + "jobs/recordInfo?" + query, key)
    state = str(raw.get("state", "")).lower()
    result = {"status": {"success": "SUCCESS", "fail": "FAILED"}.get(state, "PENDING"),
              "providerRecord": raw}
    if state == "success":
        payload = raw.get("resultJson") or "{}"
        data = json.loads(payload) if isinstance(payload, str) else payload
        tracks = data.get("data")
        if isinstance(tracks, list) and tracks:
            if any(not isinstance(t, dict) or not isinstance(t.get("audio_url"), str)
                   or not t["audio_url"] for t in tracks):
                raise PlanError("V6 candidate is missing audio_url; retain task for reconciliation")
            result["response"] = {"sunoData": [{**t, "audioUrl": t["audio_url"]} for t in tracks]}
            return result
        urls = data.get("resultUrls") or []
        if not isinstance(urls, list) or not urls or any(not isinstance(u, str) for u in urls):
            raise PlanError("V6 success lacks valid resultUrls; retain task for reconciliation")
        result["response"] = {"sunoData": [{"audioUrl": u, "id": f"{task_id}-{i}"}
                                             for i, u in enumerate(urls, 1)]}
    return result


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _validate_mp3(path: Path, requested_seconds: int) -> dict[str, Any]:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of",
             "json", str(path)], check=True, capture_output=True, text=True)
        info = json.loads(probe.stdout)
        audio = [x for x in info.get("streams", []) if x.get("codec_type") == "audio"]
        duration = float((info.get("format") or {}).get("duration") or 0)
        minimum = max(5.0, requested_seconds * 0.25)
        if not audio or duration < minimum:
            raise PlanError("candidate has no non-trivial audio stream")
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                        "-f", "null", "-"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError, ValueError,
            json.JSONDecodeError) as exc:
        raise PlanError(f"candidate MP3 failed decode: {exc}") from exc
    return {"durationSeconds": duration, "audioStreams": len(audio),
            "codec": audio[0].get("codec_name"),
            "requestedDurationSeconds": requested_seconds,
            "minimumAcceptedDurationSeconds": minimum}


def _completed_ok(plan: dict[str, Any], cue_id: str,
                  record: dict[str, Any]) -> None:
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PlanError(f"{cue_id}: downloaded state has no candidates")
    for candidate in candidates:
        path = Path(candidate["file"])
        sidecar = Path(candidate["sidecar"])
        if not path.is_file() or _sha_file(path) != candidate.get("sha256"):
            raise PlanError(f"{cue_id}: candidate missing or hash changed: {path}")
        if not sidecar.is_file() or _sha_file(sidecar) != candidate.get("sidecarSha256"):
            raise PlanError(f"{cue_id}: sidecar missing or hash changed: {sidecar}")
        try:
            source = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(f"{cue_id}: sidecar is invalid: {sidecar}") from exc
        expected = {
            "cueId": cue_id,
            "executionManifestSha256": plan["sha256"],
            "sourceDraftSha256": plan["sourceSha256"],
            "sha256": candidate.get("sha256"),
        }
        for key, value in expected.items():
            if source.get(key) != value:
                raise PlanError(f"{cue_id}: sidecar provenance mismatch: {key}")


def _preflight_output_namespace(plan: dict[str, Any], state: dict[str, Any]) -> None:
    """Reject every stale/colliding target before a paid POST or provider poll."""
    output_root = Path(plan["outputRoot"])
    for cue in plan["cues"]:
        cue_id = cue["id"]
        record = state["items"][cue_id]
        existing = set(output_root.glob(f"{cue_id}-take-*")) if output_root.exists() else set()
        if record.get("status") == "DOWNLOADED":
            _completed_ok(plan, cue_id, record)
            allowed = {
                Path(value)
                for candidate in record.get("candidates", [])
                for value in (candidate.get("file"), candidate.get("sidecar"))
                if value
            }
            unexpected = {path.resolve() for path in existing} - {path.resolve() for path in allowed}
            if unexpected:
                raise PlanError(f"{cue_id}: untracked output namespace entries: {sorted(map(str, unexpected))}")
            continue
        if existing:
            raise PlanError(
                f"{cue_id}: output namespace is not empty before paid work: "
                f"{sorted(map(str, existing))}"
            )


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PlanError(f"execution lock exists; verify no runner is active: {path}") from exc
    os.write(fd, f"pid={os.getpid()}\n".encode())
    os.fsync(fd)
    os.close(fd)
    return fd


def submit(plan: dict[str, Any], state_path: Path, key: str,
           create_fn: Callable[[dict[str, Any], str], dict[str, Any]]) -> dict[str, Any]:
    if not plan.get("financialAuthorized", plan["doc"].get("financialApproval") is True):
        raise PlanError("financialApproval is false; submission is gated")
    if plan["doc"].get("currentPriceVerified") is not True:
        raise PlanError("currentPriceVerified is false; credit ceiling is not authorized")
    state = _load_state(plan, state_path, create=True)
    _preflight_output_namespace(plan, state)
    calls = 0
    for cue in plan["cues"]:
        record = state["items"][cue["id"]]
        status = record.get("status")
        if status == "DOWNLOADED":
            _completed_ok(plan, cue["id"], record)
            continue
        if status == "SUBMITTED" or status == "PENDING" or status == "RESULT_READY":
            continue
        if status != "PLANNED":
            raise UncertainRequest(f"{cue['id']}: {status}; generation POST will not be retried")
        record.update(status="POSTING_UNCERTAIN", postAttempts=1, postStartedAt=_now())
        _persist(state_path, state)
        calls += 1
        try:
            response = create_fn(cue["payload"], key)
        except urllib.error.HTTPError as exc:
            status = "HTTP_4XX_REFUSED" if 400 <= exc.code < 500 else "UNCERTAIN_HTTP_ERROR"
            record.update(status=status, httpStatus=exc.code, postFinishedAt=_now())
            _persist(state_path, state)
            error = PlanError if status == "HTTP_4XX_REFUSED" else UncertainRequest
            raise error(f"{cue['id']}: POST failed HTTP {exc.code}; automatic retry forbidden") from exc
        except Exception as exc:
            record.update(status="UNCERTAIN_POST", postFinishedAt=_now(),
                          error=f"{type(exc).__name__}: {exc}")
            _persist(state_path, state)
            raise UncertainRequest(f"{cue['id']}: ambiguous POST; reconcile task logs") from exc
        task_id = response.get("taskId")
        if not isinstance(task_id, str) or not task_id:
            record.update(status="UNCERTAIN_RESPONSE", postFinishedAt=_now())
            _persist(state_path, state)
            raise UncertainRequest(f"{cue['id']}: success response lacked taskId")
        record.update(status="SUBMITTED", taskId=task_id, postFinishedAt=_now(),
                      providerCreateResponse=response)
        _persist(state_path, state)
    return {"operation": "SUBMIT", "generationPosts": calls,
            "state": str(state_path), "expectedCredits": plan["expectedCredits"]}


def _tracks(result: dict[str, Any]) -> list[dict[str, Any]]:
    tracks = (result.get("response") or {}).get("sunoData") or []
    if not isinstance(tracks, list) or not tracks:
        raise PlanError("successful task contains no Suno candidates")
    if any(not isinstance(x, dict) or not isinstance(x.get("audioUrl"), str)
           or not x["audioUrl"] for x in tracks):
        raise PlanError("provider candidate is missing audioUrl")
    return tracks


def poll(plan: dict[str, Any], state_path: Path, key: str,
         record_fn: Callable[[str, str], dict[str, Any]],
         download_fn: Callable[[str], bytes]) -> dict[str, Any]:
    state = _load_state(plan, state_path, create=False)
    _preflight_output_namespace(plan, state)
    output_root = Path(plan["outputRoot"])
    polls = downloads = 0
    for cue in plan["cues"]:
        item = state["items"][cue["id"]]
        status = item.get("status")
        if status == "GENERATION_FAILED" and item.get("lastProviderStatus") in IN_PROGRESS:
            # Repair only the former poller's known classification error. The
            # original task ID and provider response survive; no POST is made.
            item["status"] = status = "PENDING"
            item["statusCorrection"] = "provider intermediate success was misclassified as failure"
            _persist(state_path, state)
        if status == "DOWNLOADED":
            _completed_ok(plan, cue["id"], item)
            continue
        if status in {"PLANNED", "POSTING_UNCERTAIN", "UNCERTAIN_POST",
                      "UNCERTAIN_RESPONSE", "UNCERTAIN_HTTP_ERROR",
                      "HTTP_4XX_REFUSED", "GENERATION_FAILED"}:
            raise UncertainRequest(f"{cue['id']}: cannot poll state {status}")
        if status in {"SUBMITTED", "PENDING", "POLL_FAILED_RETRYABLE"}:
            try:
                result = record_fn(item["taskId"], key)
                polls += 1
            except Exception as exc:
                item.update(status="POLL_FAILED_RETRYABLE",
                            pollError=f"{type(exc).__name__}: {exc}")
                _persist(state_path, state)
                raise PlanError(f"{cue['id']}: poll failed; current run stopped") from exc
            provider_status = str(result.get("status") or "").upper()
            item.update(lastProviderResponse=result, lastProviderStatus=provider_status,
                        lastPolledAt=_now())
            if provider_status in IN_PROGRESS:
                item["status"] = "PENDING"
                _persist(state_path, state)
                continue
            if provider_status not in SUCCESS:
                item["status"] = "GENERATION_FAILED"
                _persist(state_path, state)
                raise PlanError(f"{cue['id']}: provider generation failed: {provider_status}")
            tracks = _tracks(result)
            item.update(status="RESULT_READY", providerCandidates=tracks,
                        candidates=[])
            _persist(state_path, state)
        if item.get("status") != "RESULT_READY":
            raise PlanError(f"{cue['id']}: unsupported poll state {item.get('status')}")
        existing = {x["index"]: x for x in item.get("candidates", [])}
        for index, track in enumerate(item["providerCandidates"], 1):
            if index in existing:
                candidate = existing[index]
                if candidate.get("status") == "COMPLETE":
                    continue
                raise PlanError(f"{cue['id']} take {index}: incomplete capture needs reconciliation")
            try:
                raw = download_fn(track["audioUrl"])
                downloads += 1
            except Exception as exc:
                item["downloadError"] = f"take {index}: {type(exc).__name__}: {exc}"
                _persist(state_path, state)
                raise PlanError(f"{cue['id']} take {index}: download failed; run stopped") from exc
            response_sha = _sha_bytes(raw)
            response_root = state_path.parent / f".{state_path.stem}-responses"
            staged = response_root / f"{cue['id']}-take-{index}-{response_sha[:16]}.mp3"
            if not staged.exists():
                _atomic_bytes_new(staged, raw)
            candidate = {"index": index, "status": "CAPTURED", "responseSha256": response_sha,
                         "staged": str(staged), "providerMetadata": track}
            item.setdefault("candidates", []).append(candidate)
            _persist(state_path, state)
            media = _validate_mp3(staged, int(cue["payload"]["duration"]))
            candidate.update(status="VALIDATED", media=media)
            _persist(state_path, state)
            output = output_root / f"{cue['id']}-take-{index}.mp3"
            sidecar = output.with_suffix(".source.json")
            if output.exists() or sidecar.exists():
                raise PlanError(f"refusing to overwrite candidate or sidecar: {output}")
            _atomic_bytes_new(output, raw)
            sidecar_doc = {"version": 1, "episode": plan["doc"].get("episode"),
                           "cueId": cue["id"], "candidateIndex": index,
                           "taskId": item["taskId"], "request": cue["payload"],
                           "requestSha256": item["requestSha256"],
                           "executionManifest": plan["path"],
                           "executionManifestSha256": plan["sha256"],
                           "approvalAdmissionSha256": plan.get("approvalAdmissionSha256"),
                           "approvalAuthorizationId": plan.get("approvalAuthorizationId"),
                           "sourceDraftSha256": plan["sourceSha256"],
                           "productionOverridesSha256": (
                               plan["productionBinding"]["sha256"]
                               if plan["productionBinding"] else None
                           ),
                           "providerMetadata": track, "media": media,
                           "file": str(output), "sha256": response_sha,
                           "selectionVerdict": "unreviewed"}
            _atomic_bytes_new(
                sidecar,
                (json.dumps(sidecar_doc, ensure_ascii=False, sort_keys=True, indent=2)
                 + "\n").encode("utf-8"),
            )
            candidate.update(status="COMPLETE", file=str(output), sha256=response_sha,
                             sidecar=str(sidecar), sidecarSha256=_sha_file(sidecar))
            _persist(state_path, state)
        item["status"] = "DOWNLOADED"
        item["downloadedAt"] = _now()
        _persist(state_path, state)
    return {"operation": "POLL", "recordGets": polls, "candidateDownloads": downloads,
            "state": str(state_path)}


def execute(manifest_path: Path, max_requests: int, max_credits: int,
            operation: str = "dry-run", state_path: Path | None = None,
            create_fn: Callable | None = None, record_fn: Callable | None = None,
            download_fn: Callable | None = None, api_key: str | None = None,
            approval_admission: Path | None = None) -> dict[str, Any]:
    plan = load_plan(manifest_path, max_requests, max_credits)
    try:
        approval = validate_paid_approval(
            approval_admission, episode=plan["doc"].get("episode"), lane="suno-score",
            manifest_path=Path(plan["path"]), manifest_sha256=plan["sha256"],
            finite_execution={"requests": plan["requestCount"],
                              "credits": plan["expectedCredits"],
                              "sourceSeconds": sum(x["payload"]["duration"] for x in plan["cues"])},
        )
    except PaidApprovalError as exc:
        raise PlanError(str(exc)) from exc
    plan["financialAuthorized"] = plan["doc"].get("financialApproval") is True or approval is not None
    plan["approvalAdmissionSha256"] = approval["sha256"] if approval else None
    plan["approvalAuthorizationId"] = approval["authorizationId"] if approval else None
    if operation == "dry-run":
        return {"operation": "DRY_RUN", "networkCalls": 0,
                "financialApproval": plan["doc"].get("financialApproval") is True,
                "financialAuthorized": plan["financialAuthorized"],
                "approvalAdmissionSha256": plan["approvalAdmissionSha256"],
                "requests": plan["requestCount"], "expectedCredits": plan["expectedCredits"],
                "maxRequests": max_requests, "maxCredits": max_credits,
                "sourceSeconds": sum(x["payload"]["duration"] for x in plan["cues"])}
    if operation not in {"submit", "poll"}:
        raise PlanError(f"unknown operation: {operation}")
    if operation == "submit" and not plan["financialAuthorized"]:
        # Refuse at the immutable plan gate before looking for a credential,
        # creating state, acquiring a lock, or touching the network.
        raise PlanError("financialApproval is false; submission is gated")
    if operation == "submit" and plan["doc"].get("currentPriceVerified") is not True:
        raise PlanError("currentPriceVerified is false; credit ceiling is not authorized")
    default_state = Path(plan["path"]).with_suffix(".state.json").resolve()
    state_path = (state_path or default_state).resolve()
    if state_path != default_state:
        raise PlanError(f"custom state paths are forbidden; use canonical {default_state}")
    lock = Path(plan["path"]).with_suffix(".execution.lock")
    _acquire_lock(lock)
    try:
        try:
            key = api_key if api_key is not None else env_key("KIE_KEY")
        except (SystemExit, OSError, ValueError) as exc:
            raise PlanError(str(exc)) from exc
        if not key:
            raise PlanError("KIE_KEY is missing from the explicit credential sources")
        if operation == "submit":
            return submit(plan, state_path, key, create_fn or _create)
        models = {cue["payload"]["model"] for cue in plan["cues"]}
        if len(models) != 1:
            raise PlanError("Each score batch must use one model for unambiguous recovery")
        return poll(plan, state_path, key, record_fn or (_record_v6 if models == {"V6"} else _record),
                    download_fn or _download)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-credits", type=int, required=True)
    parser.add_argument("--approval-admission", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--poll", action="store_true")
    args = parser.parse_args(argv)
    operation = "submit" if args.submit else "poll" if args.poll else "dry-run"
    try:
        result = execute(args.manifest, args.max_requests, args.max_credits,
                         operation, approval_admission=args.approval_admission)
    except (PlanError, UncertainRequest) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
