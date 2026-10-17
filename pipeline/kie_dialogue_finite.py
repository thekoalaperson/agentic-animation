#!/usr/bin/env python3
"""Run a source-bound Hindi Grok dialogue batch on a GPU worker; dry-run by default.

Project manifests own
content; request hashes and kie_video recovery prevent duplicate paid POSTs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import platform
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.paid_approval import PaidApprovalError, validate_paid_approval
from pipeline._common import canonical_sha256, image_dimensions, project_root, sha256_file as sha256
HERE = EPISODE = MANIFEST = DRAFT = NEW_ANGLE_AUDIT = None
PRODUCTION_OVERRIDES = PROMPT_ROOT = OUTPUT_ROOT = LOCK = AUTHORITY = None
EPISODE_CODE = None
ALLOWED_DURATIONS = set(range(1, 16))
ID = None
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
DEVANAGARI = re.compile(r"[\u0900-\u097f]")


def configure(manifest_path: Path, episode_path: Path,
              production_overrides: Path | None = None,
              episode_code: str | None = None,
              authority_path: Path | None = None) -> None:
    global HERE, EPISODE, MANIFEST, DRAFT, NEW_ANGLE_AUDIT, PRODUCTION_OVERRIDES
    global PROMPT_ROOT, OUTPUT_ROOT, LOCK, EPISODE_CODE, ID, AUTHORITY
    try:
        EPISODE = project_root(episode_path)
    except ValueError as exc:
        raise PlanError(str(exc)) from exc
    declared = read_object(manifest_path.resolve(), "execution manifest").get("episode")
    EPISODE_CODE = episode_code or declared
    if (not isinstance(EPISODE_CODE, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", EPISODE_CODE)):
        raise PlanError("project code must be a safe non-empty identifier")
    if declared != EPISODE_CODE:
        raise PlanError("project code conflicts with the execution manifest")
    board_path = EPISODE / "story/storyboard_plan.json"
    if board_path.is_file() and read_object(board_path, "storyboard").get("episode") != EPISODE_CODE:
        raise PlanError("project code must match the storyboard")
    ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
    MANIFEST = contained(manifest_path, EPISODE / "work", "manifest")
    HERE = MANIFEST.parent
    DRAFT = HERE / "draft-manifest.json"
    NEW_ANGLE_AUDIT = HERE / "source-audit.json"
    PRODUCTION_OVERRIDES = contained(
        production_overrides or EPISODE / "story/production-overrides-v1.json",
        EPISODE / "story", "production overrides")
    AUTHORITY = contained(authority_path or EPISODE / "story/GOAL.md",
                          EPISODE / "story", "authority")
    PROMPT_ROOT = HERE / "prompts"
    OUTPUT_ROOT = HERE / "motion"
    LOCK = OUTPUT_ROOT / ".grok-finite.lock"

ROOT_FIELDS = {
    "version", "episode", "provider", "route", "model", "resolution",
    "aspectRatio", "language", "financialApproval", "financialState",
    "draftManifest", "draftManifestSha256", "newAngleAudit",
    "newAngleAuditSha256", "productionOverrides",
    "productionOverridesSha256", "jobCount", "maximumGeneratedSeconds",
    "jobs", "authority", "authoritySha256", "pricing", "maximumCredits",
}
JOB_FIELDS = {
    "id", "speaker", "hindi", "durationSeconds", "source", "sourceSha256",
    "prompt", "promptSha256", "outputStem",
}


class PlanError(ValueError):
    """The frozen execution plan or local recovery state is unsafe."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as target:
            target.write((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     indent=2) + "\n").encode("utf-8"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError(f"{label} must be a JSON object: {path}")
    return value


def contained(path: Path, root: Path, label: str) -> Path:
    path = path.resolve()
    root = root.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PlanError(f"{label} escapes {root}: {path}") from exc
    return path


def project_path(value: Any, label: str, within: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise PlanError(f"{label} must be a non-empty project-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PlanError(f"{label} must be a contained project-relative path")
    return contained(EPISODE / relative, within, label)


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise PlanError(f"{label} must be a lowercase SHA-256 digest")
    return value


def execution_request(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "jobId": row["id"],
        "executionManifestSha256": plan["manifestSha256"],
        "model": "grok-imagine-video-1-5-preview",
        "sourceSha256": row["sourceSha256"],
        "promptSha256": row["promptSha256"],
        "durationSeconds": row["durationSeconds"],
        "resolution": "720p",
        "aspectRatio": "16:9",
        "audio": True,
        "outputStem": row["outputStem"],
    }


def source_geometry_warnings(rows: list[dict[str, Any]], aspect: float = 16 / 9,
                             tolerance: float = 0.03) -> list[dict[str, Any]]:
    """Warn, never refuse, when a first-frame still is far from the sent aspect.

    The runner always sends aspect_ratio 16:9. Whether output geometry follows
    the still on this route is unqualified, so this stays information until a
    pilot settles it. Dimensions come from the file header; nothing is decoded.
    """
    warnings = []
    for row in rows:
        try:
            _, width, height = image_dimensions(row["sourcePath"])
        except (OSError, ValueError):
            continue
        if height and abs(width / height - aspect) > tolerance:
            warnings.append({
                "type": "source-aspect", "jobId": row["id"], "width": width, "height": height,
                "aspect": round(width / height, 4), "sentAspectRatio": "16:9",
                "hint": "pipeline/derive_still.py can crop, pad or scale a registered still to 16:9",
            })
    return warnings


def load_plan(manifest_path: Path, max_generated_seconds: int) -> dict[str, Any]:
    if (isinstance(max_generated_seconds, bool)
            or not isinstance(max_generated_seconds, int)
            or max_generated_seconds <= 0):
        raise PlanError("--max-generated-seconds must be a positive integer")
    manifest_path = manifest_path.resolve()
    doc = read_object(manifest_path, "execution manifest")
    if set(doc) != ROOT_FIELDS:
        missing = sorted(ROOT_FIELDS - set(doc))
        extra = sorted(set(doc) - ROOT_FIELDS)
        raise PlanError(f"execution manifest schema mismatch; missing={missing}, extra={extra}")
    expected_header = {
        "version": 2,
        "episode": EPISODE_CODE,
        "provider": "kie.ai",
        "route": "grok15",
        "model": "grok-imagine-video-1-5-preview",
        "resolution": "720p",
        "aspectRatio": "16:9",
        "language": "hi-IN",
    }
    for key, expected in expected_header.items():
        if doc.get(key) != expected:
            raise PlanError(f"execution manifest {key} must be {expected!r}")
    if not isinstance(doc.get("financialApproval"), bool):
        raise PlanError("financialApproval must be boolean")
    if not isinstance(doc.get("financialState"), str) or not doc["financialState"]:
        raise PlanError("financialState must be a non-empty string")

    pricing = doc.get("pricing")
    if not isinstance(pricing, dict) or set(pricing) != {"creditsPerSecond", "verifiedAtUtc", "evidence"}:
        raise PlanError("pricing must bind a verified rate, UTC date and evidence")
    rate = pricing["creditsPerSecond"]
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or rate <= 0):
        raise PlanError("creditsPerSecond must be positive and finite")
    if not isinstance(pricing["verifiedAtUtc"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", pricing["verifiedAtUtc"]):
        raise PlanError("pricing verification must have an explicit UTC timestamp")
    if not isinstance(pricing["evidence"], str) or not pricing["evidence"].strip():
        raise PlanError("pricing evidence is required")
    maximum_credits = doc.get("maximumCredits")
    if (isinstance(maximum_credits, bool) or not isinstance(maximum_credits, (int, float))
            or not math.isfinite(maximum_credits) or maximum_credits <= 0):
        raise PlanError("maximumCredits must be an explicit positive finite ceiling")

    provenance = (
        ("draftManifest", "draftManifestSha256", DRAFT),
        ("newAngleAudit", "newAngleAuditSha256", NEW_ANGLE_AUDIT),
        ("productionOverrides", "productionOverridesSha256", PRODUCTION_OVERRIDES),
        ("authority", "authoritySha256", AUTHORITY),
    )
    for path_key, hash_key, expected_path in provenance:
        path = project_path(doc[path_key], path_key, EPISODE)
        if path != expected_path.resolve():
            raise PlanError(f"{path_key} must resolve to {expected_path}")
        expected_hash = require_sha(doc[hash_key], hash_key)
        if not path.is_file() or sha256(path) != expected_hash:
            raise PlanError(f"{path_key} changed after this execution manifest was frozen")

    jobs = doc.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise PlanError("jobs must be a non-empty array")
    ids: set[str] = set()
    stems: set[Path] = set()
    normalized: list[dict[str, Any]] = []
    total = 0
    for position, raw in enumerate(jobs):
        if not isinstance(raw, dict) or set(raw) != JOB_FIELDS:
            raise PlanError(f"job {position} fields must be exactly {sorted(JOB_FIELDS)}")
        job_id = raw.get("id")
        if not isinstance(job_id, str) or not ID.fullmatch(job_id) or job_id in ids:
            raise PlanError(f"job {position} has duplicate or malformed id: {job_id!r}")
        ids.add(job_id)
        if not isinstance(raw.get("speaker"), str) or not raw["speaker"].strip():
            raise PlanError(f"{job_id}: speaker must be non-empty")
        if (not isinstance(raw.get("hindi"), str) or not raw["hindi"].strip()
                or not DEVANAGARI.search(raw["hindi"])):
            raise PlanError(f"{job_id}: hindi must be non-empty Devanagari dialogue")
        duration = raw.get("durationSeconds")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration not in ALLOWED_DURATIONS:
            raise PlanError(f"{job_id}: durationSeconds must be one of {sorted(ALLOWED_DURATIONS)}")
        source = project_path(raw["source"], f"{job_id}.source", EPISODE)
        prompt = project_path(raw["prompt"], f"{job_id}.prompt", PROMPT_ROOT)
        stem = project_path(raw["outputStem"], f"{job_id}.outputStem", OUTPUT_ROOT)
        if stem.parent != OUTPUT_ROOT.resolve() or stem.name != job_id:
            raise PlanError(f"{job_id}: outputStem must be motion/{job_id}")
        if stem in stems:
            raise PlanError(f"duplicate output stem: {stem}")
        stems.add(stem)
        source_hash = require_sha(raw["sourceSha256"], f"{job_id}.sourceSha256")
        prompt_hash = require_sha(raw["promptSha256"], f"{job_id}.promptSha256")
        if not source.is_file() or sha256(source) != source_hash:
            raise PlanError(f"{job_id}: source missing or hash changed")
        if not prompt.is_file() or sha256(prompt) != prompt_hash:
            raise PlanError(f"{job_id}: prompt missing or hash changed")
        prompt_text = prompt.read_text(encoding="utf-8").strip()
        if raw["hindi"] not in prompt_text or len(prompt_text) > 4096:
            raise PlanError(f"{job_id}: exact Hindi missing or provider prompt limit exceeded")
        if source.stat().st_size > 20 * 1024 * 1024:
            raise PlanError(f"{job_id}: source exceeds provider image limit")
        row = dict(raw)
        row.update(sourcePath=source, promptPath=prompt, outputPath=stem)
        normalized.append(row)
        total += duration

    if (isinstance(doc.get("jobCount"), bool) or doc.get("jobCount") != len(jobs)
            or isinstance(doc.get("maximumGeneratedSeconds"), bool)
            or doc.get("maximumGeneratedSeconds") != total):
        raise PlanError("execution manifest job/seconds aggregate mismatch")
    if max_generated_seconds != total:
        raise PlanError(
            f"--max-generated-seconds must exactly equal frozen/computed total "
            f"{total}s; received {max_generated_seconds}s")
    if total * rate > maximum_credits:
        raise PlanError("verified price exceeds the manifest's maximumCredits")
    plan = {
        "manifestPath": manifest_path,
        "manifestSha256": sha256(manifest_path),
        "financialApproval": doc["financialApproval"],
        "financialState": doc["financialState"],
        "jobs": normalized,
        "totalSeconds": total,
        "maxGeneratedSeconds": max_generated_seconds,
        "maximumCredits": maximum_credits, "pricing": pricing,
        "geometryWarnings": source_geometry_warnings(normalized),
    }
    for row in normalized:
        row["executionRequestSha256"] = canonical_sha256(execution_request(plan, row))
    return plan


def validate_media(path: Path, expected_duration: int) -> dict[str, Any]:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of",
             "json", str(path)], check=True, capture_output=True, text=True)
        info = json.loads(probe.stdout)
        streams = info.get("streams", [])
        video = [stream for stream in streams if stream.get("codec_type") == "video"]
        audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if not video or not audio:
            raise PlanError(f"video must contain picture and native dialogue audio: {path}")
        duration = float((info.get("format") or {}).get("duration") or 0)
        if not expected_duration * 0.8 <= duration <= expected_duration * 1.25:
            raise PlanError(
                f"duration {duration:.3f}s does not match requested {expected_duration}s: {path}")
        width = int(video[0].get("width") or 0)
        height = int(video[0].get("height") or 0)
        if width < 1200 or height < 675 or abs(width / max(height, 1) - 16 / 9) > 0.03:
            raise PlanError(f"video is not a valid 720p-class 16:9 result: {path}")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
             "-map", "0:a:0", "-f", "null", "-"], check=True,
            capture_output=True)
    except PlanError:
        raise
    except (OSError, subprocess.CalledProcessError, ValueError,
            json.JSONDecodeError) as exc:
        raise PlanError(f"media decode validation failed for {path}: {exc}") from exc
    return {
        "durationSeconds": duration,
        "videoStreams": len(video),
        "audioStreams": len(audio),
        "width": width,
        "height": height,
        "sha256": sha256(path),
    }


def validate_request_envelope(record: dict[str, Any], expected: dict[str, Any],
                              label: str) -> tuple[dict[str, Any], str]:
    fingerprint = record.get("requestFingerprint")
    request = record.get("request")
    if (not isinstance(fingerprint, str) or not SHA256.fullmatch(fingerprint)
            or record.get("requestSha256") != fingerprint
            or not isinstance(request, dict)
            or canonical_sha256(request) != fingerprint):
        raise PlanError(f"{label}: canonical request fingerprint is invalid")
    request_checks = {
        "model": expected["model"],
        "sourceSha256": expected["sourceSha256"],
        "promptSha256": expected["promptSha256"],
        "executionManifestSha256": expected["executionManifestSha256"],
        "executionRequestSha256": expected["executionRequestSha256"],
    }
    for key, value in request_checks.items():
        if request.get(key) != value:
            raise PlanError(f"{label}: request {key} does not match the frozen job")
    provider_input = request.get("input")
    if not isinstance(provider_input, dict):
        raise PlanError(f"{label}: request input is missing")
    if (provider_input.get("prompt") != expected["promptText"]
            or provider_input.get("resolution") != expected["resolution"]
            or provider_input.get("aspect_ratio") != expected["aspectRatio"]
            or provider_input.get("duration") != expected["durationSeconds"]):
        raise PlanError(f"{label}: provider input differs from the frozen job")
    return request, fingerprint


def validate_local_envelope(record: dict[str, Any], expected: dict[str, Any],
                            label: str) -> None:
    local = record.get("localRequest")
    local_sha = record.get("localRequestSha256")
    if (not isinstance(local, dict) or not isinstance(local_sha, str)
            or not SHA256.fullmatch(local_sha)
            or canonical_sha256(local) != local_sha):
        raise PlanError(f"{label}: local request fingerprint is invalid")
    checks = {
        "model": expected["model"],
        "jobId": expected["jobId"],
        "promptSha256": expected["promptSha256"],
        "sourceSha256": expected["sourceSha256"],
        "executionManifestSha256": expected["executionManifestSha256"],
        "executionRequestSha256": expected["executionRequestSha256"],
        "resolution": expected["resolution"],
        "durationSeconds": expected["durationSeconds"],
        "aspectRatio": expected["aspectRatio"],
        "audio": expected["audio"],
    }
    for key, value in checks.items():
        if local.get(key) != value:
            raise PlanError(f"{label}: local request {key} differs from frozen job")
    input_hashes = local.get("inputHashes")
    if not isinstance(input_hashes, dict) or input_hashes.get("primary") != expected["sourceSha256"]:
        raise PlanError(f"{label}: local primary source hash is invalid")


def expected_record(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "grok-imagine-video-1-5-preview",
        "jobId": row["id"],
        "executionManifestSha256": plan["manifestSha256"],
        "executionRequestSha256": row["executionRequestSha256"],
        "sourceSha256": row["sourceSha256"],
        "promptSha256": row["promptSha256"],
        "durationSeconds": row["durationSeconds"],
        "resolution": "720p",
        "aspectRatio": "16:9",
        "audio": True,
        "promptText": row["promptPath"].read_text(encoding="utf-8").strip(),
    }


def validate_completed(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    stem = row["outputPath"]
    video = stem.with_suffix(".mp4")
    sidecar = stem.with_suffix(".vid.json")
    meta = read_object(sidecar, f"{row['id']} sidecar")
    expected = expected_record(plan, row)
    fields = {
        "provenanceVersion": 2,
        "provider": "kie.ai",
        **{key: value for key, value in expected.items() if key != "promptText"},
    }
    for key, value in fields.items():
        if meta.get(key) != value:
            raise PlanError(f"{row['id']}: sidecar {key} does not match frozen job")
    validate_local_envelope(meta, expected, row["id"])
    _, fingerprint = validate_request_envelope(meta, expected, row["id"])
    media = validate_media(video, row["durationSeconds"])
    if meta.get("outputSha256") != media["sha256"] or meta.get("sha256") != media["sha256"]:
        raise PlanError(f"{row['id']}: output bytes do not match the sidecar")
    task_id = meta.get("taskId")
    if not isinstance(task_id, str) or not task_id:
        raise PlanError(f"{row['id']}: sidecar lacks taskId")
    return {"id": row["id"], **media, "taskId": task_id,
            "requestFingerprint": fingerprint, "resumed": True}


def validate_pending(plan: dict[str, Any], row: dict[str, Any], path: Path,
                     allowed_statuses: set[str] | None = None) -> dict[str, Any]:
    record = read_object(path, f"{row['id']} pending recovery record")
    expected = expected_record(plan, row)
    fields = {
        "provenanceVersion": 2,
        "jobId": row["id"],
        "model": expected["model"],
        "executionManifestSha256": plan["manifestSha256"],
        "executionRequestSha256": row["executionRequestSha256"],
        "sourceSha256": row["sourceSha256"],
        "promptSha256": row["promptSha256"],
    }
    for key, value in fields.items():
        if record.get(key) != value:
            raise PlanError(f"{row['id']}: pending {key} does not match frozen job")
    validate_local_envelope(record, expected, row["id"])
    validate_request_envelope(record, expected, row["id"])
    task_id = record.get("taskId")
    allowed_statuses = allowed_statuses or {"PENDING"}
    if record.get("status") not in allowed_statuses or not isinstance(task_id, str) or not task_id:
        raise PlanError(
            f"{row['id']}: pending state {record.get('status')!r} is not auto-retryable; "
            "reconcile the provider taskId manually")
    return record


def preflight(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Inspect every destination/recovery record before the first paid subprocess."""
    pending: list[dict[str, Any]] = []
    completed: dict[str, dict[str, Any]] = {}
    for row in plan["jobs"]:
        stem = row["outputPath"]
        video = stem.with_suffix(".mp4")
        sidecar = stem.with_suffix(".vid.json")
        recovery = stem.with_suffix(".pending.json")
        if video.exists() != sidecar.exists():
            raise PlanError(f"{row['id']}: incomplete landed pair requires reconciliation")
        if video.exists():
            completed[row["id"]] = validate_completed(plan, row)
            if recovery.exists():
                record = validate_pending(
                    plan, row, recovery, {"PENDING", "LANDED"})
                if (record.get("taskId") != completed[row["id"]]["taskId"]
                        or record.get("requestFingerprint")
                        != completed[row["id"]]["requestFingerprint"]):
                    raise PlanError(f"{row['id']}: landed pair conflicts with pending paid task")
                recovery.unlink()
            continue
        if recovery.exists():
            validate_pending(plan, row, recovery)
        pending.append(row)
    return pending, completed


def acquire_lock(path: Path, manifest_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PlanError(f"canonical output lock exists; verify no runner is active: {path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} manifest={manifest_sha256}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute(max_generated_seconds: int, submit: bool,
            approval_admission: Path | None = None, concurrency: int = 1) -> dict[str, Any]:
    plan = load_plan(MANIFEST, max_generated_seconds)
    try:
        approval = validate_paid_approval(
            approval_admission, episode=EPISODE_CODE, lane="grok-dialogue",
            manifest_path=plan["manifestPath"], manifest_sha256=plan["manifestSha256"],
            finite_execution={"jobs": len(plan["jobs"]), "generatedSeconds": plan["totalSeconds"],
                              "credits": plan["maximumCredits"]},
        )
    except PaidApprovalError as exc:
        raise PlanError(str(exc)) from exc
    authorized = plan["financialApproval"] or approval is not None
    if not submit:
        return {
            "mode": "DRY_RUN",
            "networkCalls": 0,
            "jobs": len(plan["jobs"]),
            "generatedSeconds": plan["totalSeconds"],
            "capSeconds": plan["maxGeneratedSeconds"],
            "maximumCredits": plan["maximumCredits"],
            "financialApproval": plan["financialApproval"],
            "financialAuthorized": authorized,
            "approvalAdmissionSha256": approval["sha256"] if approval else None,
            "manifestSha256": plan["manifestSha256"],
            "geometryWarnings": plan["geometryWarnings"],
        }
    if not authorized:
        raise PlanError("financialApproval is false; submission is gated")

    if platform.system() != "Linux":
        raise PlanError("Submission and heavy media validation require the qualified GPU worker")
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, capture_output=True)
    if concurrency not in (1, 2, 3):
        raise PlanError("concurrency must be 1, 2 or 3")
    acquire_lock(LOCK, plan["manifestSha256"])
    try:
        jobs, completed = preflight(plan)
        authorization_path = HERE / "execution-authorization.json"
        evidence = {"executionManifestSha256": plan["manifestSha256"], "approval": approval}
        if authorization_path.exists():
            if read_object(authorization_path, "execution authorization") != evidence:
                raise PlanError("execution authorization differs from the existing batch")
        else:
            atomic_json(authorization_path, evidence)
        sys.path.insert(0, str(ROOT / "pipeline"))
        from _common import kie
        wallet_before = float(kie("chat/credit"))
        remaining_seconds = sum(row["durationSeconds"] for row in jobs
                                if not row["outputPath"].with_suffix(".pending.json").exists())
        estimated_credits = remaining_seconds * plan["pricing"]["creditsPerSecond"]
        if wallet_before < estimated_credits:
            raise PlanError("Insufficient existing Kie credits; no top-up or overage")
        atomic_json(HERE / "credit-preflight.json", {
            "manifestSha256": plan["manifestSha256"], "availableCredits": wallet_before,
            "estimatedNewCredits": estimated_credits, "pricing": plan["pricing"],
            "maximumCredits": plan["maximumCredits"],
            "concurrency": concurrency, "noTopup": True,
        })
        def generate_row(row):
            stem = row["outputPath"]
            command = [
                sys.executable,
                str(ROOT / "pipeline/kie_video.py"),
                "grok15",
                str(stem),
                str(row["promptPath"]),
                str(row["sourcePath"]),
                "--res", "720p",
                "--dur", str(row["durationSeconds"]),
                "--ar", "16:9",
                "--job-id", row["id"],
                "--manifest-sha256", plan["manifestSha256"],
                "--source-sha256", row["sourceSha256"],
                "--execution-request-sha256", row["executionRequestSha256"],
            ]
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                recovery = stem.with_suffix(".pending.json")
                state = ("durable recovery state exists" if recovery.exists()
                         else "no recovery state was recorded")
                raise PlanError(f"{row['id']} failed ({state}); stopped before later jobs")
            result = validate_completed(plan, row)
            result["resumed"] = False
            return result
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for offset in range(0, len(jobs), concurrency):
                group = jobs[offset:offset + concurrency]
                futures = [pool.submit(generate_row, row) for row in group]
                errors = []
                for row, future in zip(group, futures):
                    try:
                        completed[row["id"]] = future.result()
                        atomic_json(HERE / "progress.json", {
                            "manifestSha256": plan["manifestSha256"],
                            "completed": list(completed.values()),
                            "jobCount": len(plan["jobs"]),
                        })
                        print(f"TECHNICAL COMPLETE {len(completed)}/{len(plan['jobs'])}: {row['id']}", flush=True)
                    except Exception as exc:
                        errors.append(f"{row['id']}: {exc}")
                if errors:
                    raise PlanError("; ".join(errors))
        ordered = [completed[row["id"]] for row in plan["jobs"]]
        audit = {
            "episode": EPISODE_CODE,
            "executionManifestSha256": plan["manifestSha256"],
            "approvalAdmissionSha256": approval["sha256"] if approval else None,
            "status": "technical-complete-creative-review-required",
            "generatedSecondsCap": max_generated_seconds,
            "count": len(ordered),
            "items": ordered,
            "walletBefore": wallet_before,
            "walletAfter": float(kie("chat/credit")),
            "accountingNote": "Shared wallet deltas are not per-task billing attribution",
        }
        audit_path = HERE / "technical-audit.json"
        atomic_json(audit_path, audit)
        return {"mode": "SUBMIT", "networkCalls": len(jobs),
                "status": "COMPLETED", "audit": str(audit_path),
                "items": len(ordered)}
    finally:
        LOCK.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project", "--episode-root", dest="episode_root", type=Path,
                        help="Private project path (or AGENTIC_PROJECT_ROOT)")
    parser.add_argument("--episode-code",
                        help="Project code; defaults to the frozen manifest identifier")
    parser.add_argument("--authority-file", type=Path,
                        help="Explicit production authority inside this episode's story directory")
    parser.add_argument("--production-overrides", type=Path,
                        help="Explicit versioned script overrides inside this episode's story directory")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--credential-stdin", action="store_true")
    parser.add_argument("--max-generated-seconds", type=int, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--approval-admission", type=Path)
    args = parser.parse_args(argv)
    try:
        configure(args.manifest, args.episode_root, args.production_overrides,
                  args.episode_code, args.authority_file)
        if args.submit and args.credential_stdin:
            credential = json.load(sys.stdin)
            key = credential.get("kieApiKey")
            if not isinstance(key, str) or not key:
                raise PlanError("Missing Kie credential on stdin")
            os.environ["KIE_KEY"] = key
        result = execute(args.max_generated_seconds, args.submit, args.approval_admission, args.concurrency)
    except PlanError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
