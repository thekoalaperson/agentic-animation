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
from pipeline._common import (adopt_legacy_record, attempt_records, canonical_sha256, project_root,
                              sha256_file as sha256, usd_per_credit, write_attempt_json)
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
# Optional root fields: absent, a manifest validates exactly as before.
OPTIONAL_ROOT_FIELDS = {"siblingManifests"}
APPROVAL_LANE = "grok-dialogue"
FIXED_HEADER = {
    "version": 2,
    "provider": "kie.ai",
    "route": "grok15",
    "model": "grok-imagine-video-1-5-preview",
    "resolution": "720p",
    "aspectRatio": "16:9",
    "language": "hi-IN",
}
PRICING_FIELDS = {"creditsPerSecond", "verifiedAtUtc", "evidence"}

# Version 3 reads the model, resolution and aspect ratio from the manifest,
# drawn from this table, and its pricing block names the model and resolution
# the rate was read for. Version 2 keeps FIXED_HEADER exactly. Values the
# provider publishes but no batch has run are labelled so; adding one needs a
# dated reading of the provider's page, never a guess.
V3_PRICING_FIELDS = PRICING_FIELDS | {"model", "resolution"}
REFERENCE_FIELDS = {"path", "sha256"}
ROUTE_VALUES = {
    "grok-imagine-video-1-5-preview": {
        "route": "grok15",
        "providerPage": "https://docs.kie.ai/market/grok-imagine/1-5-preview (read 2026-10-18)",
        "resolutions": {"720p": "exercised",
                        "480p": "provider-published, unexercised",
                        "1080p": "provider-published, unexercised"},
        "aspectRatios": {"16:9": "exercised",
                         "9:16": "provider-published, unexercised",
                         "1:1": "provider-published, unexercised",
                         "3:2": "provider-published, unexercised",
                         "2:3": "provider-published, unexercised"},
        # The page allows seven images; the first frame is one of them.
        "maxReferences": 6,
        "notes": [
            "The provider page says aspect_ratio is ignored for a single image, so a "
            "no-reference clip's geometry follows its still; a 4:3 film needs 4:3 stills.",
            "The provider page says only one image is supported at 1080p. References at "
            "1080p are reported by the dry run, not refused, until a pilot settles it.",
            "Whether image 1 still pins the first frame when references are sent, and how "
            "@image(n) prompt syntax behaves, are unqualified until a one-clip paid pilot.",
            "auto is not offered: the runner checks landed geometry against the frozen ratio.",
        ],
    },
}


def schema() -> dict[str, Any]:
    """The frozen execution manifest contract, read from the validator's own sets."""
    return {
        "manifest": "work/<batch>/execution-manifest.json",
        "rootFields": sorted(ROOT_FIELDS),
        "optionalRootFields": {
            "siblingManifests": "[{path, sha256}] of other frozen dialogue manifests in this "
                                "project; any shared job id or output stem is refused",
        },
        "fixedValues": {**FIXED_HEADER, "episode": "the project code (storyboard episode)"},
        "jobFields": sorted(JOB_FIELDS),
        "pricingFields": sorted(PRICING_FIELDS),
        "durationSeconds": sorted(ALLOWED_DURATIONS),
        "boundFiles": {
            "draftManifest": "draft-manifest.json beside the manifest",
            "newAngleAudit": "source-audit.json beside the manifest",
            "productionOverrides": "story/production-overrides-v1.json or --production-overrides",
            "authority": "story/GOAL.md or --authority-file",
        },
        "jobPaths": {"prompt": "prompts/ beside the manifest; must contain the exact line",
                     "outputStem": "motion/<id> beside the manifest",
                     "source": "any project file, hash-bound"},
        "aggregates": {"jobCount": "number of jobs",
                       "maximumGeneratedSeconds": "sum of durationSeconds",
                       "maximumCredits": "at least maximumGeneratedSeconds x creditsPerSecond"},
        "financialApproval": "boolean; reported, never sufficient: submission needs an approval record",
        "approvalLane": APPROVAL_LANE,
        "approvalScope": ["jobs", "generatedSeconds", "credits"],
        "version3": {
            "meaning": "model, resolution and aspectRatio are read from the manifest; version 2 "
                       "keeps fixedValues exactly",
            "fixedValues": {"provider": FIXED_HEADER["provider"],
                            "language": FIXED_HEADER["language"]},
            "allowedValues": ROUTE_VALUES,
            "pricingFields": sorted(V3_PRICING_FIELDS),
            "optionalJobFields": {
                "references": "[{path, sha256}] project images sent after the first frame, "
                              "in order (at most maxReferences); recorded as provenance version 3",
            },
        },
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
    request = {
        "version": 1,
        "jobId": row["id"],
        "executionManifestSha256": plan["manifestSha256"],
        "model": plan["model"],
        "sourceSha256": row["sourceSha256"],
        "promptSha256": row["promptSha256"],
        "durationSeconds": row["durationSeconds"],
        "resolution": plan["resolution"],
        "aspectRatio": plan["aspectRatio"],
        "audio": True,
        "outputStem": row["outputStem"],
    }
    if row["referenceSha256s"]:
        # Absent references keep the version 1 request byte for byte.
        request.update(version=2, referenceSha256s=list(row["referenceSha256s"]))
    return request


def record_version(row: dict[str, Any]) -> int:
    """kie_video.py writes provenance version 3 exactly when references are sent."""
    return 3 if row["referenceSha256s"] else 2


def check_header(doc: dict[str, Any]) -> tuple[int, set[str]]:
    """Validate the manifest header; return its version and the job field sets allowed."""
    version = doc.get("version")
    if version == 2:
        for key, expected in {**FIXED_HEADER, "episode": EPISODE_CODE}.items():
            if doc.get(key) != expected:
                raise PlanError(f"execution manifest {key} must be {expected!r}")
        return 2, PRICING_FIELDS
    if version != 3:
        raise PlanError("execution manifest version must be 2 (fixed 720p 16:9) or 3 (frozen values)")
    fixed = {key: FIXED_HEADER[key] for key in ("provider", "language")}
    for key, expected in {**fixed, "episode": EPISODE_CODE}.items():
        if doc.get(key) != expected:
            raise PlanError(f"execution manifest {key} must be {expected!r}")
    values = ROUTE_VALUES.get(doc.get("model"))
    if values is None:
        raise PlanError(f"execution manifest model must be one of {sorted(ROUTE_VALUES)}")
    if doc.get("route") != values["route"]:
        raise PlanError(f"execution manifest route must be {values['route']!r} for {doc['model']}")
    if doc.get("resolution") not in values["resolutions"]:
        raise PlanError(f"resolution must be one of {sorted(values['resolutions'])}")
    if doc.get("aspectRatio") not in values["aspectRatios"]:
        raise PlanError(f"aspectRatio must be one of {sorted(values['aspectRatios'])}")
    return 3, V3_PRICING_FIELDS


def check_references(value: Any, job_id: str, maximum: int) -> list[tuple[Path, str]]:
    if not isinstance(value, list) or not value:
        raise PlanError(f"{job_id}: references must be a non-empty list; omit the field for none")
    if len(value) > maximum:
        raise PlanError(f"{job_id}: at most {maximum} references follow the first frame")
    found: list[tuple[Path, str]] = []
    for position, entry in enumerate(value):
        label = f"{job_id}.references[{position}]"
        if not isinstance(entry, dict) or set(entry) != REFERENCE_FIELDS:
            raise PlanError(f"{label} must be exactly {{path, sha256}}")
        path = project_path(entry["path"], label, EPISODE)
        digest = require_sha(entry["sha256"], f"{label}.sha256")
        if not path.is_file() or sha256(path) != digest:
            raise PlanError(f"{label} missing or hash changed: {path}")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise PlanError(f"{label} exceeds provider image limit")
        if path in {known for known, _ in found}:
            raise PlanError(f"{label} repeats a reference")
        found.append((path, digest))
    return found


def check_siblings(value: Any, manifest_path: Path, ids: set[str], stems: set[Path]) -> None:
    """Refuse a job or output that another frozen manifest of this project also holds.

    Only the siblings' frozen bytes are read, never their completion state, so
    this manifest's scope stays exactly what its approval record bound.
    """
    if not isinstance(value, list) or not value:
        raise PlanError("siblingManifests must be a non-empty list of {path, sha256}")
    for position, entry in enumerate(value):
        label = f"siblingManifests[{position}]"
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise PlanError(f"{label} must be exactly {{path, sha256}}")
        path = project_path(entry["path"], label, EPISODE / "work")
        if path == manifest_path.resolve():
            raise PlanError(f"{label} names this manifest itself")
        if not path.is_file() or sha256(path) != require_sha(entry["sha256"], f"{label}.sha256"):
            raise PlanError(f"{label} is missing or changed after this manifest was frozen: {path}")
        other = read_object(path, label).get("jobs")
        if not isinstance(other, list):
            raise PlanError(f"{label} has no jobs list")
        shared = sorted(ids & {job.get("id") for job in other if isinstance(job, dict)})
        if shared:
            raise PlanError(f"{label} already holds job ids {shared}; a job is paid for once")
        other_stems = {(EPISODE / job["outputStem"]).resolve() for job in other
                       if isinstance(job, dict) and isinstance(job.get("outputStem"), str)}
        if stems & other_stems:
            raise PlanError(f"{label} writes the same output stems {sorted(map(str, stems & other_stems))}")


def load_plan(manifest_path: Path, max_generated_seconds: int) -> dict[str, Any]:
    if (isinstance(max_generated_seconds, bool)
            or not isinstance(max_generated_seconds, int)
            or max_generated_seconds <= 0):
        raise PlanError("--max-generated-seconds must be a positive integer")
    manifest_path = manifest_path.resolve()
    doc = read_object(manifest_path, "execution manifest")
    if set(doc) - OPTIONAL_ROOT_FIELDS != ROOT_FIELDS:
        missing = sorted(ROOT_FIELDS - set(doc))
        extra = sorted(set(doc) - ROOT_FIELDS - OPTIONAL_ROOT_FIELDS)
        raise PlanError(f"execution manifest schema mismatch; missing={missing}, extra={extra}")
    version, pricing_fields = check_header(doc)
    if not isinstance(doc.get("financialApproval"), bool):
        raise PlanError("financialApproval must be boolean")
    if not isinstance(doc.get("financialState"), str) or not doc["financialState"]:
        raise PlanError("financialState must be a non-empty string")

    pricing = doc.get("pricing")
    if not isinstance(pricing, dict) or set(pricing) != pricing_fields:
        raise PlanError("pricing must bind a verified rate, UTC date and evidence"
                        + (", and the model and resolution it was read for" if version == 3 else ""))
    if version == 3 and (pricing["model"], pricing["resolution"]) != (doc["model"], doc["resolution"]):
        raise PlanError("pricing was read for a different model or resolution than this manifest sends")
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
        if not isinstance(raw, dict) or not (
                set(raw) == JOB_FIELDS
                or (version == 3 and set(raw) == JOB_FIELDS | {"references"})):
            raise PlanError(f"job {position} fields must be exactly {sorted(JOB_FIELDS)}"
                            + (" plus optional references" if version == 3 else ""))
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
        references = (check_references(raw["references"], job_id,
                                       ROUTE_VALUES[doc["model"]]["maxReferences"])
                      if "references" in raw else [])
        row = dict(raw)
        row.update(sourcePath=source, promptPath=prompt, outputPath=stem,
                   referencePaths=[path for path, _ in references],
                   referenceSha256s=[digest for _, digest in references])
        normalized.append(row)
        total += duration

    if "siblingManifests" in doc:
        check_siblings(doc["siblingManifests"], manifest_path, ids, stems)
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
        "version": version, "model": doc["model"],
        "resolution": doc["resolution"], "aspectRatio": doc["aspectRatio"],
        "financialApproval": doc["financialApproval"],
        "financialState": doc["financialState"],
        "jobs": normalized,
        "totalSeconds": total,
        "maxGeneratedSeconds": max_generated_seconds,
        "maximumCredits": maximum_credits, "pricing": pricing,
    }
    for row in normalized:
        row["executionRequestSha256"] = canonical_sha256(execution_request(plan, row))
    return plan


def minimum_geometry(resolution: str, aspect_ratio: str) -> tuple[float, float]:
    """The smallest landed width and height accepted for a frozen resolution and ratio.

    The nominal short edge is the resolution's number; 0.9375 of each nominal
    side is the margin version 2 always allowed (1200x675 for 720p 16:9).
    """
    short = int(resolution.rstrip("p"))
    across, down = (int(part) for part in aspect_ratio.split(":"))
    ratio = across / down
    nominal = (short * ratio, short) if ratio >= 1 else (short, short / ratio)
    return nominal[0] * 0.9375, nominal[1] * 0.9375


def validate_media(path: Path, expected_duration: int, resolution: str = "720p",
                   aspect_ratio: str = "16:9") -> dict[str, Any]:
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
        min_width, min_height = minimum_geometry(resolution, aspect_ratio)
        across, down = (int(part) for part in aspect_ratio.split(":"))
        if (width < min_width or height < min_height
                or abs(width / max(height, 1) - across / down) > 0.03):
            raise PlanError(f"video is not a valid {resolution}-class {aspect_ratio} result: {path}")
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
                            label: str, references: list[str] = ()) -> None:
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
    if (input_hashes.get("references") != list(references) or input_hashes.get("last") is not None
            or input_hashes.get("videos") != []):
        raise PlanError(f"{label}: local input hashes differ from the frozen still and references")


def expected_record(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": plan["model"],
        "jobId": row["id"],
        "executionManifestSha256": plan["manifestSha256"],
        "executionRequestSha256": row["executionRequestSha256"],
        "sourceSha256": row["sourceSha256"],
        "promptSha256": row["promptSha256"],
        "durationSeconds": row["durationSeconds"],
        "resolution": plan["resolution"],
        "aspectRatio": plan["aspectRatio"],
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
        "provenanceVersion": record_version(row),
        "provider": "kie.ai",
        **{key: value for key, value in expected.items() if key != "promptText"},
    }
    for key, value in fields.items():
        if meta.get(key) != value:
            raise PlanError(f"{row['id']}: sidecar {key} does not match frozen job")
    validate_local_envelope(meta, expected, row["id"], row["referenceSha256s"])
    _, fingerprint = validate_request_envelope(meta, expected, row["id"])
    media = validate_media(video, row["durationSeconds"], plan["resolution"], plan["aspectRatio"])
    if meta.get("outputSha256") != media["sha256"] or meta.get("sha256") != media["sha256"]:
        raise PlanError(f"{row['id']}: output bytes do not match the sidecar")
    task_id = meta.get("taskId")
    if not isinstance(task_id, str) or not task_id:
        raise PlanError(f"{row['id']}: sidecar lacks taskId")
    return {"id": row["id"], **media, "taskId": task_id,
            "requestFingerprint": fingerprint, "resumed": True,
            "credits": meta.get("credits"), "creditsSource": meta.get("creditsSource"),
            "firstFrame": first_frame_difference(video, row["sourcePath"])}


FRAME_SCALE = (480, 270)


def first_frame_difference(video: Path, still: Path) -> dict[str, Any]:
    """How far landed frame 0 is from the source still; a report, never a refusal.

    The arithmetic is the continuation seam audit's: both images scaled to
    480x270 RGB and compared by mean absolute channel difference (0-255).
    """
    import numpy as np
    width, height = FRAME_SCALE
    def decode(path: Path, *frame_args: str):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), *frame_args, "-frames:v", "1",
             "-vf", f"scale={width}:{height}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
            check=True, capture_output=True).stdout
        return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)
    try:
        difference = float(np.mean(np.abs(decode(video) - decode(still))))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        return {"method": "unmeasured", "error": str(exc)[-300:]}
    return {"method": "mean absolute RGB difference at 480x270, frame 0 against the source still",
            "meanAbsoluteDifference": round(difference, 4), "note": "report only; no threshold"}


PRICE_VARIANCE_REPORT_TOLERANCE = 1.25


def _credit(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def check_charges(plan: dict[str, Any], completed: dict[str, dict[str, Any]],
                  failed: dict[str, dict[str, Any]] | None = None,
                  history: dict[str, list[dict[str, Any]]] | None = None,
                  retry_credits: float = 0) -> dict[str, Any]:
    """Compare landed charges with the frozen price; stop past maximumCredits.

    The cumulative ceiling is the refusal: every landed credit counts, however
    it was measured. Per-job rates come only from provider receipts, because
    wallet deltas overlap when jobs run concurrently; they are reported, and a
    rate above the declared one is information for the next manifest's price.
    A job's first attempt counts toward maximumCredits; every retried attempt
    (``history`` holds the archived ones) counts toward the approval's
    retryBudget credits instead.
    """
    rate = plan["pricing"]["creditsPerSecond"]
    history = history or {}
    rows, cumulative, retried = [], 0.0, 0.0
    for job_id, attempts in history.items():
        cumulative += _credit(attempts[0].get("creditsConsumed"))
        retried += sum(_credit(attempt.get("creditsConsumed")) for attempt in attempts[1:])
        for number, attempt in enumerate(attempts, start=1):
            rows.append({"id": job_id, "attempt": number, "credits": attempt.get("creditsConsumed"),
                         "definiteFailure": True, "archived": True})
    for job_id, landed in completed.items():
        credits = landed.get("credits")
        if job_id in history:
            retried += _credit(credits)
        else:
            cumulative += _credit(credits)
        row = {"id": job_id, "credits": credits, "creditsSource": landed.get("creditsSource"),
               "measuredSeconds": landed.get("durationSeconds")}
        if (landed.get("creditsSource") == "receipt" and credits is not None
                and landed.get("durationSeconds")):
            per_second = float(credits) / float(landed["durationSeconds"])
            row.update(creditsPerMeasuredSecond=round(per_second, 4),
                       ratioToDeclared=round(per_second / rate, 4),
                       aboveReportTolerance=per_second > rate * PRICE_VARIANCE_REPORT_TOLERANCE)
        rows.append(row)
    for job_id, facts in (failed or {}).items():
        # A failed task can still be charged; it counts toward the ceiling.
        credits = facts.get("creditsConsumed")
        if job_id in history:
            retried += _credit(credits)
        else:
            cumulative += _credit(credits)
        rows.append({"id": job_id, "credits": credits,
                     "creditsSource": "failure-receipt" if credits is not None else None,
                     "definiteFailure": True})
    over = cumulative > plan["maximumCredits"]
    over_retry = retried > retry_credits
    report = {"manifestSha256": plan["manifestSha256"], "declaredCreditsPerSecond": rate,
              "maximumCredits": plan["maximumCredits"], "cumulativeLandedCredits": cumulative,
              "reportTolerance": PRICE_VARIANCE_REPORT_TOLERANCE,
              "status": "REFUSED" if over or over_retry else "WITHIN_CEILING", "jobs": rows,
              **({"retryCredits": retried, "retryBudgetCredits": retry_credits}
                 if history else {})}
    atomic_json(HERE / "price-variance.json", report)
    if over:
        raise PlanError(f"landed charges {cumulative} exceed maximumCredits "
                        f"{plan['maximumCredits']}; see price-variance.json")
    if over_retry:
        raise PlanError(f"retried attempts charged {retried}, past the approval's retry budget "
                        f"{retry_credits}; see price-variance.json")
    return report


def attempt_history(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Archived definite failures per job, oldest first, from motion/<id>.attempts/."""
    history = {}
    for row in plan["jobs"]:
        folder = row["outputPath"].parent / f"{row['id']}.attempts"
        attempts = []
        for path in sorted(folder.glob("attempt-*.pending.json")):
            record = read_object(path, f"{row['id']} archived attempt")
            if record.get("status") != DEFINITE_FAILURE or not record.get("taskId"):
                raise PlanError(f"{path} is not an archived definite failure")
            attempts.append(failure_facts(row, record))
        if attempts:
            history[row["id"]] = attempts
    return history


def archive_failure(row: dict[str, Any], number: int) -> Path:
    """Move a GENERATION_FAILED record into the job's numbered attempts; never overwrite."""
    recovery = row["outputPath"].with_suffix(".pending.json")
    target = row["outputPath"].parent / f"{row['id']}.attempts" / f"attempt-{number:03d}.pending.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(recovery, target)
    recovery.unlink()
    return target


def plan_warnings(plan: dict[str, Any]) -> list[str]:
    """Facts worth reading before paying; none of them refuses."""
    warnings = []
    referenced = [row["id"] for row in plan["jobs"] if row["referenceSha256s"]]
    if referenced and plan["resolution"] == "1080p":
        warnings.append("the provider page says only one image is supported at 1080p; these jobs "
                        f"send references: {referenced}")
    labels = ROUTE_VALUES[plan["model"]]
    for kind, value in (("resolutions", plan["resolution"]), ("aspectRatios", plan["aspectRatio"])):
        if labels[kind][value] != "exercised":
            warnings.append(f"{value} is {labels[kind][value]} on this route")
    if referenced:
        warnings.append("references are unqualified on this route until a one-clip pilot "
                        "records how image 1 and the references behave")
    return warnings


def estimated_cost(plan: dict[str, Any]) -> dict[str, Any]:
    credits = plan["totalSeconds"] * plan["pricing"]["creditsPerSecond"]
    usd = usd_per_credit()
    return {"estimatedCredits": credits,
            "estimatedUsd": round(credits * usd, 4) if usd is not None else None,
            "usdPerCredit": usd}


def validate_pending(plan: dict[str, Any], row: dict[str, Any], path: Path,
                     allowed_statuses: set[str] | None = None) -> dict[str, Any]:
    record = read_object(path, f"{row['id']} pending recovery record")
    expected = expected_record(plan, row)
    fields = {
        "provenanceVersion": record_version(row),
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
    validate_local_envelope(record, expected, row["id"], row["referenceSha256s"])
    validate_request_envelope(record, expected, row["id"])
    task_id = record.get("taskId")
    allowed_statuses = allowed_statuses or {"PENDING"}
    if record.get("status") not in allowed_statuses or not isinstance(task_id, str) or not task_id:
        raise PlanError(
            f"{row['id']}: pending state {record.get('status')!r} is not auto-retryable; "
            "reconcile the provider taskId manually")
    return record


DEFINITE_FAILURE = "GENERATION_FAILED"
# A wave stops after this many definite failures in a row, or when every job
# of a concurrency group of two or more fails, so one bad prompt or a provider
# outage cannot burn through a batch.
CONSECUTIVE_FAILURE_BRAKE = 3


def failure_facts(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """The raw facts of a definite provider failure; nothing is classified."""
    response = record.get("lastProviderResponse")
    response = response if isinstance(response, dict) else {}
    def fact(key: str, response_key: str) -> Any:
        return record[key] if record.get(key) is not None else response.get(response_key)
    return {"id": row["id"], "taskId": record["taskId"], "promptSha256": row["promptSha256"],
            "failCode": fact("failCode", "failCode"), "failMsg": fact("failMsg", "failMsg"),
            "creditsConsumed": fact("credits", "creditsConsumed")}


def log_refusals(failed: dict[str, dict[str, Any]]) -> None:
    """Append each new definite failure to refusals.jsonl beside the manifest.

    One line per provider task; operatorResponse is left for a person or agent
    to fill with what they did about it.
    """
    path = HERE / "refusals.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line).get("taskId"))
            except json.JSONDecodeError as exc:
                raise PlanError(f"refusals log is unreadable: {path}: {exc}") from exc
    from datetime import datetime, timezone
    new = [facts for facts in failed.values() if facts["taskId"] not in seen]
    if not new:
        return
    with path.open("a", encoding="utf-8") as log:
        for facts in new:
            log.write(json.dumps({
                "jobId": facts["id"], "taskId": facts["taskId"],
                "promptSha256": facts["promptSha256"], "failCode": facts["failCode"],
                "failMsg": facts["failMsg"], "creditsConsumed": facts["creditsConsumed"],
                "loggedAtUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "operatorResponse": None}, ensure_ascii=False, sort_keys=True) + "\n")
        log.flush()
        os.fsync(log.fileno())


def definite_failure(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    """Facts of this job's recorded GENERATION_FAILED task, or None for any other state."""
    recovery = row["outputPath"].with_suffix(".pending.json")
    if not recovery.exists():
        return None
    status = read_object(recovery, f"{row['id']} pending recovery record").get("status")
    if status != DEFINITE_FAILURE:
        return None
    return failure_facts(row, validate_pending(plan, row, recovery, {DEFINITE_FAILURE}))


def preflight(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]],
                                             dict[str, dict[str, Any]]]:
    """Inspect every destination/recovery record before the first paid subprocess.

    Returns jobs still to run, landed jobs, and definite provider failures. A
    failure with a recorded task ID is skipped and never re-POSTed; every
    other recovery state still halts.
    """
    pending: list[dict[str, Any]] = []
    completed: dict[str, dict[str, Any]] = {}
    failed: dict[str, dict[str, Any]] = {}
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
            record = validate_pending(plan, row, recovery, {"PENDING", DEFINITE_FAILURE})
            if record["status"] == DEFINITE_FAILURE:
                failed[row["id"]] = failure_facts(row, record)
                continue
        pending.append(row)
    return pending, completed, failed


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




def approval_scope(plan: dict[str, Any]) -> dict[str, Any]:
    """The finite execution limits an approval record must bind exactly."""
    return {"jobs": len(plan["jobs"]), "generatedSeconds": plan["totalSeconds"],
            "credits": plan["maximumCredits"]}


def kie_video_command(plan: dict[str, Any], row: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "pipeline/kie_video.py"),
        "grok15",
        str(row["outputPath"]),
        str(row["promptPath"]),
        str(row["sourcePath"]),
        "--res", plan["resolution"],
        "--dur", str(row["durationSeconds"]),
        "--ar", plan["aspectRatio"],
        "--job-id", row["id"],
        "--manifest-sha256", plan["manifestSha256"],
        "--source-sha256", row["sourceSha256"],
        "--execution-request-sha256", row["executionRequestSha256"],
    ] + [argument for path in row["referencePaths"] for argument in ("--ref", str(path))]


def upload_env() -> dict[str, str]:
    """kie_video.py logs every upload to this project's work/upload-log.jsonl."""
    return {**os.environ, "AGENTIC_UPLOAD_LOG": str(EPISODE / "work/upload-log.jsonl")}


def require_media_host() -> None:
    if platform.system() != "Linux":
        raise PlanError("Submission and heavy media validation require the qualified GPU worker")
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, capture_output=True)


def write_progress(plan: dict[str, Any], completed: dict[str, dict[str, Any]],
                   failed: dict[str, dict[str, Any]] | None = None) -> None:
    atomic_json(HERE / "progress.json", {
        "manifestSha256": plan["manifestSha256"],
        "completed": list(completed.values()),
        "jobCount": len(plan["jobs"]),
        **({"definiteFailures": list(failed.values())} if failed else {}),
    })


def print_requests(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Each job's provider request with placeholder URLs; kie_video never uploads here."""
    shown = []
    for row in plan["jobs"]:
        done = subprocess.run(kie_video_command(plan, row) + ["--print-request"], cwd=ROOT,
                              capture_output=True, text=True)
        if done.returncode:
            raise PlanError(f"{row['id']}: request preview failed: {done.stderr.strip()[-400:]}")
        request = json.loads(done.stdout)
        shown.append({"id": row["id"], "endpoint": request["endpoint"], "body": request["body"],
                      "localRequestSha256": request["localRequestSha256"]})
    return shown


def reconcile(max_generated_seconds: int) -> dict[str, Any]:
    """Finish already-submitted tasks without any new generation POST.

    A batch keeps its recorded task IDs whatever authorized it, so recovery
    needs no approval record: kie_video.py --resume-only polls a recorded
    taskId and refuses to create one. Jobs with no recovery record are listed
    and left alone.
    """
    plan = load_plan(MANIFEST, max_generated_seconds)
    require_media_host()
    acquire_lock(LOCK, plan["manifestSha256"])
    try:
        jobs, completed, definite = preflight(plan)
        reconciled, unsubmitted, failed = [], [], []
        for row in jobs:
            if not row["outputPath"].with_suffix(".pending.json").exists():
                unsubmitted.append(row["id"])
                continue
            result = subprocess.run(kie_video_command(plan, row) + ["--resume-only"], cwd=ROOT,
                                    env=upload_env())
            if result.returncode:
                facts = definite_failure(plan, row)
                if facts:
                    definite[row["id"]] = facts
                else:
                    failed.append(row["id"])
                continue
            completed[row["id"]] = {**validate_completed(plan, row), "resumed": True}
            reconciled.append(row["id"])
            write_progress(plan, completed, definite)
        log_refusals(definite)
        return {"mode": "RECONCILE_ONLY", "generationPosts": 0,
                "reconciled": reconciled, "unsubmitted": unsubmitted, "failed": failed,
                **({"definiteFailures": sorted(definite)} if definite else {}),
                "completed": len(completed), "jobCount": len(plan["jobs"]),
                "manifestSha256": plan["manifestSha256"]}
    finally:
        LOCK.unlink(missing_ok=True)


def execute(max_generated_seconds: int, submit: bool,
            approval_admission: Path | None = None, concurrency: int = 1,
            show_requests: bool = False, retry_failed: bool = False) -> dict[str, Any]:
    plan = load_plan(MANIFEST, max_generated_seconds)
    try:
        approval = validate_paid_approval(
            approval_admission, episode=EPISODE_CODE, lane=APPROVAL_LANE,
            manifest_path=plan["manifestPath"], manifest_sha256=plan["manifestSha256"],
            finite_execution=approval_scope(plan), allow_retry_budget=True,
        )
    except PaidApprovalError as exc:
        raise PlanError(str(exc)) from exc
    # The manifest's own financialApproval is reported, never trusted: agents
    # write manifests, so only the separate hash-bound record authorizes.
    authorized = approval is not None
    if not submit:
        return {
            "mode": "DRY_RUN",
            "networkCalls": 0,
            "jobs": len(plan["jobs"]),
            "generatedSeconds": plan["totalSeconds"],
            "capSeconds": plan["maxGeneratedSeconds"],
            "maximumCredits": plan["maximumCredits"],
            **estimated_cost(plan),
            "financialApproval": plan["financialApproval"],
            "financialAuthorized": authorized,
            "approvalAdmissionSha256": approval["sha256"] if approval else None,
            "manifestSha256": plan["manifestSha256"],
            **({"warnings": plan_warnings(plan)} if plan_warnings(plan) else {}),
            **({"requests": print_requests(plan),
                "requestNote": "image URLs are placeholders; nothing was uploaded"}
               if show_requests else {}),
        }
    if not authorized:
        raise PlanError("submission requires a current approval record (--approval-admission); "
                        "the manifest's financialApproval never authorizes spending")
    retry_budget = approval.get("retryBudget") if approval.get("automaticRetryAuthorized") else None
    if retry_failed and retry_budget is None:
        raise PlanError("--retry-failed is a new paid POST: it needs a version 3 approval record "
                        "with automaticRetryAuthorized and a retryBudget; otherwise freeze the "
                        "failed jobs as a new manifest with its own approval")

    require_media_host()
    if concurrency not in (1, 2, 3):
        raise PlanError("concurrency must be 1, 2 or 3")
    acquire_lock(LOCK, plan["manifestSha256"])
    try:
        jobs, completed, failed = preflight(plan)
        log_refusals(failed)
        authorization_path = HERE / "execution-authorization.json"
        evidence = {"executionManifestSha256": plan["manifestSha256"], "approval": approval}
        if authorization_path.exists():
            recorded = read_object(authorization_path, "execution authorization")
            if recorded != evidence:
                if recorded.get("approval") is None:
                    raise PlanError(
                        "this batch was started on the manifest's financialApproval with no "
                        "approval record; land its posted tasks with --reconcile-only, and "
                        "submit the remainder only with the runner that started it")
                if retry_failed:
                    raise PlanError(
                        "this batch started under a different approval record; a retry grant "
                        "must be in place before a batch's first submission, so freeze the failed "
                        "jobs as a new manifest with its own approval instead")
                raise PlanError("execution authorization differs from the existing batch")
        else:
            atomic_json(authorization_path, evidence)
        history = attempt_history(plan)
        retry_ids = set(failed) if retry_failed else set()
        if retry_ids:
            used = sum(len(attempts) for attempts in history.values())
            if used + len(retry_ids) > retry_budget["jobs"]:
                raise PlanError(f"retrying {sorted(retry_ids)} would make {used + len(retry_ids)} "
                                f"retries; the approval's retry budget is {retry_budget['jobs']}")
        retry_credits = retry_budget["credits"] if retry_budget else 0
        sys.path.insert(0, str(ROOT / "pipeline"))
        from _common import kie
        wallet_before = float(kie("chat/credit"))
        remaining_seconds = sum(row["durationSeconds"] for row in jobs
                                if not row["outputPath"].with_suffix(".pending.json").exists())
        remaining_seconds += sum(row["durationSeconds"] for row in plan["jobs"]
                                 if row["id"] in retry_ids)
        estimated_credits = remaining_seconds * plan["pricing"]["creditsPerSecond"]
        if wallet_before < estimated_credits:
            raise PlanError("Insufficient existing Kie credits; no top-up or overage")
        adopt_legacy_record(HERE, "credit-preflight")
        write_attempt_json(HERE, "credit-preflight", {
            "manifestSha256": plan["manifestSha256"], "availableCredits": wallet_before,
            "estimatedNewCredits": estimated_credits, "pricing": plan["pricing"],
            "maximumCredits": plan["maximumCredits"],
            "concurrency": concurrency, "noTopup": True,
            **({"retrying": sorted(retry_ids), "retryBudget": retry_budget} if retry_ids else {}),
        })
        if retry_ids:
            # Each retry is a new paid POST of the identical request; the failed
            # record moves into the job's numbered attempts and is never deleted.
            for row in plan["jobs"]:
                if row["id"] in retry_ids:
                    archive_failure(row, len(history.get(row["id"], [])) + 1)
            history = attempt_history(plan)
            jobs = [row for row in plan["jobs"]
                    if row["id"] in retry_ids or row["id"] in {job["id"] for job in jobs}]
            failed = {}
        class DefiniteFailure(Exception):
            def __init__(self, facts):
                super().__init__(f"provider failed task {facts['taskId']} "
                                 f"({facts['failCode']}: {facts['failMsg']})")
                self.facts = facts

        def generate_row(row):
            stem = row["outputPath"]
            result = subprocess.run(kie_video_command(plan, row), cwd=ROOT, env=upload_env())
            if result.returncode:
                facts = definite_failure(plan, row)
                if facts:
                    raise DefiniteFailure(facts)
                recovery = stem.with_suffix(".pending.json")
                state = ("durable recovery state exists" if recovery.exists()
                         else "no recovery state was recorded")
                raise PlanError(f"{row['id']} failed ({state}); stopped before later jobs")
            result = validate_completed(plan, row)
            result["resumed"] = False
            return result
        check_charges(plan, completed, failed, history, retry_credits)
        consecutive = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for offset in range(0, len(jobs), concurrency):
                group = jobs[offset:offset + concurrency]
                futures = [pool.submit(generate_row, row) for row in group]
                errors, group_failures = [], 0
                for row, future in zip(group, futures):
                    try:
                        completed[row["id"]] = future.result()
                        consecutive = 0
                        write_progress(plan, completed, failed)
                        print(f"TECHNICAL COMPLETE {len(completed)}/{len(plan['jobs'])}: {row['id']}", flush=True)
                    except DefiniteFailure as exc:
                        failed[row["id"]] = exc.facts
                        consecutive += 1
                        group_failures += 1
                        write_progress(plan, completed, failed)
                        print(f"DEFINITE FAILURE {row['id']}: {exc}", flush=True)
                    except Exception as exc:
                        errors.append(f"{row['id']}: {exc}")
                log_refusals(failed)
                if errors:
                    raise PlanError("; ".join(errors))
                check_charges(plan, completed, failed, history, retry_credits)
                if consecutive >= CONSECUTIVE_FAILURE_BRAKE or (len(group) > 1 and group_failures == len(group)):
                    raise PlanError(
                        f"stopped after definite provider failures {sorted(failed)}; read "
                        "refusals.jsonl before running the rest (landed and failed jobs are skipped)")
        ordered = [completed[row["id"]] for row in plan["jobs"] if row["id"] in completed]
        audit = {
            "episode": EPISODE_CODE,
            "executionManifestSha256": plan["manifestSha256"],
            "approvalAdmissionSha256": approval["sha256"] if approval else None,
            **({"retriedAttempts": history} if history else {}),
            "status": ("partial" if failed
                       else "technical-complete-creative-review-required"),
            **({"definiteFailures": list(failed.values()),
                "nextStep": "creative review of landed jobs; a failed job is retried only "
                            "as a new frozen manifest with its own approval"} if failed else {}),
            "generatedSecondsCap": max_generated_seconds,
            "count": len(ordered),
            "items": ordered,
            "walletBefore": wallet_before,
            "walletAfter": float(kie("chat/credit")),
            "preflightAttempts": attempt_records(HERE, "credit-preflight"),
            "accountingNote": "Shared wallet deltas are not per-task billing attribution",
        }
        audit_path = HERE / "technical-audit.json"
        atomic_json(audit_path, audit)
        return {"mode": "SUBMIT", "networkCalls": len(jobs),
                "status": "PARTIAL" if failed else "COMPLETED", "audit": str(audit_path),
                "items": len(ordered), **({"failed": sorted(failed)} if failed else {})}
    finally:
        LOCK.unlink(missing_ok=True)


UTC_SECOND = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
NO_AUDIT = {"kind": "source-audit", "newAngleAuditPerformed": False,
            "note": "No new-angle source audit was performed before this freeze."}
NO_OVERRIDES = {"items": {}, "note": "No production overrides were declared at freeze time."}


def _write_new_json(path: Path, value: Any) -> None:
    from pipeline._common import atomic_new_bytes
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        atomic_new_bytes(path, raw)
    except FileExistsError as exc:
        raise PlanError(f"refusing to overwrite {path}") from exc


def _relative(path: Path) -> str:
    return path.resolve().relative_to(EPISODE).as_posix()


def _draft_source(item: dict[str, Any]) -> Path:
    value = item.get("source", item.get("still"))
    if not isinstance(value, str) or not value:
        raise PlanError(f"{item.get('id')}: draft item names no source still")
    path = Path(value)
    return contained(path if path.is_absolute() else EPISODE / path, EPISODE, f"{item.get('id')}.source")


def freeze(draft_path: Path, manifest_path: Path, *, credits_per_second: float,
           price_verified_at: str, price_evidence: str, maximum_credits: float | None = None,
           durations: dict[str, int] | None = None, only: list[str] | None = None,
           omit_from: list[Path] | None = None, selection_reason: str | None = None,
           no_source_audit: bool = False, no_production_overrides: bool = False,
           episode_code: str | None = None, production_overrides: Path | None = None,
           authority_path: Path | None = None, resolution: str | None = None,
           aspect_ratio: str | None = None) -> dict[str, Any]:
    """Write execution-manifest.json from a draft with every hash computed.

    The field list comes from ROOT_FIELDS and JOB_FIELDS, so the writer and the
    validator cannot drift. The price is the operator's own live reading; the
    tool never supplies a rate. financialApproval is always false: only a
    separate approval record, drafted after this freeze, authorizes spending.
    A draft item's optional ``references`` (project image paths), or a
    resolution or aspect ratio other than version 2's, writes version 3; the
    price is then recorded for that model and resolution.
    """
    if EPISODE is None:
        raise PlanError("freeze needs --project (or AGENTIC_PROJECT_ROOT)")
    manifest_path = contained(manifest_path, EPISODE / "work", "manifest")
    if manifest_path.exists():
        raise PlanError(f"refusing to overwrite a frozen manifest: {manifest_path}")
    here = manifest_path.parent
    draft = read_object(draft_path.resolve(), "dialogue draft")
    code = episode_code or draft.get("episode")
    if not isinstance(code, str) or not code:
        raise PlanError("the draft names no episode; pass --episode-code")
    if (isinstance(credits_per_second, bool) or not isinstance(credits_per_second, (int, float))
            or not math.isfinite(credits_per_second) or credits_per_second <= 0):
        raise PlanError("--credits-per-second must be the positive rate you verified today")
    if not isinstance(price_verified_at, str) or not UTC_SECOND.fullmatch(price_verified_at):
        raise PlanError("--price-verified-at must be a UTC time such as 2026-01-01T12:00:00Z")
    if not isinstance(price_evidence, str) or not price_evidence.strip():
        raise PlanError("--price-evidence must say where the live price was read")
    values = ROUTE_VALUES[FIXED_HEADER["model"]]
    resolution = resolution or FIXED_HEADER["resolution"]
    aspect_ratio = aspect_ratio or FIXED_HEADER["aspectRatio"]
    if resolution not in values["resolutions"] or aspect_ratio not in values["aspectRatios"]:
        raise PlanError(f"--resolution must be one of {sorted(values['resolutions'])} and "
                        f"--aspect-ratio one of {sorted(values['aspectRatios'])}")
    items = draft.get("items")
    if not isinstance(items, list) or not items:
        raise PlanError("draft items must be a non-empty list")
    by_id = {item.get("id"): item for item in items if isinstance(item, dict)}
    if len(by_id) != len(items):
        raise PlanError("draft items need unique ids")

    omitted: dict[str, list[str]] = {}
    siblings = []
    for other_path in omit_from or []:
        other_path = contained(other_path, EPISODE / "work", "--omit-from manifest")
        other = read_object(other_path, "--omit-from manifest").get("jobs") or []
        held = sorted(job["id"] for job in other if isinstance(job, dict) and job.get("id") in by_id)
        omitted[_relative(other_path)] = held
        siblings.append({"path": _relative(other_path), "sha256": sha256(other_path)})
    selected = list(by_id)
    if only:
        unknown = sorted(set(only) - set(by_id))
        if unknown:
            raise PlanError(f"--only names ids missing from the draft: {unknown}")
        selected = [job_id for job_id in selected if job_id in set(only)]
    skip = {job_id for held in omitted.values() for job_id in held}
    selected = [job_id for job_id in selected if job_id not in skip]
    if not selected:
        raise PlanError("the selection leaves no jobs to freeze")
    if (only or omit_from) and not (selection_reason and selection_reason.strip()):
        raise PlanError("a selection needs --selection-reason, kept in selection-note.json")

    problems, jobs = [], []
    durations = durations or {}
    for job_id in selected:
        item = by_id[job_id]
        line = item.get("hindi", item.get("text"))
        duration = item.get("durationSeconds", durations.get(job_id))
        prompt = (EPISODE / item["prompt"]) if isinstance(item.get("prompt"), str) else here / "prompts" / f"{job_id}.txt"
        found = []
        if not isinstance(item.get("speaker"), str) or not item["speaker"].strip():
            found.append(f"{job_id}: no speaker")
        if not isinstance(line, str) or not line.strip():
            found.append(f"{job_id}: no dialogue line")
        if (isinstance(duration, bool) or not isinstance(duration, int)
                or duration not in ALLOWED_DURATIONS):
            found.append(f"{job_id}: durationSeconds missing or not a whole number "
                         f"{min(ALLOWED_DURATIONS)}-{max(ALLOWED_DURATIONS)}")
        if not prompt.is_file():
            found.append(f"{job_id}: prompt not written yet at {prompt}")
        elif isinstance(line, str) and line not in prompt.read_text(encoding="utf-8"):
            found.append(f"{job_id}: prompt does not contain the exact line")
        try:
            source = _draft_source(item)
            if not source.is_file():
                found.append(f"{job_id}: source still missing at {source}")
        except PlanError as exc:
            found.append(str(exc))
        references = []
        if "references" in item:
            raw_refs = item["references"]
            if (not isinstance(raw_refs, list) or not raw_refs
                    or len(raw_refs) > values["maxReferences"]):
                found.append(f"{job_id}: references must list 1-{values['maxReferences']} images")
                raw_refs = []
            for value in raw_refs:
                try:
                    ref = _draft_source({"id": job_id, "source": value})
                    if not ref.is_file():
                        found.append(f"{job_id}: reference missing at {ref}")
                    references.append(ref)
                except PlanError as exc:
                    found.append(str(exc))
        problems.extend(found)
        if found:
            continue
        job = {"id": job_id, "speaker": item["speaker"], "hindi": line,
               "durationSeconds": duration, "source": _relative(source),
               "sourceSha256": sha256(source), "prompt": _relative(prompt),
               "promptSha256": sha256(prompt),
               "outputStem": _relative(here / "motion" / job_id)}
        if references:
            job["references"] = [{"path": _relative(ref), "sha256": sha256(ref)}
                                 for ref in references]
        jobs.append(job)
    if problems:
        raise PlanError("cannot freeze yet: " + "; ".join(problems))

    here.mkdir(parents=True, exist_ok=True)
    draft_copy = here / "draft-manifest.json"
    if draft_copy.resolve() != draft_path.resolve():
        if draft_copy.exists() and draft_copy.read_bytes() != draft_path.read_bytes():
            raise PlanError(f"{draft_copy} already holds a different draft")
        if not draft_copy.exists():
            from pipeline._common import atomic_new_bytes
            atomic_new_bytes(draft_copy, draft_path.read_bytes())
    audit = here / "source-audit.json"
    if not audit.exists():
        if not no_source_audit:
            raise PlanError(f"write {audit} from your source inspection, or pass "
                            "--no-source-audit to record that none was performed")
        _write_new_json(audit, NO_AUDIT)
    overrides = contained(production_overrides or EPISODE / "story/production-overrides-v1.json",
                          EPISODE / "story", "production overrides")
    if not overrides.exists():
        if not no_production_overrides:
            raise PlanError(f"{overrides} is missing; pass --no-production-overrides to record none")
        _write_new_json(overrides, NO_OVERRIDES)
    authority = contained(authority_path or EPISODE / "story/GOAL.md", EPISODE / "story", "authority")
    if not authority.is_file():
        raise PlanError(f"authority file is missing: {authority}")

    total = sum(job["durationSeconds"] for job in jobs)
    ceiling = total * credits_per_second if maximum_credits is None else maximum_credits
    version3 = (any("references" in job for job in jobs)
                or (resolution, aspect_ratio) != (FIXED_HEADER["resolution"], FIXED_HEADER["aspectRatio"]))
    header = {**FIXED_HEADER, "episode": code}
    if version3:
        header.update(version=3, resolution=resolution, aspectRatio=aspect_ratio)
    doc = {
        **header,
        "financialApproval": False,
        "financialState": "frozen; awaiting a hash-bound approval record",
        "draftManifest": _relative(draft_copy), "draftManifestSha256": sha256(draft_copy),
        "newAngleAudit": _relative(audit), "newAngleAuditSha256": sha256(audit),
        "productionOverrides": _relative(overrides), "productionOverridesSha256": sha256(overrides),
        "authority": _relative(authority), "authoritySha256": sha256(authority),
        "jobCount": len(jobs), "maximumGeneratedSeconds": total, "jobs": jobs,
        "pricing": {"creditsPerSecond": credits_per_second, "verifiedAtUtc": price_verified_at,
                    "evidence": price_evidence.strip(),
                    **({"model": header["model"], "resolution": resolution} if version3 else {})},
        "maximumCredits": ceiling,
    }
    if siblings:
        doc["siblingManifests"] = siblings
    if (set(doc) - OPTIONAL_ROOT_FIELDS != ROOT_FIELDS
            or any(set(job) - {"references"} != JOB_FIELDS for job in jobs)):
        raise PlanError("freeze produced fields the validator does not accept")
    _write_new_json(manifest_path, doc)
    try:
        configure(manifest_path, EPISODE, overrides, code, authority)
        result = execute(total, False)
    except PlanError:
        manifest_path.unlink()
        raise
    if only or omit_from:
        _write_new_json(here / "selection-note.json", {
            "reason": selection_reason.strip(), "selected": selected,
            "only": sorted(only or []), "omittedBecauseHeldBy": omitted,
            "manifest": _relative(manifest_path), "manifestSha256": sha256(manifest_path),
            "draft": _relative(draft_copy), "draftJobCount": len(by_id)})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Manifest root fields: " + ", ".join(sorted(ROOT_FIELDS)) + ".\nJob fields: "
                + ", ".join(sorted(JOB_FIELDS)) + ".\nRun --print-schema for the whole contract; "
                "the approval lane is " + APPROVAL_LANE + "."))
    parser.add_argument("manifest", type=Path, nargs="?")
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
    parser.add_argument("--max-generated-seconds", type=int,
                        help="must equal the frozen total; required except with --print-schema")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--submit", action="store_true",
                       help="paid run; requires --approval-admission")
    parser.add_argument("--retry-failed", action="store_true",
                        help="with --submit: re-POST each GENERATION_FAILED job with the identical "
                             "request, moving its record to motion/<id>.attempts/; needs a version 3 "
                             "approval record granting a retry budget")
    modes.add_argument("--reconcile-only", action="store_true",
                       help="poll and land already-submitted tasks; never POSTs and needs no approval")
    parser.add_argument("--approval-admission", type=Path,
                        help="hash-bound paid authorization record (paid_approval.py)")
    modes.add_argument("--print-requests", action="store_true",
                       help="dry run that also prints each provider request with placeholder URLs")
    modes.add_argument("--print-schema", action="store_true",
                       help="print the execution manifest contract and exit")
    modes.add_argument("--freeze", type=Path, metavar="DRAFT",
                       help="write MANIFEST from a dialogue draft with every hash computed, "
                            "then dry-run it; never overwrites")
    freezing = parser.add_argument_group("freeze inputs (the price is your own live reading)")
    freezing.add_argument("--credits-per-second", type=float)
    freezing.add_argument("--price-verified-at", help="UTC time you read the provider's price")
    freezing.add_argument("--price-evidence", help="where you read it: page URL and model")
    freezing.add_argument("--maximum-credits", type=float,
                          help="ceiling; defaults to frozen seconds x credits per second")
    freezing.add_argument("--durations", type=Path,
                          help="JSON {job id: seconds} for draft items without durationSeconds")
    freezing.add_argument("--only", help="comma-separated job ids to freeze (a pilot, a retake set)")
    freezing.add_argument("--omit-from", type=Path, action="append",
                          help="omit jobs another frozen manifest already holds, and bind it "
                               "as a sibling so no job is paid for twice")
    freezing.add_argument("--selection-reason", help="why this subset; kept in selection-note.json")
    freezing.add_argument("--resolution", help="resolution to send (writes version 3 unless 720p)")
    freezing.add_argument("--aspect-ratio", help="aspect ratio to send (writes version 3 unless 16:9)")
    freezing.add_argument("--no-source-audit", action="store_true",
                          help="record that no new-angle source audit was performed")
    freezing.add_argument("--no-production-overrides", action="store_true",
                          help="record that this episode has no production overrides")
    args = parser.parse_args(argv)
    if args.print_schema:
        print(json.dumps(schema(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.freeze is not None:
        if args.manifest is None:
            parser.error("--freeze DRAFT needs the MANIFEST path to write")
        try:
            global EPISODE
            EPISODE = project_root(args.episode_root)
            durations = (read_object(args.durations.resolve(), "durations")
                         if args.durations else None)
            result = freeze(
                args.freeze, args.manifest, credits_per_second=args.credits_per_second,
                price_verified_at=args.price_verified_at, price_evidence=args.price_evidence,
                maximum_credits=args.maximum_credits, durations=durations,
                only=[x for x in (args.only or "").split(",") if x] or None,
                omit_from=args.omit_from, selection_reason=args.selection_reason,
                no_source_audit=args.no_source_audit,
                no_production_overrides=args.no_production_overrides,
                episode_code=args.episode_code, production_overrides=args.production_overrides,
                authority_path=args.authority_file, resolution=args.resolution,
                aspect_ratio=args.aspect_ratio)
        except (PlanError, ValueError) as exc:
            raise SystemExit(f"REFUSED: {exc}") from exc
        print(json.dumps({"frozen": str(args.manifest.resolve()), **result},
                         ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.manifest is None or args.max_generated_seconds is None:
        parser.error("manifest and --max-generated-seconds are required")
    try:
        configure(args.manifest, args.episode_root, args.production_overrides,
                  args.episode_code, args.authority_file)
        if (args.submit or args.reconcile_only) and args.credential_stdin:
            credential = json.load(sys.stdin)
            key = credential.get("kieApiKey")
            if not isinstance(key, str) or not key:
                raise PlanError("Missing Kie credential on stdin")
            os.environ["KIE_KEY"] = key
        if args.reconcile_only:
            result = reconcile(args.max_generated_seconds)
        else:
            if args.retry_failed and not args.submit:
                raise PlanError("--retry-failed only modifies --submit")
            result = execute(args.max_generated_seconds, args.submit,
                             args.approval_admission, args.concurrency,
                             show_requests=args.print_requests, retry_failed=args.retry_failed)
    except PlanError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
