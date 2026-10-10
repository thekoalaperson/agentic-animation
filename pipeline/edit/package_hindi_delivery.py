#!/usr/bin/env python3
"""Stage, verify, and explicitly promote one clutter-free Hindi delivery.

This is the qualified Hindi master, bilingual subtitle and Short preset.
It never deletes or rewrites source media.  ``--stage`` writes only below the
configured episode ``work/`` directory.  The canonical ``deliverables/``
directory is touched only by the explicit ``--promote-to-deliverables`` action,
and promotion refuses both collisions and pre-existing clutter.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit


LAYOUT = "hindi-finals-only-v1"
SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 3
# ffprobe otherwise decodes these long masters on one core. This only bounds
# decoder parallelism; frame selection, timing and validation stay unchanged.
MEDIA_DECODE_THREADS = "8"
GPU_FINISHING_PROBE = False  # Explicit CLI opt-in; legacy validation remains default.

COPY_ASSET_IDS = (
    "long-hi",
    "subtitles-hi",
    "subtitles-en",
    "short-hi",
    "short-hi-social",
    "short-subtitles-hi",
    "short-subtitles-en",
)
MEDIA_ASSET_IDS = ("long-hi", "short-hi", "short-hi-social")
SUBTITLE_ASSET_IDS = (
    "subtitles-hi",
    "subtitles-en",
    "short-subtitles-hi",
    "short-subtitles-en",
)
THUMBNAIL_OUTPUTS = {
    "thumbnail-master": "thumbnails/thumb_hi_titled.png",
    "thumbnail-hi": "thumbnails/thumb_hi_upload_1280.jpg",
}
GENERATED_OUTPUTS = {"upload-kit": "upload_kit.html", **THUMBNAIL_OUTPUTS}
ALL_ASSET_IDS = COPY_ASSET_IDS + tuple(GENERATED_OUTPUTS)
ALLOWED_FINAL_DIRECTORIES = {"shorts", "thumbnails"}
REQUIRED_KIT_IDS = (
    "hi-title",
    "hi-description",
    "common-tags",
    "short-hi-title",
    "short-hi-desc",
    "short-tags",
)
APPROVAL_IDS = ("longForm", "subtitles", "short", "thumbnail", "title")
REVIEW_NAME_TOKENS = re.compile(
    r"(?:^|[-_.])(review|candidate|variant|option|sample|debug|draft|temp|720p)(?:[-_.]|$)",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
RFC3339_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class DeliveryError(RuntimeError):
    """A deliberate packaging refusal with an actionable reason."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DeliveryError(f"Missing config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DeliveryError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeliveryError(f"Top level must be an object: {path}")
    return value


def _config_snapshot(path: Path) -> Tuple[Dict[str, Any], str]:
    """Read config once and return the parsed document plus that byte snapshot's hash."""
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise DeliveryError(f"Missing config: {path}") from exc
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeliveryError(f"Invalid UTF-8 JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeliveryError(f"Top level must be an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _schema_type_matches(value: Any, expected: str) -> bool:
    """Match JSON types without Python's bool-is-an-int ambiguity."""
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    raise DeliveryError(f"Unsupported JSON Schema type in delivery schema: {expected}")


def _schema_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _resolve_schema_ref(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise DeliveryError(f"Only local JSON Schema references are supported: {reference}")
    value: Any = root
    for raw in reference[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or token not in value:
            raise DeliveryError(f"Broken JSON Schema reference: {reference}")
        value = value[token]
    if not isinstance(value, dict):
        raise DeliveryError(f"JSON Schema reference is not an object: {reference}")
    return value


def _validate_schema(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], label: str = "config") -> None:
    """Validate the deliberately small Draft 2020-12 subset used by our schema.

    Keeping this subset in-tree avoids turning final packaging into a network or
    third-party-Python dependency while still making the checked-in schema the
    executable structural authority.
    """
    if "$ref" in schema:
        _validate_schema(value, _resolve_schema_ref(root, str(schema["$ref"])), root, label)
    for branch in schema.get("allOf", []):
        _validate_schema(value, branch, root, label)
    if "anyOf" in schema:
        errors: List[str] = []
        for branch in schema["anyOf"]:
            try:
                _validate_schema(value, branch, root, label)
                break
            except DeliveryError as exc:
                errors.append(str(exc))
        else:
            raise DeliveryError(f"{label} does not match any allowed schema form: {'; '.join(errors)}")

    expected_type = schema.get("type")
    if expected_type is not None:
        choices = [expected_type] if isinstance(expected_type, str) else list(expected_type)
        if not any(_schema_type_matches(value, choice) for choice in choices):
            raise DeliveryError(f"{label} has the wrong JSON type; expected {choices}")
    if "const" in schema and not _schema_equal(value, schema["const"]):
        raise DeliveryError(f"{label} must equal {schema['const']!r}")
    if "enum" in schema and not any(_schema_equal(value, option) for option in schema["enum"]):
        raise DeliveryError(f"{label} is not one of the allowed values")

    if isinstance(value, dict):
        required = set(schema.get("required", []))
        missing = sorted(required - set(value))
        if missing:
            raise DeliveryError(f"{label} is missing required keys: {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise DeliveryError(f"{label} has unsupported keys: {extra}")
        for key, child in properties.items():
            if key in value:
                _validate_schema(value[key], child, root, f"{label}.{key}")
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise DeliveryError(f"{label} must contain at least {schema['minItems']} items")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], root, f"{label}[{index}]")
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise DeliveryError(f"{label} is shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise DeliveryError(f"{label} is longer than {schema['maxLength']} characters")
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            raise DeliveryError(f"{label} does not match its required pattern")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise DeliveryError(f"{label} must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            raise DeliveryError(f"{label} must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise DeliveryError(f"{label} must be at most {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise DeliveryError(f"{label} must be greater than {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise DeliveryError(f"{label} must be less than {schema['exclusiveMaximum']}")


def _validate_config_schema(config: Mapping[str, Any]) -> None:
    schema_path = Path(__file__).with_name("hindi_delivery_config.schema.json")
    schema = _json(schema_path)
    _validate_schema(config, schema, schema)


def _keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    allowed: Iterable[str],
    label: str,
) -> None:
    required_set, allowed_set = set(required), set(allowed)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed_set)
    if missing or extra:
        raise DeliveryError(f"{label} keys invalid; missing={missing}, extra={extra}")


def _relative_path(raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise DeliveryError(f"{label} must be a non-empty relative path")
    if "\\" in raw:
        raise DeliveryError(f"{label} must use POSIX separators: {raw}")
    path = Path(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DeliveryError(f"Unsafe {label}: {raw}")
    return path


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_symlink_chain(path: Path, root: Path, label: str) -> None:
    current = root
    relative = path.relative_to(root)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise DeliveryError(f"Symlinks are not accepted for {label}: {current}")


def episode_path(
    episode: Path,
    raw: Any,
    label: str,
    *,
    must_exist: bool = False,
) -> Path:
    relative = _relative_path(raw, label)
    candidate = episode / relative
    resolved_parent = candidate.parent.resolve()
    if not _is_below(resolved_parent, episode):
        raise DeliveryError(f"{label} escapes the episode: {raw}")
    _reject_symlink_chain(candidate, episode, label)
    if must_exist and not candidate.is_file():
        raise DeliveryError(f"Missing {label}: {candidate}")
    return candidate


def _validate_destination(asset_id: str, raw: Any) -> str:
    rel = _relative_path(raw, f"assets.{asset_id}.destination")
    text = rel.as_posix()
    if REVIEW_NAME_TOKENS.search(rel.name):
        raise DeliveryError(f"Review/variant naming is forbidden in finals: {text}")
    if asset_id == "long-hi":
        valid = len(rel.parts) == 1 and re.fullmatch(
            r"[^/]+_hi_1080p(?:[-_][^/]*)?\.mp4", rel.name, re.IGNORECASE
        ) is not None
    elif asset_id in {"subtitles-hi", "subtitles-en"}:
        valid = len(rel.parts) == 1 and rel.suffix.lower() == ".srt"
        valid = valid and rel.stem.endswith("-hi" if asset_id == "subtitles-hi" else "-en")
    elif asset_id in {"short-hi", "short-hi-social"}:
        valid = len(rel.parts) == 2 and rel.parts[0] == "shorts" and rel.suffix.lower() == ".mp4"
        suffix = "_short_hi_social.mp4" if asset_id == "short-hi-social" else "_short_hi.mp4"
        valid = valid and rel.name.endswith(suffix)
    else:
        valid = len(rel.parts) == 2 and rel.parts[0] == "shorts" and rel.suffix.lower() == ".srt"
        valid = valid and rel.stem.endswith("-hi" if asset_id == "short-subtitles-hi" else "-en")
    if not valid:
        raise DeliveryError(f"Destination does not match the {asset_id} final slot: {text}")
    return text


def _fraction(raw: Any, label: str) -> Fraction:
    if not isinstance(raw, str):
        raise DeliveryError(f"{label} must be a rational string such as 24/1")
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError) as exc:
        raise DeliveryError(f"Invalid {label}: {raw}") from exc
    if value <= 0:
        raise DeliveryError(f"{label} must be positive")
    return value


def _validate_media_expectation(asset_id: str, expected: Mapping[str, Any]) -> None:
    required = {
        "width", "height", "fps", "frames", "durationSeconds", "videoCodec",
        "pixelFormat", "audioCodec", "audioChannels", "audioSampleRate",
        "maxAvDeltaSeconds", "loudness",
    }
    _keys(expected, required=required, allowed=required, label=f"assets.{asset_id}.expected")
    geometry = (1920, 1080) if asset_id == "long-hi" else (1080, 1920)
    if (expected["width"], expected["height"]) != geometry:
        raise DeliveryError(f"{asset_id} must declare exact geometry {geometry[0]}x{geometry[1]}")
    if expected["fps"] != "24/1":
        raise DeliveryError(f"{asset_id} final cadence must be 24/1")
    _fraction(expected["fps"], f"assets.{asset_id}.expected.fps")
    for key in ("frames", "durationSeconds"):
        value = expected[key]
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise DeliveryError(f"assets.{asset_id}.expected.{key} must be positive or null")
    if expected["frames"] is not None and not isinstance(expected["frames"], int):
        raise DeliveryError(f"assets.{asset_id}.expected.frames must be an integer or null")
    if expected["frames"] is not None and expected["durationSeconds"] is not None:
        derived = expected["frames"] / float(_fraction(expected["fps"], "fps"))
        if abs(derived - float(expected["durationSeconds"])) > 1.0 / 24.0:
            raise DeliveryError(f"{asset_id} expected frames and duration disagree")
    if asset_id.startswith("short-hi") and expected["durationSeconds"] is not None and expected["durationSeconds"] >= 60.0:
        raise DeliveryError(f"{asset_id} must stay under 60 seconds")
    if expected["videoCodec"] != "h264" or expected["pixelFormat"] != "yuv420p":
        raise DeliveryError(f"{asset_id} must declare upload-safe h264/yuv420p picture")
    if expected["audioCodec"] != "aac" or expected["audioChannels"] != 2:
        raise DeliveryError(f"{asset_id} must declare one AAC stereo audio stream")
    if expected["audioSampleRate"] != 48000:
        raise DeliveryError(f"{asset_id} must declare 48 kHz delivery audio")
    delta = expected["maxAvDeltaSeconds"]
    if not isinstance(delta, (int, float)) or not 0 <= delta <= 0.5:
        raise DeliveryError(f"{asset_id} maxAvDeltaSeconds must be between 0 and 0.5")
    loudness = expected["loudness"]
    if not isinstance(loudness, dict):
        raise DeliveryError(f"assets.{asset_id}.expected.loudness must be an object")
    loud_keys = {"targetIntegratedLufs", "toleranceLu", "maxTruePeakDbtp"}
    _keys(loudness, required=loud_keys, allowed=loud_keys, label=f"assets.{asset_id}.expected.loudness")
    fixed_target = -10.0 if asset_id == "short-hi-social" else -14.0
    if loudness["targetIntegratedLufs"] != fixed_target:
        raise DeliveryError(f"{asset_id} loudness target must be {fixed_target:g} LUFS")
    if not isinstance(loudness["toleranceLu"], (int, float)) or not 0 < loudness["toleranceLu"] <= 1.0:
        raise DeliveryError(f"{asset_id} loudness tolerance must be >0 and <=1 LU")
    if not isinstance(loudness["maxTruePeakDbtp"], (int, float)) or loudness["maxTruePeakDbtp"] > -0.5:
        raise DeliveryError(f"{asset_id} true-peak ceiling must be at most -0.5 dBTP")


def _valid_utc_timestamp(raw: Any) -> bool:
    if not isinstance(raw, str) or RFC3339_UTC_RE.fullmatch(raw) is None:
        return False
    try:
        datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _approval_subject_hashes(config: Mapping[str, Any], approval_id: str) -> List[str]:
    assets = config["assets"]
    metadata = config["metadata"]
    long_metadata = {
        "channel": config["channel"],
        "title": metadata["title"],
        "description": metadata["description"],
        "tags": metadata["tags"],
        "hashtags": metadata["hashtags"],
        "pinnedComment": metadata["pinnedComment"],
        "chapters": metadata["chapters"],
        "uploadSettings": config["metadata"]["uploadSettings"],
    }
    short_metadata = {
        "title": metadata["shortTitle"],
        "description": metadata["shortDescription"],
        "tags": metadata["shortTags"],
        "humanReview": config["short"]["humanReview"],
        "revision": config["short"]["revision"],
        "ctaText": config["short"]["ctaText"],
        "ctaSeconds": config["short"]["ctaSeconds"],
        "relatedVideoId": config["short"]["relatedVideoId"],
    }
    if 'operatorNotes' in metadata:
        long_metadata['operatorNotes'] = metadata['operatorNotes']
    if config['short'].get('delegatedAuthority'):
        short_metadata['delegatedAuthority']=config['short']['delegatedAuthority']
    mapping = {
        "longForm": [assets["long-hi"]["sha256"], _canonical_json_sha256(long_metadata)],
        "subtitles": [assets["subtitles-hi"]["sha256"], assets["subtitles-en"]["sha256"]],
        "short": [
            assets["short-hi"]["sha256"], assets["short-hi-social"]["sha256"],
            assets["short-subtitles-hi"]["sha256"], assets["short-subtitles-en"]["sha256"],
            _canonical_json_sha256(short_metadata),
        ],
        "thumbnail": [config["thumbnail"]["sha256"]] + [r['sha256'] for r in config.get('thumbnailAlternatives', [])],
        "title": [hashlib.sha256(config["metadata"]["title"].encode("utf-8")).hexdigest()],
    }
    values = mapping[approval_id]
    if not all(isinstance(value, str) and SHA256_RE.fullmatch(value) for value in values):
        raise DeliveryError(f"Approved {approval_id} cannot bind missing or invalid subject hashes")
    return values


def _validate_approval_evidence(
    episode: Path,
    approval_id: str,
    row: Mapping[str, Any],
    expected_subjects: Sequence[str],
) -> None:
    if not _valid_utc_timestamp(row["approvedAt"]):
        raise DeliveryError(f"approvals.{approval_id}.approvedAt must be second-precision RFC3339 UTC")
    evidence = episode_path(episode, row["evidence"], f"approvals.{approval_id}.evidence")
    if evidence.suffix.lower() != ".json" or _is_below(evidence.resolve(), (episode / "deliverables").resolve()):
        raise DeliveryError(f"Approval evidence for {approval_id} must be a JSON record outside deliverables/")
    if not evidence.is_file():
        raise FileNotFoundError(str(evidence))
    if sha256_file(evidence) != row["evidenceSha256"]:
        raise DeliveryError(f"Approval evidence SHA-256 mismatch for {approval_id}: {evidence}")
    doc = _json(evidence)
    evidence_keys = {"schemaVersion", "approvalId", "verdict", "approvedAt", "subjectSha256"}
    _keys(doc, required=evidence_keys, allowed=evidence_keys, label=f"approval evidence {approval_id}")
    expected = {
        "schemaVersion": 1,
        "approvalId": approval_id,
        "verdict": "approved",
        "approvedAt": row["approvedAt"],
        "subjectSha256": list(expected_subjects),
    }
    if doc != expected:
        raise DeliveryError(f"Approval evidence does not bind the exact {approval_id} subjects and verdict")


def analyze_config(
    episode: Path,
    config_path: Path,
    *,
    snapshot: Optional[Tuple[Dict[str, Any], str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    episode = episode.resolve()
    config_path = config_path.resolve()
    if not episode.is_dir():
        raise DeliveryError(f"Episode directory does not exist: {episode}")
    config = snapshot[0] if snapshot is not None else _config_snapshot(config_path)[0]
    _validate_config_schema(config)
    top = {
        "schemaVersion", "layout", "episode", "stagingDir", "publishingAuthorized",
        "publishAt", "channel", "approvals", "assets", "thumbnail", "metadata", "short",
    }
    _keys(config, required=top, allowed=top | {'thumbnailAlternatives'}, label="config")
    if config["schemaVersion"] != SCHEMA_VERSION or config["layout"] != LAYOUT:
        raise DeliveryError(f"Expected schemaVersion={SCHEMA_VERSION} and layout={LAYOUT}")
    if config["episode"] != episode.name:
        raise DeliveryError(f"Config episode {config['episode']!r} does not match {episode.name!r}")
    if config["publishingAuthorized"] is not False or config["publishAt"] is not None:
        raise DeliveryError("A delivery package cannot authorize publication or set publishAt")

    channel = config["channel"]
    if not isinstance(channel, dict):
        raise DeliveryError("channel must be an object")
    _keys(channel, required={"handle", "id", "language"}, allowed={"handle", "id", "language", "name"}, label="channel")
    if channel["language"] != "hi":
        raise DeliveryError("This delivery preset is qualified for Hindi performance with Hindi and English subtitles")

    staging_rel = _relative_path(config["stagingDir"], "stagingDir")
    if not staging_rel.parts or staging_rel.parts[0] != "work" or staging_rel.name == "deliverables":
        raise DeliveryError("stagingDir must be an episode-relative path below work/")
    episode_path(episode, config["stagingDir"], "stagingDir")

    approvals = config["approvals"]
    if not isinstance(approvals, dict) or set(approvals) != set(APPROVAL_IDS):
        raise DeliveryError(f"approvals must contain exactly {list(APPROVAL_IDS)}")
    gates: List[str] = []
    for approval_id in APPROVAL_IDS:
        row = approvals[approval_id]
        if not isinstance(row, dict):
            raise DeliveryError(f"approvals.{approval_id} must be an object")
        approval_keys = {"status", "evidence", "evidenceSha256", "approvedAt", "subjectSha256"}
        _keys(row, required=approval_keys, allowed=approval_keys, label=f"approvals.{approval_id}")
        if row["status"] not in {"pending", "approved"}:
            raise DeliveryError(f"approvals.{approval_id}.status must be pending or approved")
        if row["status"] != "approved":
            gates.append(f"approval:{approval_id}")
        else:
            if not row["evidence"] or not row["approvedAt"] or not row["evidenceSha256"] or not row["subjectSha256"]:
                gates.append(f"approval-evidence:{approval_id}")
            else:
                expected_subjects = _approval_subject_hashes(config, approval_id)
                if row["subjectSha256"] != expected_subjects:
                    raise DeliveryError(f"approvals.{approval_id}.subjectSha256 does not bind the current approved assets")
                try:
                    _validate_approval_evidence(episode, approval_id, row, expected_subjects)
                except FileNotFoundError:
                    gates.append(f"missing-approval-evidence:{approval_id}")

    assets = config["assets"]
    if not isinstance(assets, dict) or set(assets) != set(COPY_ASSET_IDS):
        raise DeliveryError(f"assets must contain exactly {list(COPY_ASSET_IDS)}")
    destinations: Dict[str, str] = {}
    for asset_id in COPY_ASSET_IDS:
        row = assets[asset_id]
        if not isinstance(row, dict):
            raise DeliveryError(f"assets.{asset_id} must be an object")
        if asset_id in MEDIA_ASSET_IDS:
            allowed = required = {"source", "sha256", "destination", "expected"}
        else:
            allowed = required = {"source", "sha256", "destination", "authored"}
        _keys(row, required=required, allowed=allowed, label=f"assets.{asset_id}")
        destinations[asset_id] = _validate_destination(asset_id, row["destination"])
        if asset_id in MEDIA_ASSET_IDS:
            if not isinstance(row["expected"], dict):
                raise DeliveryError(f"assets.{asset_id}.expected must be an object")
            _validate_media_expectation(asset_id, row["expected"])
            if row["expected"]["frames"] is None:
                gates.append(f"expected-frames:{asset_id}")
            if row["expected"]["durationSeconds"] is None:
                gates.append(f"expected-duration:{asset_id}")
        elif row["authored"] is not True:
            gates.append(f"authored-subtitle:{asset_id}")
        source: Optional[Path] = None
        if row["source"] is None:
            gates.append(f"source:{asset_id}")
        else:
            source = episode_path(episode, row["source"], f"assets.{asset_id}.source")
            if _is_below(source.resolve(), (episode / "deliverables").resolve()):
                raise DeliveryError(f"Source may not be read back from deliverables: {row['source']}")
            if not source.is_file():
                gates.append(f"missing-source:{asset_id}")
        digest = row["sha256"]
        if digest is None:
            gates.append(f"sha256:{asset_id}")
        elif not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise DeliveryError(f"assets.{asset_id}.sha256 must be lowercase SHA-256 or null")
        elif source is not None and source.is_file() and sha256_file(source) != digest:
            raise DeliveryError(f"Configured source SHA-256 mismatch for {asset_id}: {source}")

    if len(set(destinations.values())) != len(destinations):
        raise DeliveryError("Two asset IDs resolve to the same final destination")

    thumbnail = config["thumbnail"]
    if not isinstance(thumbnail, dict):
        raise DeliveryError("thumbnail must be an object")
    thumb_keys = {
        "method", "textlessSource", "textlessSha256", "titleRecipe",
        "titleRecipeSha256", "source", "sha256", "choice", "jpegMaxBytes",
    }
    _keys(thumbnail, required=thumb_keys, allowed=thumb_keys, label="thumbnail")
    if thumbnail["method"] not in {"builtin-imagegen-edit", "gpu-typography"}:
        raise DeliveryError("Unsupported thumbnail provenance method")
    if not isinstance(thumbnail["jpegMaxBytes"], int) or not 100_000 <= thumbnail["jpegMaxBytes"] <= 2_000_000:
        raise DeliveryError("thumbnail.jpegMaxBytes must be 100000..2000000")
    if not thumbnail["choice"]:
        gates.append("thumbnail-choice")

    textless_source: Optional[Path] = None
    if thumbnail["textlessSource"] is None:
        gates.append("thumbnail-textless-source")
    else:
        textless_source = episode_path(episode, thumbnail["textlessSource"], "thumbnail.textlessSource")
        if _is_below(textless_source.resolve(), (episode / "deliverables").resolve()):
            raise DeliveryError("Thumbnail textless source may not be read back from deliverables")
        if not textless_source.is_file():
            gates.append("missing-thumbnail-textless-source")
    if thumbnail["textlessSha256"] is None:
        gates.append("thumbnail-textless-sha256")
    elif textless_source is not None and textless_source.is_file() and sha256_file(textless_source) != thumbnail["textlessSha256"]:
        raise DeliveryError(f"Configured textless thumbnail SHA-256 mismatch: {textless_source}")

    recipe: Optional[Dict[str, Any]] = None
    recipe_path: Optional[Path] = None
    if thumbnail["titleRecipe"] is None:
        gates.append("thumbnail-title-recipe")
    else:
        recipe_path = episode_path(episode, thumbnail["titleRecipe"], "thumbnail.titleRecipe")
        if _is_below(recipe_path.resolve(), (episode / "deliverables").resolve()):
            raise DeliveryError("Thumbnail title recipe may not be read back from deliverables")
        if not recipe_path.is_file():
            gates.append("missing-thumbnail-title-recipe")
        else:
            recipe = _json(recipe_path)
    if thumbnail["titleRecipeSha256"] is None:
        gates.append("thumbnail-title-recipe-sha256")
    elif recipe_path is not None and recipe_path.is_file() and sha256_file(recipe_path) != thumbnail["titleRecipeSha256"]:
        raise DeliveryError(f"Configured thumbnail title-recipe SHA-256 mismatch: {recipe_path}")

    if recipe is not None and thumbnail["method"] == "gpu-typography":
        choices = [r for r in recipe.get('variants', []) if r.get('id') == thumbnail['choice']]
        if (not str(recipe.get('method', '')).startswith('GPU-host FreeType') or len(choices) != 1
                or choices[0].get('sourceSha256') != thumbnail['textlessSha256']
                or choices[0].get('files', {}).get(Path(thumbnail['source']).name) != thumbnail['sha256']):
            raise DeliveryError('GPU typography receipt must bind the chosen source and exact output')

    if recipe is not None and thumbnail["method"] == "builtin-imagegen-edit":
        recipe_keys = {"schemaVersion", "method", "renderer", "textlessSource",
                       "textlessSha256", "output", "outputSha256", "prompt",
                       "promptSha256", "titleLines"}
        _keys(recipe, required=recipe_keys, allowed=recipe_keys, label="ImageGen thumbnail recipe")
        if (recipe["schemaVersion"] != 1 or recipe["method"] != thumbnail["method"]
                or recipe["renderer"] != "image_gen.imagegen"
                or recipe["textlessSource"] != thumbnail["textlessSource"]
                or recipe["textlessSha256"] != thumbnail["textlessSha256"]
                or not isinstance(recipe["titleLines"], list) or not recipe["titleLines"]
                or not all(isinstance(line, str) and line.strip() for line in recipe["titleLines"])):
            raise DeliveryError("Invalid built-in ImageGen thumbnail provenance")
        prompt = episode_path(episode, recipe["prompt"], "thumbnail prompt", must_exist=True)
        if _is_below(prompt, episode / "deliverables") or sha256_file(prompt) != recipe["promptSha256"]:
            raise DeliveryError("Thumbnail prompt provenance mismatch")
        output_rel = _relative_path(recipe["output"], "ImageGen thumbnail output").as_posix()
        if not output_rel.startswith(("work/", "art/thumbnails/")) or output_rel == recipe["textlessSource"]:
            raise DeliveryError("ImageGen source must be a distinct immutable work/art thumbnail")
        if not isinstance(recipe["outputSha256"], str) or not SHA256_RE.fullmatch(recipe["outputSha256"]):
            raise DeliveryError("ImageGen output must have a recorded SHA-256")
        if thumbnail["source"] != output_rel or thumbnail["sha256"] != recipe["outputSha256"]:
            raise DeliveryError("ImageGen source must match its recorded output")

    thumbnail_source: Optional[Path] = None
    if thumbnail["source"] is None:
        gates.append("thumbnail-source")
    else:
        thumbnail_source = episode_path(episode, thumbnail["source"], "thumbnail.source")
        if _is_below(thumbnail_source.resolve(), (episode / "deliverables").resolve()):
            raise DeliveryError("Thumbnail source may not be read back from deliverables")
        if not thumbnail_source.is_file():
            gates.append("missing-thumbnail-source")
    if thumbnail["sha256"] is None:
        gates.append("thumbnail-sha256")
    elif not isinstance(thumbnail["sha256"], str) or not SHA256_RE.fullmatch(thumbnail["sha256"]):
        raise DeliveryError("thumbnail.sha256 must be lowercase SHA-256 or null")
    elif thumbnail_source is not None and thumbnail_source.is_file() and sha256_file(thumbnail_source) != thumbnail["sha256"]:
        raise DeliveryError(f"Configured thumbnail source SHA-256 mismatch: {thumbnail_source}")

    for alternative in config.get('thumbnailAlternatives', []):
        source = episode_path(episode, alternative['source'], 'thumbnail alternative', must_exist=True)
        destination = _relative_path(alternative['destination'], 'thumbnail alternative destination')
        if (len(destination.parts) != 2 or destination.parts[0] != 'thumbnails'
                or destination.suffix.lower() != '.jpg' or REVIEW_NAME_TOKENS.search(destination.name)
                or sha256_file(source) != alternative['sha256'] or source.stat().st_size > thumbnail['jpegMaxBytes']):
            raise DeliveryError('Invalid approved thumbnail alternative')
        if _is_below(source.resolve(), (episode / 'deliverables').resolve()):
            raise DeliveryError('Thumbnail alternatives must retain external source provenance')

    metadata = config["metadata"]
    if not isinstance(metadata, dict):
        raise DeliveryError("metadata must be an object")
    metadata_keys = {
        "title", "description", "tags", "hashtags", "pinnedComment", "chapters",
        "shortTitle", "shortDescription", "shortTags", "uploadSettings",
    }
    _keys(metadata, required=metadata_keys, allowed=metadata_keys | {'operatorNotes'}, label="metadata")
    for field in ("title", "description", "shortTitle", "shortDescription"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            gates.append(f"metadata:{field}")
    if isinstance(metadata["title"], str) and len(metadata["title"]) > 100:
        raise DeliveryError("metadata.title exceeds YouTube's 100-character limit")
    if isinstance(metadata["shortTitle"], str) and len(metadata["shortTitle"]) > 100:
        raise DeliveryError("metadata.shortTitle exceeds YouTube's 100-character limit")
    for field in ("description", "shortDescription"):
        if isinstance(metadata[field], str) and len(metadata[field]) > 5000:
            raise DeliveryError(f"metadata.{field} exceeds 5000 characters")
    for field in ("tags", "shortTags", "hashtags"):
        values = metadata[field]
        if not isinstance(values, list) or not values or not all(isinstance(x, str) and x.strip() for x in values):
            gates.append(f"metadata:{field}")
            continue
        if field != "hashtags" and len(", ".join(values)) > 500:
            raise DeliveryError(f"metadata.{field} exceeds 500 joined characters")
    if metadata["pinnedComment"] is not None and not isinstance(metadata["pinnedComment"], str):
        raise DeliveryError("metadata.pinnedComment must be a string or null")
    chapters = metadata["chapters"]
    if not isinstance(chapters, list) or not chapters:
        gates.append("metadata:chapters")
    else:
        previous = -10.0
        for index, chapter in enumerate(chapters):
            if not isinstance(chapter, dict):
                raise DeliveryError(f"metadata.chapters[{index}] must be an object")
            _keys(chapter, required={"seconds", "title"}, allowed={"seconds", "title"}, label=f"metadata.chapters[{index}]")
            seconds = chapter["seconds"]
            if not isinstance(seconds, (int, float)) or seconds < 0 or not chapter["title"]:
                raise DeliveryError(f"Invalid metadata.chapters[{index}]")
            if index == 0 and seconds != 0:
                raise DeliveryError("The first YouTube chapter must start at 0 seconds")
            if index and seconds - previous < 10.0:
                raise DeliveryError("YouTube chapters must be at least 10 seconds apart")
            previous = float(seconds)

    short = config["short"]
    if not isinstance(short, dict):
        raise DeliveryError("short must be an object")
    short_keys = {"humanReview", "revision", "relatedVideoId", "ctaText", "ctaSeconds"}
    _keys(short, required=short_keys, allowed=short_keys | {'delegatedAuthority'}, label="short")
    if short["humanReview"] not in {"pending", "approved", "delegated-studio-approved"}:
        raise DeliveryError("short.humanReview must name pending, human approval, or delegated studio approval")
    if short['humanReview']=='delegated-studio-approved':
        authority=short.get('delegatedAuthority')
        if not isinstance(authority,dict) or set(authority)!={'path','sha256'}:
            raise DeliveryError('Delegated studio review requires exact recorded user authority')
        path=episode_path(episode,authority['path'],'delegated review authority',must_exist=True)
        if _is_below(path,episode/'deliverables') or sha256_file(path)!=authority['sha256']:
            raise DeliveryError('Delegated studio authority binding changed')
    elif short.get('delegatedAuthority'):
        raise DeliveryError('Delegated authority is only valid with an explicit delegated studio verdict')
    if short["humanReview"] == "pending":
        gates.append("short-human-review")
    if not isinstance(short["revision"], str) or not short["revision"].strip():
        gates.append("short-revision")
    if short["relatedVideoId"] is not None:
        raise DeliveryError("relatedVideoId must remain null before publication")
    if not isinstance(short["ctaText"], str) or not short["ctaText"].strip():
        raise DeliveryError("The Short CTA must contain the approved display text")
    if not isinstance(short["ctaSeconds"], (int, float)) or short["ctaSeconds"] < 3.0:
        raise DeliveryError("The Short CTA must remain readable for at least 3 seconds")

    return config, sorted(set(gates))


def _run(command: Sequence[str], *, label: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, text=True, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise DeliveryError(f"Required executable not found for {label}: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-2000:]
        raise DeliveryError(f"{label} failed: {detail}") from exc


def _number(raw: Any, label: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise DeliveryError(f"Missing numeric {label}: {raw!r}") from exc


def _stream_duration(stream: Mapping[str, Any], fallback: float) -> float:
    if stream.get("duration") not in {None, "N/A"}:
        return _number(stream["duration"], "stream duration")
    if stream.get("duration_ts") not in {None, "N/A"} and stream.get("time_base") not in {None, "N/A"}:
        return float(int(stream["duration_ts"]) * Fraction(stream["time_base"]))
    return fallback


def _frame_timing_report(path: Path, expected_frames: int, fps: Fraction) -> Dict[str, Any]:
    result = _run(
        [
            "ffprobe", "-v", "error", "-threads", MEDIA_DECODE_THREADS,
            "-select_streams", "v:0", "-show_entries",
            "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path),
        ],
        label=f"decoded frame timing for {path.name}",
    )
    timestamps: List[float] = []
    for line in result.stdout.splitlines():
        raw = line.strip().split(",", 1)[0]
        if raw and raw != "N/A":
            timestamps.append(_number(raw, "frame timestamp"))
    if len(timestamps) != expected_frames:
        raise DeliveryError(f"{path.name} exposes {len(timestamps)} frame timestamps, expected {expected_frames}")
    expected_delta = 1.0 / float(fps)
    errors = [abs((right - left) - expected_delta) for left, right in zip(timestamps, timestamps[1:])]
    max_error = max(errors, default=0.0)
    if max_error > 0.00075:
        raise DeliveryError(f"{path.name} is not CFR; maximum frame-step error is {max_error:.6f}s")
    encoded = "\n".join(f"{value:.6f}" for value in timestamps).encode("ascii")
    return {
        "firstSeconds": round(timestamps[0], 6),
        "lastSeconds": round(timestamps[-1], 6),
        "maxCadenceErrorSeconds": round(max_error, 9),
        "timelineSha256": hashlib.sha256(encoded).hexdigest(),
    }


def _audio_packet_timing_report(path: Path) -> Dict[str, Any]:
    result = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets",
            "-show_entries", "packet=pts_time,duration_time", "-of", "json", str(path),
        ],
        label=f"AAC packet timing for {path.name}",
    )
    packets: List[Tuple[float, float]] = []
    # FFprobe4.4 CSV can append Skip Samples side-data directly to a numeric
    # duration. Structured fields keep priming metadata separate; never strip
    # a suffix and guess a packet clock.
    try:
        rows = json.loads(result.stdout)["packets"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DeliveryError(f"{path.name} has invalid AAC packet JSON") from exc
    for row in rows:
        if not isinstance(row, dict) or row.get("pts_time") in {None, "", "N/A"} or row.get("duration_time") in {None, "", "N/A"}:
            raise DeliveryError(f"{path.name} has an AAC packet without numeric PTS/duration")
        pts = _number(row["pts_time"], "AAC packet PTS")
        duration = _number(row["duration_time"], "AAC packet duration")
        if duration <= 0:
            raise DeliveryError(f"{path.name} has a non-positive AAC packet duration")
        packets.append((pts, duration))
    if not packets:
        raise DeliveryError(f"{path.name} exposes no AAC packets")
    continuity_errors: List[float] = []
    for (left_pts, left_duration), (right_pts, _right_duration) in zip(packets, packets[1:]):
        if right_pts <= left_pts:
            raise DeliveryError(f"{path.name} AAC packet PTS is not strictly monotonic")
        continuity_errors.append(abs(right_pts - (left_pts + left_duration)))
    max_error = max(continuity_errors, default=0.0)
    if max_error > 0.0015:
        raise DeliveryError(f"{path.name} AAC packet timeline has a gap/overlap of {max_error:.6f}s")
    encoded = "\n".join(f"{pts:.6f},{duration:.6f}" for pts, duration in packets).encode("ascii")
    return {
        "packetCount": len(packets),
        "firstPtsSeconds": round(packets[0][0], 6),
        "lastEndSeconds": round(packets[-1][0] + packets[-1][1], 6),
        "maxContinuityErrorSeconds": round(max_error, 9),
        "timelineSha256": hashlib.sha256(encoded).hexdigest(),
    }


def _video_payload_sha256(path: Path) -> str:
    result = _run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "hash", "-hash", "sha256", "-"],
        label=f"video-payload hash for {path.name}",
    )
    match = re.search(r"SHA256=([0-9a-fA-F]{64})", result.stdout)
    if not match:
        raise DeliveryError(f"Could not read video-payload hash for {path}")
    return match.group(1).lower()


def _measure_loudness(path: Path, target_lufs: float, peak_ceiling: float) -> Dict[str, float]:
    result = _run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-i", str(path),
            "-map", "0:a:0", "-af",
            f"loudnorm=I={target_lufs}:TP={peak_ceiling}:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        label=f"loudness measurement for {path.name}",
    )
    matches = re.findall(r"\{\s*\"input_i\".*?\}", result.stderr, re.DOTALL)
    if not matches:
        raise DeliveryError(f"ffmpeg did not emit loudnorm JSON for {path}")
    try:
        raw = json.loads(matches[-1])
        return {
            "integratedLufs": float(raw["input_i"]),
            "truePeakDbtp": float(raw["input_tp"]),
            "loudnessRangeLu": float(raw["input_lra"]),
            "thresholdLufs": float(raw["input_thresh"]),
        }
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise DeliveryError(f"Malformed loudnorm report for {path}") from exc


_MEDIA_VALIDATION_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def validate_media_file(path: Path, asset_id: str, expected: Mapping[str, Any]) -> Dict[str, Any]:
    """Reuse this process's fresh measurements only for identical bytes/contracts.

    Stage/promote rehash copied files; a verified immutable file need not undergo
    three identical full decodes in one operation. No disk/imported audit cache.
    """
    before = sha256_file(path)
    implementation = {"packager": sha256_file(Path(__file__)), "gpuFinishingProbe": GPU_FINISHING_PROBE}
    if GPU_FINISHING_PROBE:
        import gpu_finishing_probe as gpu_probe
        binary = shutil.which("ffmpeg")
        if binary is None:
            raise DeliveryError("ffmpeg unavailable for GPU finishing")
        implementation.update(helper=sha256_file(Path(gpu_probe.__file__)), ffmpeg=sha256_file(Path(binary)))
    key = (before, _canonical_json_sha256(expected), _canonical_json_sha256(implementation))
    if key in _MEDIA_VALIDATION_CACHE:
        return json.loads(json.dumps(_MEDIA_VALIDATION_CACHE[key]))
    report = _validate_media_file_uncached(path, asset_id, expected)
    if sha256_file(path) != before:
        raise DeliveryError(f"{asset_id} bytes changed during media validation")
    _MEDIA_VALIDATION_CACHE[key] = json.loads(json.dumps(report))
    return report


def _validate_media_file_uncached(path: Path, asset_id: str, expected: Mapping[str, Any]) -> Dict[str, Any]:
    packet_clock = None
    if GPU_FINISHING_PROBE:
        import gpu_finishing_probe as gpu_probe
        doc, packet_clock = gpu_probe.inspect_h264_mp4(path, expected["frames"])
    else:
        probe = _run(
            [
                "ffprobe", "-v", "error", "-threads", MEDIA_DECODE_THREADS,
                "-count_frames", "-show_streams", "-show_format",
                "-of", "json", str(path),
            ],
            label=f"ffprobe {asset_id}",
        )
        try:
            doc = json.loads(probe.stdout)
        except json.JSONDecodeError as exc:
            raise DeliveryError(f"Invalid ffprobe JSON for {path}") from exc
    streams = doc.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1 or len(streams) != 2:
        raise DeliveryError(f"{asset_id} must contain exactly one video and one audio stream")
    video, audio = videos[0], audios[0]
    if (video.get("width"), video.get("height")) != (expected["width"], expected["height"]):
        raise DeliveryError(f"{asset_id} geometry is {video.get('width')}x{video.get('height')}, expected {expected['width']}x{expected['height']}")
    avg_fps = _fraction(video.get("avg_frame_rate"), f"{asset_id} avg_frame_rate")
    declared_fps = _fraction(expected["fps"], f"{asset_id} expected fps")
    if avg_fps != declared_fps or _fraction(video.get("r_frame_rate"), f"{asset_id} r_frame_rate") != declared_fps:
        raise DeliveryError(f"{asset_id} cadence is not exact {expected['fps']} CFR")
    frames_raw = packet_clock["frames"] if packet_clock is not None else video.get("nb_read_frames") or video.get("nb_frames")
    if frames_raw in {None, "N/A"}:
        raise DeliveryError(f"{asset_id} frame count is unavailable")
    frames = int(frames_raw)
    if frames != expected["frames"]:
        raise DeliveryError(f"{asset_id} has {frames} frames, expected {expected['frames']}")
    if video.get("codec_name") != expected["videoCodec"] or video.get("pix_fmt") != expected["pixelFormat"]:
        raise DeliveryError(f"{asset_id} picture codec/pixel format does not match its contract")
    if audio.get("codec_name") != expected["audioCodec"]:
        raise DeliveryError(f"{asset_id} audio codec is {audio.get('codec_name')}, expected {expected['audioCodec']}")
    if int(audio.get("channels", 0)) != expected["audioChannels"] or int(audio.get("sample_rate", 0)) != expected["audioSampleRate"]:
        raise DeliveryError(f"{asset_id} audio must be {expected['audioChannels']}ch/{expected['audioSampleRate']}Hz")

    format_duration = _number(doc.get("format", {}).get("duration"), f"{asset_id} format duration")
    video_duration = _stream_duration(video, frames / float(avg_fps))
    audio_duration = _stream_duration(audio, format_duration)
    frame_timing = packet_clock if packet_clock is not None else _frame_timing_report(path, frames, avg_fps)
    audio_timing = _audio_packet_timing_report(path)
    video_start = frame_timing["firstSeconds"]
    audio_start = audio_timing["firstPtsSeconds"]
    frame_seconds = 1.0 / float(avg_fps)
    if abs(video_duration - float(expected["durationSeconds"])) > frame_seconds + 0.002:
        raise DeliveryError(f"{asset_id} video duration {video_duration:.6f}s disagrees with expected {expected['durationSeconds']:.6f}s")
    av_start_delta = abs(audio_start - video_start)
    av_end_delta = abs(audio_timing["lastEndSeconds"] - (video_start + video_duration))
    if max(av_start_delta, av_end_delta) > float(expected["maxAvDeltaSeconds"]):
        raise DeliveryError(
            f"{asset_id} A/V boundary delta exceeds contract "
            f"(start={av_start_delta:.6f}s, end={av_end_delta:.6f}s)"
        )

    gpu_decode = None
    if GPU_FINISHING_PROBE:
        gpu_decode = gpu_probe.decode_h264_worker(path, frames)
    else:
        _run(
            ["ffmpeg", "-v", "error", "-xerror", "-threads", MEDIA_DECODE_THREADS,
             "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
            label=f"full decode {asset_id}",
        )
    loud = _measure_loudness(
        path,
        float(expected["loudness"]["targetIntegratedLufs"]),
        float(expected["loudness"]["maxTruePeakDbtp"]),
    )
    if abs(loud["integratedLufs"] - float(expected["loudness"]["targetIntegratedLufs"])) > float(expected["loudness"]["toleranceLu"]):
        raise DeliveryError(f"{asset_id} measured {loud['integratedLufs']:.2f} LUFS outside target tolerance")
    if loud["truePeakDbtp"] > float(expected["loudness"]["maxTruePeakDbtp"]):
        raise DeliveryError(f"{asset_id} true peak {loud['truePeakDbtp']:.2f} dBTP exceeds ceiling")
    report = {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": expected["fps"],
        "frames": frames,
        "videoDurationSeconds": round(video_duration, 6),
        "audioDurationSeconds": round(audio_duration, 6),
        "formatDurationSeconds": round(format_duration, 6),
        "videoStartSeconds": round(video_start, 6),
        "audioStartSeconds": round(audio_start, 6),
        "avStartDeltaSeconds": round(av_start_delta, 6),
        "avEndDeltaSeconds": round(av_end_delta, 6),
        "frameCadenceMaxErrorSeconds": frame_timing["maxCadenceErrorSeconds"],
        "videoTimelineSha256": frame_timing["timelineSha256"],
        "audioPacketCount": audio_timing["packetCount"],
        "audioPacketContinuityMaxErrorSeconds": audio_timing["maxContinuityErrorSeconds"],
        "audioPacketTimelineSha256": audio_timing["timelineSha256"],
        "videoCodec": video["codec_name"],
        "pixelFormat": video["pix_fmt"],
        "audioCodec": audio["codec_name"],
        "audioChannels": int(audio["channels"]),
        "audioSampleRate": int(audio["sample_rate"]),
        "fullDecode": "passed",
        "videoPayloadSha256": _video_payload_sha256(path),
        "loudness": loud,
    }
    if gpu_decode is not None:
        # Stable evidence only: temporary paths, elapsed time and progress logs
        # differ across a genuine staging revalidation and are not media facts.
        stable_keys = ("sourceSha256", "actualDecodedFrameCount", "decoder", "hardwareAcceleration",
                       "hardwareOutputFormat", "errorLogEmpty", "errorLogSha256", "duplicatedFrames",
                       "droppedFrames", "toolSha256", "ffmpegBinarySha256", "passed")
        report.update(frameTimingBasis=packet_clock["basis"],
                      integerVideoPacketClockSha256=packet_clock["integerPacketClockSha256"],
                      gpuDecode={key: gpu_decode[key] for key in stable_keys})
    return report


SRT_TIME_RE = re.compile(r"^(\d{2,}):(\d{2}):(\d{2}),(\d{3})$")


def _srt_milliseconds(raw: str) -> int:
    match = SRT_TIME_RE.fullmatch(raw.strip())
    if not match:
        raise DeliveryError(f"Invalid SRT timestamp: {raw!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    if minutes > 59 or seconds > 59:
        raise DeliveryError(f"Invalid SRT timestamp: {raw!r}")
    return (((hours * 60) + minutes) * 60 + seconds) * 1000 + millis


def parse_srt(path: Path, *, language: str, media_duration: float) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
    if not text:
        raise DeliveryError(f"Empty authored subtitle: {path}")
    blocks = re.split(r"\n\s*\n", text)
    timings: List[Tuple[int, int]] = []
    bodies: List[str] = []
    previous_end = 0
    for expected_index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0].strip() != str(expected_index):
            raise DeliveryError(f"{path.name} cue {expected_index} has invalid numbering/body")
        if " --> " not in lines[1]:
            raise DeliveryError(f"{path.name} cue {expected_index} has invalid timing line")
        start_raw, end_raw = lines[1].split(" --> ", 1)
        start, end = _srt_milliseconds(start_raw), _srt_milliseconds(end_raw)
        if start < previous_end or end <= start:
            raise DeliveryError(f"{path.name} cue {expected_index} overlaps or has non-positive duration")
        body = "\n".join(lines[2:]).strip()
        if not body:
            raise DeliveryError(f"{path.name} cue {expected_index} is empty")
        timings.append((start, end))
        bodies.append(body)
        previous_end = end
    if timings[-1][1] > int((media_duration + 0.25) * 1000):
        raise DeliveryError(f"{path.name} extends beyond its media duration")
    joined = " ".join(bodies)
    if language == "hi" and not DEVANAGARI_RE.search(joined):
        raise DeliveryError(f"Hindi authored SRT has no Devanagari text: {path.name}")
    if language == "en" and DEVANAGARI_RE.search(joined):
        raise DeliveryError(f"English authored SRT contains Devanagari text: {path.name}")
    return {
        "cueCount": len(timings),
        "firstStartSeconds": timings[0][0] / 1000.0,
        "lastEndSeconds": timings[-1][1] / 1000.0,
        "timings": timings,
        "authored": True,
        "language": language,
    }


def _probe_image(path: Path) -> Dict[str, Any]:
    result = _run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        label=f"image probe for {path.name}",
    )
    try:
        streams = json.loads(result.stdout)["streams"]
    except (KeyError, json.JSONDecodeError) as exc:
        raise DeliveryError(f"Invalid image probe for {path}") from exc
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise DeliveryError(f"Thumbnail must decode as one image: {path}")
    _run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-frames:v", "1", "-f", "null", "-"], label=f"image decode for {path.name}")
    return {"width": int(videos[0]["width"]), "height": int(videos[0]["height"]), "pixelFormat": videos[0].get("pix_fmt"), "fullDecode": "passed"}


def render_thumbnail(source: Path, png: Path, jpg: Path, jpeg_max_bytes: int) -> Dict[str, Any]:
    source_probe = _probe_image(source)
    if abs((source_probe["width"] / source_probe["height"]) - (16.0 / 9.0)) > 0.002:
        raise DeliveryError(f"Approved thumbnail source is not exact 16:9: {source_probe['width']}x{source_probe['height']}")
    png.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf",
            "scale=1280:720:flags=lanczos,format=rgb24", "-frames:v", "1",
            "-compression_level", "9", "-update", "1", str(png),
        ],
        label="lossless 1280x720 thumbnail export",
    )
    chosen_quality: Optional[int] = None
    for quality in (2, 3, 4, 5, 6, 7, 8, 10, 12):
        _run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(png), "-frames:v", "1",
                "-pix_fmt", "yuvj420p", "-q:v", str(quality), "-update", "1", str(jpg),
            ],
            label=f"upload JPEG thumbnail export q={quality}",
        )
        if jpg.stat().st_size <= jpeg_max_bytes:
            chosen_quality = quality
            break
    if chosen_quality is None:
        raise DeliveryError(f"Thumbnail JPEG cannot meet {jpeg_max_bytes} bytes without excessive compression")
    png_probe, jpg_probe = _probe_image(png), _probe_image(jpg)
    if (png_probe["width"], png_probe["height"]) != (1280, 720) or (jpg_probe["width"], jpg_probe["height"]) != (1280, 720):
        raise DeliveryError("Both final thumbnail formats must be exactly 1280x720")
    return {
        "source": source_probe,
        "png": {**png_probe, "sha256": sha256_file(png), "bytes": png.stat().st_size},
        "jpeg": {**jpg_probe, "sha256": sha256_file(jpg), "bytes": jpg.stat().st_size, "ffmpegQscale": chosen_quality, "derivedFromPngSha256": sha256_file(png)},
        "sameApprovedComposition": True,
    }


def _clock(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _copy_block(label: str, ident: str, value: str) -> str:
    return (
        '<section class="copy-block"><div><h2>' + html.escape(label) +
        f'</h2><button type="button" data-copy="{html.escape(ident, quote=True)}">Copy</button></div>' +
        f'<pre id="{html.escape(ident, quote=True)}">{html.escape(value)}</pre></section>'
    )


def description_with_chapters(metadata: Mapping[str, Any]) -> str:
    chapter_text = "\n".join(f"{_clock(float(row['seconds']))} {row['title']}" for row in metadata["chapters"])
    hashtag_text = " ".join(metadata["hashtags"])
    return "\n\n".join(part for part in (metadata["description"].strip(), chapter_text, hashtag_text) if part)


def upload_settings_html(config: Mapping[str, Any]) -> str:
    """Display the producer's explicit settings without inferring publication authority."""
    settings = config["metadata"]["uploadSettings"]
    fields = (("Video language", "videoLanguage"), ("AI-use declaration", "aiUse"),
              ("Requested visibility", "visibility"), ("Made for kids", "madeForKids"),
              ("Category", "category"))
    rows = "".join(f"<dt>{label}</dt><dd>{html.escape(settings[key])}</dd>" for label, key in fields)
    return ('<section id="hi-upload-settings"><h2>Provided upload settings</h2><dl>' + rows
            + '</dl><p>These settings describe the intended upload. Verify declarations against the actual content; '
            'publication and scheduling require a separate instruction.</p></section>')


def build_upload_kit(config: Mapping[str, Any], root: Path) -> str:
    metadata, assets = config["metadata"], config["assets"]
    description = description_with_chapters(metadata)
    tags = ", ".join(metadata["tags"])
    short_tags = ", ".join(metadata["shortTags"])
    review_notice = (
        '<p>Studio-approved under recorded delegated authority. This approval '
        'does not certify a personal final watch by the producer.</p>'
        if config['short'].get('humanReview') == 'delegated-studio-approved' else ''
    )
    links = (
        f'<a href="{html.escape(assets["long-hi"]["destination"], quote=True)}">Hindi 1080p master</a>'
        f' · <a href="{THUMBNAIL_OUTPUTS["thumbnail-hi"]}">upload JPEG</a>'
        f' · <a href="{THUMBNAIL_OUTPUTS["thumbnail-master"]}">lossless PNG</a>'
        f' · <a href="FINAL_MANIFEST.json">machine manifest</a>'
    )
    body = (
        f'<header><p>FINAL LOCAL HANDOFF · {html.escape(config["episode"].upper())}</p>'
        f'<h1>{html.escape(metadata["title"])}</h1>'
        f'<p>Hindi delivery preset · {html.escape(config["channel"]["handle"])} · '
        'two authored selectable subtitle tracks · publication not authorized.</p>'
        + review_notice + '</header>'
        f'<nav>{links}</nav><main>'
        f'<section class="thumb"><img src="{THUMBNAIL_OUTPUTS["thumbnail-hi"]}" alt="Approved Hindi upload thumbnail">'
        '<p>The PNG and JPEG are exports of the same approved image. Use the JPEG only if the PNG exceeds the upload limit.</p></section>'
        + _copy_block("Approved Hindi public title", "hi-title", metadata["title"])
        + _copy_block("Hindi description + measured chapters", "hi-description", description)
        + _copy_block("Common bilingual tags", "common-tags", tags)
        + upload_settings_html(config)
    )
    for alternative in config.get('thumbnailAlternatives', []):
        target = html.escape(alternative['destination'], quote=True)
        label = html.escape(alternative['label'])
        body += f'<section class="thumb"><h2>{label}</h2><a href="{target}"><img src="{target}" alt="{label}">Download thumbnail</a></section>'
    if metadata.get("pinnedComment"):
        body += _copy_block("Pinned comment", "hi-pinned-comment", metadata["pinnedComment"])
    if metadata.get('operatorNotes'):
        body += _copy_block('Upload operator notes — not public description', 'operator-notes', metadata['operatorNotes'])
    body += (
        '<section><h2>Selectable subtitle files</h2>'
        f'<p><a href="{html.escape(assets["subtitles-hi"]["destination"], quote=True)}">Hindi SRT</a> · '
        f'<a href="{html.escape(assets["subtitles-en"]["destination"], quote=True)}">English SRT</a></p></section>'
        '<section><h2>Hindi Short</h2>'
        f'<video controls preload="metadata" src="{html.escape(assets["short-hi"]["destination"], quote=True)}"></video>'
        f'<p><a href="{html.escape(assets["short-hi"]["destination"], quote=True)}">YouTube −14 LUFS</a> · '
        f'<a href="{html.escape(assets["short-hi-social"]["destination"], quote=True)}">Reels −10 LUFS</a> · '
        f'<a href="{html.escape(assets["short-subtitles-hi"]["destination"], quote=True)}">Hindi Short SRT</a> · '
        f'<a href="{html.escape(assets["short-subtitles-en"]["destination"], quote=True)}">English Short SRT</a></p></section>'
        + _copy_block("Hindi Short title", "short-hi-title", metadata["shortTitle"])
        + _copy_block("Hindi Short description", "short-hi-desc", metadata["shortDescription"])
        + _copy_block("Short tags", "short-tags", short_tags)
        + '<aside><h2>Uploader boundary</h2><p>This package is locally verified but does not grant permission to upload, publish, select visibility, set a schedule, or alter the distribution ledger. Follow the project publication workflow when separately authorized. Set the Short related video only after the real long-form video ID exists.</p></aside></main>'
    )
    css = """
:root{--ink:#eee7d8;--gold:#e2b45d;--bg:#111820;--panel:#1b2732;--line:#53616c}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}header,nav,main{max-width:1000px;margin:auto;padding:22px}header{padding-top:42px}header p{color:#b8c4cc}h1{font:500 clamp(30px,5vw,52px)/1.05 Georgia,serif}nav{display:flex;gap:10px;flex-wrap:wrap;border-block:1px solid var(--line)}a{color:var(--gold)}main{display:grid;gap:18px}.copy-block,.thumb,aside,main>section{background:var(--panel);border:1px solid var(--line);padding:16px}.copy-block>div{display:flex;align-items:center;justify-content:space-between}.copy-block h2{font-size:16px;margin:0}.copy-block button{background:var(--gold);border:0;padding:8px 14px;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;margin:14px 0 0}.thumb img{display:block;width:min(100%,720px);aspect-ratio:16/9;object-fit:cover}video{display:block;width:min(100%,420px);max-height:68vh;background:#000}button:focus-visible,a:focus-visible{outline:3px solid #70d4e0;outline-offset:3px}
"""
    js = """
async function copyText(value){if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(value);return}const area=document.createElement('textarea');area.value=value;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();document.execCommand('copy');area.remove()}
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{const node=document.getElementById(button.dataset.copy);const old=button.textContent;await copyText(node.textContent);button.textContent='Copied';setTimeout(()=>button.textContent=old,1200)}));
"""
    page = (
        '<!doctype html><html lang="en" translate="no"><head><meta charset="utf-8">'
        '<meta name="google" content="notranslate"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(config["episode"])} · Hindi upload kit</title><style>{css}</style></head>'
        f'<body>{body}<script>{js}</script></body></html>'
    )
    return page


class KitAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: Dict[str, int] = {}
        self.links: List[str] = []
        self.copy_targets: List[str] = []
        self.scripts: List[str] = []
        self.copy_text: Dict[str, str] = {}
        self._pre_id: Optional[str] = None
        self._inside_script = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            ident = str(values["id"])
            self.ids[ident] = self.ids.get(ident, 0) + 1
        if values.get("data-copy"):
            self.copy_targets.append(str(values["data-copy"]))
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(str(values[key]))
        if tag == "script":
            self._inside_script = True
        if tag == "pre" and values.get("id"):
            self._pre_id = str(values["id"])
            self.copy_text.setdefault(self._pre_id, "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._inside_script = False
        if tag == "pre":
            self._pre_id = None

    def handle_data(self, data: str) -> None:
        if self._inside_script:
            self.scripts.append(data)
        if self._pre_id is not None:
            self.copy_text[self._pre_id] += data


def validate_upload_kit(path: Path, package_root: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    payload = path.read_bytes()
    expected_payload = build_upload_kit(config, package_root).encode("utf-8")
    if payload != expected_payload:
        raise DeliveryError("upload_kit.html is not byte-for-byte reproducible from the approved config")
    try:
        page = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeliveryError("upload_kit.html is not valid UTF-8") from exc
    parser = KitAudit()
    parser.feed(page)
    for ident in REQUIRED_KIT_IDS:
        if parser.ids.get(ident) != 1 or parser.copy_targets.count(ident) != 1:
            raise DeliveryError(f"upload_kit.html must contain exactly one copy block for {ident}")
    missing_copy_targets = sorted(target for target in parser.copy_targets if parser.ids.get(target) != 1)
    if missing_copy_targets:
        raise DeliveryError(f"Upload-kit buttons reference missing/non-unique IDs: {missing_copy_targets}")
    duplicates = sorted(ident for ident, count in parser.ids.items() if count > 1)
    if duplicates:
        raise DeliveryError(f"Duplicate HTML IDs in upload kit: {duplicates}")
    expected_copy = {
        "hi-title": config["metadata"]["title"],
        "hi-description": description_with_chapters(config["metadata"]),
        "common-tags": ", ".join(config["metadata"]["tags"]),
        "short-hi-title": config["metadata"]["shortTitle"],
        "short-hi-desc": config["metadata"]["shortDescription"],
        "short-tags": ", ".join(config["metadata"]["shortTags"]),
    }
    if 'operatorNotes' in config['metadata']:
        expected_copy['operator-notes'] = config['metadata']['operatorNotes']
    for ident, expected in expected_copy.items():
        if parser.copy_text.get(ident) != expected:
            raise DeliveryError(f"upload_kit.html copy content is stale or incorrect for {ident}")
    lowered = page.lower()
    if "publication not authorized" not in lowered or "does not grant permission to upload" not in lowered:
        raise DeliveryError("upload_kit.html is missing the explicit non-publication boundary")
    if html.escape(config["channel"]["handle"]) not in page:
        raise DeliveryError("upload_kit.html is missing the approved Hindi channel handle")
    if parser.ids.get("hi-upload-settings") != 1 or upload_settings_html(config) not in page:
        raise DeliveryError("upload_kit.html does not contain the exact non-authorizing Hindi upload settings")
    checked_links = 0
    for raw in parser.links:
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc:
            raise DeliveryError(f"External link is not allowed in the zero-context final kit: {raw}")
        if not parsed.path:
            continue
        target = (package_root / unquote(parsed.path)).resolve()
        if not _is_below(target, package_root.resolve()) or not target.is_file():
            raise DeliveryError(f"Broken or escaping upload-kit link: {raw}")
        checked_links += 1
    javascript = "not-run-node-unavailable"
    if shutil.which("node"):
        for script in parser.scripts:
            try:
                subprocess.run(["node", "--check", "-"], input=script, text=True, capture_output=True, check=True)
            except subprocess.CalledProcessError as exc:
                raise DeliveryError(f"upload-kit JavaScript syntax failed: {exc.stderr.strip()}") from exc
        javascript = "passed"
    return {
        "requiredCopyIds": list(REQUIRED_KIT_IDS),
        "copyContent": "matches-config",
        "byteReproducible": True,
        "uploadSettings": config["metadata"]["uploadSettings"],
        "nonPublicationBoundary": "present",
        "localLinks": checked_links,
        "javascript": javascript,
    }


def _expected_output_paths(config: Mapping[str, Any]) -> Dict[str, str]:
    outputs = {asset_id: config["assets"][asset_id]["destination"] for asset_id in COPY_ASSET_IDS}
    outputs.update(GENERATED_OUTPUTS)
    outputs.update({f'thumbnail-alt-{i+1}': row['destination'] for i,row in enumerate(config.get('thumbnailAlternatives', []))})
    if len(set(outputs.values())) != len(outputs):
        raise DeliveryError("Final output destinations collide")
    return outputs


def _package_inventory(root: Path) -> List[str]:
    if root.is_symlink():
        raise DeliveryError(f"Package root cannot be a symlink: {root}")
    found: List[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DeliveryError(f"Final package may not contain symlinks: {path}")
        if path.is_file():
            found.append(path.relative_to(root).as_posix())
        elif path.is_dir() and path.relative_to(root).as_posix() not in ALLOWED_FINAL_DIRECTORIES:
            raise DeliveryError(f"Unexpected directory in final package: {path.relative_to(root)}")
    return found


def validate_package_structure(config: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    outputs = _expected_output_paths(config)
    expected = set(outputs.values()) | {"FINAL_MANIFEST.json"}
    actual = set(_package_inventory(root))
    if actual != expected:
        raise DeliveryError(f"Final whitelist mismatch; extra={sorted(actual-expected)}, missing={sorted(expected-actual)}")
    manifest = _json(root / "FINAL_MANIFEST.json")
    manifest_keys = {
        "schemaVersion", "layout", "episode", "status", "publishingAuthorized",
        "publishAt", "channelHandle", "channelId", "language", "title",
        "titleApproved", "thumbnailApproved", "thumbnailChoice", "assets",
        "metadata", "short", "approvals", "validation", "provenance", "createdAtUTC",
    }
    _keys(manifest, required=manifest_keys, allowed=manifest_keys, label="FINAL_MANIFEST.json")
    if manifest.get("schemaVersion") != MANIFEST_SCHEMA_VERSION or manifest.get("layout") != LAYOUT:
        raise DeliveryError("FINAL_MANIFEST.json schema/layout mismatch")
    if (
        manifest.get("status") != "verified local final assets; publication not authorized"
        or manifest.get("publishingAuthorized") is not False
        or manifest.get("publishAt") is not None
    ):
        raise DeliveryError("FINAL_MANIFEST.json may not authorize publication")
    rows = manifest.get("assets")
    if (
        not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
        or [row.get("id") for row in rows] != list(_expected_output_paths(config))
    ):
        raise DeliveryError("FINAL_MANIFEST.json asset IDs do not match the strict final set")
    for index, row in enumerate(rows):
        asset_row_keys = {"id", "canonical", "sha256", "bytes"}
        _keys(row, required=asset_row_keys, allowed=asset_row_keys, label=f"FINAL_MANIFEST.json.assets[{index}]")
    if manifest.get("episode") != config["episode"] or manifest.get("language") != "hi":
        raise DeliveryError("FINAL_MANIFEST.json episode/language mismatch")
    if manifest.get("channelHandle") != config["channel"]["handle"] or manifest.get("channelId") != config["channel"]["id"]:
        raise DeliveryError("FINAL_MANIFEST.json channel mismatch")
    if manifest.get("title") != config["metadata"]["title"] or manifest.get("thumbnailChoice") != config["thumbnail"]["choice"]:
        raise DeliveryError("FINAL_MANIFEST.json approved title/thumbnail mismatch")
    if manifest.get("titleApproved") is not True or manifest.get("thumbnailApproved") is not True:
        raise DeliveryError("FINAL_MANIFEST.json final approval flags are missing")

    expected_metadata = {
        "description": description_with_chapters(config["metadata"]),
        "tags": config["metadata"]["tags"],
        "hashtags": config["metadata"]["hashtags"],
        "pinnedComment": config["metadata"]["pinnedComment"],
        "chapters": config["metadata"]["chapters"],
        "uploadSettings": config["metadata"]["uploadSettings"],
    }
    if 'operatorNotes' in config['metadata']:
        expected_metadata['operatorNotes'] = config['metadata']['operatorNotes']
    if manifest.get("metadata") != expected_metadata:
        raise DeliveryError("FINAL_MANIFEST.json metadata differs from the approved config")
    short = manifest.get("short")
    short_keys = {
        "humanReview", "revision", "duration", "ctaText", "ctaSeconds", "title",
        "description", "tags", "relatedVideoId", "uploadAllowed",
    }
    if not isinstance(short, dict):
        raise DeliveryError("FINAL_MANIFEST.json Short record is missing")
    _keys(short, required=short_keys, allowed=short_keys, label="FINAL_MANIFEST.json.short")
    expected_short = {
        "humanReview": config['short']['humanReview'],
        "revision": config["short"]["revision"],
        "ctaText": config["short"]["ctaText"],
        "ctaSeconds": config["short"]["ctaSeconds"],
        "title": config["metadata"]["shortTitle"],
        "description": config["metadata"]["shortDescription"],
        "tags": config["metadata"]["shortTags"],
        "relatedVideoId": None,
        "uploadAllowed": False,
    }
    if any(short.get(key) != value for key, value in expected_short.items()):
        raise DeliveryError("FINAL_MANIFEST.json Short state differs or attempts upload authority")
    if not isinstance(short.get("duration"), (int, float)) or isinstance(short.get("duration"), bool) or short["duration"] <= 0:
        raise DeliveryError("FINAL_MANIFEST.json Short duration is invalid")
    if manifest.get("approvals") != config["approvals"]:
        raise DeliveryError("FINAL_MANIFEST.json approvals differ from the approved config")

    validation = manifest.get("validation")
    validation_keys = {"media", "subtitles", "thumbnail", "strictFinalWhitelist", "sourcesDeleted"}
    if not isinstance(validation, dict):
        raise DeliveryError("FINAL_MANIFEST.json validation record is missing")
    _keys(validation, required=validation_keys, allowed=validation_keys, label="FINAL_MANIFEST.json.validation")
    if validation["strictFinalWhitelist"] is not True or validation["sourcesDeleted"] is not False:
        raise DeliveryError("FINAL_MANIFEST.json validation safety flags are invalid")
    if not all(isinstance(validation[key], dict) for key in ("media", "subtitles", "thumbnail")):
        raise DeliveryError("FINAL_MANIFEST.json validation reports must be objects")

    provenance = manifest.get("provenance")
    provenance_keys = {
        "config", "configSha256", "thumbnailMethod", "thumbnailTextlessSource",
        "thumbnailTextlessSha256", "thumbnailTitleRecipe", "thumbnailTitleRecipeSha256",
        "thumbnailSource", "thumbnailSourceSha256",
    }
    if not isinstance(provenance, dict):
        raise DeliveryError("FINAL_MANIFEST.json provenance is missing")
    _keys(provenance, required=provenance_keys, allowed=provenance_keys, label="FINAL_MANIFEST.json.provenance")
    expected_thumbnail_provenance = {
        "thumbnailMethod": config["thumbnail"]["method"],
        "thumbnailTextlessSource": config["thumbnail"]["textlessSource"],
        "thumbnailTextlessSha256": config["thumbnail"]["textlessSha256"],
        "thumbnailTitleRecipe": config["thumbnail"]["titleRecipe"],
        "thumbnailTitleRecipeSha256": config["thumbnail"]["titleRecipeSha256"],
        "thumbnailSource": config["thumbnail"]["source"],
        "thumbnailSourceSha256": config["thumbnail"]["sha256"],
    }
    if any(provenance[key] != value for key, value in expected_thumbnail_provenance.items()):
        raise DeliveryError("FINAL_MANIFEST.json thumbnail provenance differs from config")
    if not _valid_utc_timestamp(manifest.get("createdAtUTC")):
        raise DeliveryError("FINAL_MANIFEST.json createdAtUTC must be second-precision RFC3339 UTC")

    by_id = {row["id"]: row for row in rows}
    for asset_id, relative in outputs.items():
        row = by_id[asset_id]
        if row.get("canonical") != "deliverables/" + relative:
            raise DeliveryError(f"Manifest canonical path mismatch for {asset_id}")
        path = root / relative
        if row.get("sha256") != sha256_file(path) or row.get("bytes") != path.stat().st_size:
            raise DeliveryError(f"Manifest hash/size mismatch for {asset_id}")
    kit = validate_upload_kit(root / "upload_kit.html", root, config)
    return {"exactWhitelist": True, "fileCount": len(actual), "manifestHashes": "passed", "uploadKit": kit}


def _validate_subtitle_pairs(config: Mapping[str, Any], root: Path, media_reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    assets = config["assets"]
    long_duration = float(media_reports["long-hi"]["videoDurationSeconds"])
    short_duration = float(media_reports["short-hi"]["videoDurationSeconds"])
    if float(config["metadata"]["chapters"][-1]["seconds"]) >= long_duration:
        raise DeliveryError("The final chapter must start before the long-form master ends")
    if float(config["short"]["ctaSeconds"]) >= short_duration:
        raise DeliveryError("The Short CTA cannot consume the entire Short runtime")
    reports = {
        "subtitles-hi": parse_srt(root / assets["subtitles-hi"]["destination"], language="hi", media_duration=long_duration),
        "subtitles-en": parse_srt(root / assets["subtitles-en"]["destination"], language="en", media_duration=long_duration),
        "short-subtitles-hi": parse_srt(root / assets["short-subtitles-hi"]["destination"], language="hi", media_duration=short_duration),
        "short-subtitles-en": parse_srt(root / assets["short-subtitles-en"]["destination"], language="en", media_duration=short_duration),
    }
    for left, right in (("subtitles-hi", "subtitles-en"), ("short-subtitles-hi", "short-subtitles-en")):
        if reports[left]["timings"] != reports[right]["timings"]:
            raise DeliveryError(f"{left} and {right} must use the same authored timeline")
    return {key: {field: value for field, value in report.items() if field != "timings"} for key, report in reports.items()}


def _validate_all_media(config: Mapping[str, Any], root: Path) -> Dict[str, Dict[str, Any]]:
    reports = {
        asset_id: validate_media_file(
            root / config["assets"][asset_id]["destination"],
            asset_id,
            config["assets"][asset_id]["expected"],
        )
        for asset_id in MEDIA_ASSET_IDS
    }
    first, second = reports["short-hi"], reports["short-hi-social"]
    for field in ("width", "height", "fps", "frames", "videoDurationSeconds"):
        if first[field] != second[field]:
            raise DeliveryError(f"YouTube/Reels Short picture contract differs at {field}")
    if first["videoPayloadSha256"] != second["videoPayloadSha256"]:
        raise DeliveryError("YouTube/Reels Short must carry the same picture packets; only loudness may differ")
    if first["videoTimelineSha256"] != second["videoTimelineSha256"]:
        raise DeliveryError("YouTube/Reels Short must carry the same picture timeline; only loudness may differ")
    return reports


def _assert_source_hashes(episode: Path, config: Mapping[str, Any]) -> None:
    for asset_id in COPY_ASSET_IDS:
        row = config["assets"][asset_id]
        path = episode_path(episode, row["source"], f"assets.{asset_id}.source", must_exist=True)
        if sha256_file(path) != row["sha256"]:
            raise DeliveryError(f"Source SHA-256 mismatch for {asset_id}: {path}")
    source = episode_path(episode, config["thumbnail"]["source"], "thumbnail.source", must_exist=True)
    if sha256_file(source) != config["thumbnail"]["sha256"]:
        raise DeliveryError(f"Approved thumbnail source SHA-256 mismatch: {source}")
    for alternative in config.get('thumbnailAlternatives', []):
        source = episode_path(episode, alternative['source'], 'thumbnail alternative', must_exist=True)
        if sha256_file(source) != alternative['sha256']:
            raise DeliveryError('Thumbnail alternative source changed')


def _regenerated_thumbnail_report(
    episode: Path,
    config: Mapping[str, Any],
    root: Path,
    recorded: Mapping[str, Any],
) -> Dict[str, Any]:
    report_keys = {"source", "png", "jpeg", "sameApprovedComposition"}
    _keys(recorded, required=report_keys, allowed=report_keys, label="FINAL_MANIFEST.json.validation.thumbnail")
    if recorded["sameApprovedComposition"] is not True:
        raise DeliveryError("Thumbnail lineage no longer asserts one approved composition")
    source = episode_path(episode, config["thumbnail"]["source"], "thumbnail.source", must_exist=True)
    staged_png = root / THUMBNAIL_OUTPUTS["thumbnail-master"]
    staged_jpg = root / THUMBNAIL_OUTPUTS["thumbnail-hi"]
    with tempfile.TemporaryDirectory(prefix=".thumbnail-repro-", dir=str(root.parent)) as temporary:
        temporary_root = Path(temporary)
        regenerated = render_thumbnail(
            source,
            temporary_root / "thumb_hi_titled.png",
            temporary_root / "thumb_hi_upload_1280.jpg",
            int(config["thumbnail"]["jpegMaxBytes"]),
        )
        comparisons = (
            (temporary_root / "thumb_hi_titled.png", staged_png, "PNG"),
            (temporary_root / "thumb_hi_upload_1280.jpg", staged_jpg, "JPEG"),
        )
        for rebuilt, staged, label in comparisons:
            if rebuilt.stat().st_size != staged.stat().st_size or sha256_file(rebuilt) != sha256_file(staged):
                raise DeliveryError(f"Staged thumbnail {label} is not reproducible from the approved titled source")
    if regenerated != recorded:
        raise DeliveryError("Regenerated thumbnail audit differs from FINAL_MANIFEST.json")
    return regenerated


def _asset_rows(config: Mapping[str, Any], root: Path) -> List[Dict[str, Any]]:
    rows = []
    for asset_id, relative in _expected_output_paths(config).items():
        path = root / relative
        rows.append({"id": asset_id, "canonical": "deliverables/" + relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def _manifest(
    episode: Path,
    config_path: Path,
    config_sha256: str,
    config: Mapping[str, Any],
    root: Path,
    media_reports: Mapping[str, Any],
    subtitle_reports: Mapping[str, Any],
    thumbnail_report: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = config["metadata"]
    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "layout": LAYOUT,
        "episode": episode.name,
        "status": "verified local final assets; publication not authorized",
        "publishingAuthorized": False,
        "publishAt": None,
        "channelHandle": config["channel"]["handle"],
        "channelId": config["channel"]["id"],
        "language": "hi",
        "title": metadata["title"],
        "titleApproved": True,
        "thumbnailApproved": True,
        "thumbnailChoice": config["thumbnail"]["choice"],
        "assets": _asset_rows(config, root),
        "metadata": {
            **({'operatorNotes': metadata['operatorNotes']} if 'operatorNotes' in metadata else {}),
            "description": description_with_chapters(metadata),
            "tags": metadata["tags"],
            "hashtags": metadata["hashtags"],
            "pinnedComment": metadata["pinnedComment"],
            "chapters": metadata["chapters"],
            "uploadSettings": config["metadata"]["uploadSettings"],
        },
        "short": {
            "humanReview": config["short"]["humanReview"],
            "revision": config["short"]["revision"],
            "duration": media_reports["short-hi"]["videoDurationSeconds"],
            "ctaText": config["short"]["ctaText"],
            "ctaSeconds": config["short"]["ctaSeconds"],
            "title": metadata["shortTitle"],
            "description": metadata["shortDescription"],
            "tags": metadata["shortTags"],
            "relatedVideoId": None,
            "uploadAllowed": False,
        },
        "approvals": config["approvals"],
        "validation": {
            "media": media_reports,
            "subtitles": subtitle_reports,
            "thumbnail": thumbnail_report,
            "strictFinalWhitelist": True,
            "sourcesDeleted": False,
        },
        "provenance": {
            "config": config_path.relative_to(episode).as_posix(),
            "configSha256": config_sha256,
            "thumbnailMethod": config["thumbnail"]["method"],
            "thumbnailTextlessSource": config["thumbnail"]["textlessSource"],
            "thumbnailTextlessSha256": config["thumbnail"]["textlessSha256"],
            "thumbnailTitleRecipe": config["thumbnail"]["titleRecipe"],
            "thumbnailTitleRecipeSha256": config["thumbnail"]["titleRecipeSha256"],
            "thumbnailSource": config["thumbnail"]["source"],
            "thumbnailSourceSha256": config["thumbnail"]["sha256"],
        },
        "createdAtUTC": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def stage_package(episode: Path, config_path: Path) -> Dict[str, Any]:
    episode = episode.resolve()
    config_path = config_path.resolve()
    snapshot = _config_snapshot(config_path)
    config, gates = analyze_config(episode, config_path, snapshot=snapshot)
    if gates:
        raise DeliveryError("Final inputs are not ready: " + ", ".join(gates))
    _assert_source_hashes(episode, config)
    stage = episode_path(episode, config["stagingDir"], "stagingDir")
    if stage.exists():
        raise DeliveryError(f"Staging path already exists; preserve it and choose a new version: {stage}")
    stage.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hindi-delivery-building-", dir=str(stage.parent)) as temporary:
        root = Path(temporary) / "package"
        root.mkdir()
        for asset_id in COPY_ASSET_IDS:
            row = config["assets"][asset_id]
            source = episode_path(episode, row["source"], f"assets.{asset_id}.source", must_exist=True)
            destination = root / row["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            destination_hash = sha256_file(destination)
            if destination_hash != row["sha256"] or sha256_file(source) != row["sha256"]:
                raise DeliveryError(f"Copied bytes no longer match the configured source SHA-256 for {asset_id}")
        thumbnail_source = episode_path(episode, config["thumbnail"]["source"], "thumbnail.source", must_exist=True)
        thumbnail_report = render_thumbnail(
            thumbnail_source,
            root / THUMBNAIL_OUTPUTS["thumbnail-master"],
            root / THUMBNAIL_OUTPUTS["thumbnail-hi"],
            int(config["thumbnail"]["jpegMaxBytes"]),
        )
        for alternative in config.get('thumbnailAlternatives', []):
            source = episode_path(episode, alternative['source'], 'thumbnail alternative', must_exist=True)
            destination = root / alternative['destination']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            image = _probe_image(destination)
            if (image['width'], image['height']) != (1280,720) or sha256_file(destination) != alternative['sha256']:
                raise DeliveryError('Thumbnail alternative copy or geometry mismatch')
        (root / "upload_kit.html").write_text(build_upload_kit(config, root), encoding="utf-8")
        media_reports = _validate_all_media(config, root)
        subtitle_reports = _validate_subtitle_pairs(config, root, media_reports)
        manifest = _manifest(episode, config_path, snapshot[1], config, root, media_reports, subtitle_reports, thumbnail_report)
        (root / "FINAL_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        structural = _validate_package_root(episode, config_path, config, snapshot[1], root)
        os.replace(root, stage)
    return {"status": "staged-and-verified", "stage": str(stage), "files": structural["files"], "publishingAuthorized": False}


def _validate_package_root(
    episode: Path,
    config_path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    root: Path,
) -> Dict[str, Any]:
    _assert_source_hashes(episode, config)
    structure = validate_package_structure(config, root)
    media_reports = _validate_all_media(config, root)
    subtitle_reports = _validate_subtitle_pairs(config, root, media_reports)
    png = _probe_image(root / THUMBNAIL_OUTPUTS["thumbnail-master"])
    jpg_path = root / THUMBNAIL_OUTPUTS["thumbnail-hi"]
    jpg = _probe_image(jpg_path)
    if (png["width"], png["height"]) != (1280, 720) or (jpg["width"], jpg["height"]) != (1280, 720):
        raise DeliveryError("Staged thumbnails are not both 1280x720")
    if jpg_path.stat().st_size > config["thumbnail"]["jpegMaxBytes"]:
        raise DeliveryError("Staged upload JPEG exceeds its configured byte ceiling")
    manifest = _json(root / "FINAL_MANIFEST.json")
    provenance = manifest.get("provenance", {})
    expected_config_path = config_path.relative_to(episode).as_posix()
    if provenance.get("config") != expected_config_path or provenance.get("configSha256") != config_sha256:
        raise DeliveryError("Staging was built from a different config revision; preserve it and stage a new version")
    if manifest.get("validation", {}).get("media") != media_reports:
        raise DeliveryError("Current staged media measurements differ from FINAL_MANIFEST.json")
    if manifest.get("validation", {}).get("subtitles") != subtitle_reports:
        raise DeliveryError("Current staged subtitle audit differs from FINAL_MANIFEST.json")
    thumbnail_report = _regenerated_thumbnail_report(
        episode,
        config,
        root,
        manifest.get("validation", {}).get("thumbnail", {}),
    )
    if manifest.get("validation", {}).get("thumbnail") != thumbnail_report:
        raise DeliveryError("Current staged thumbnail audit differs from FINAL_MANIFEST.json")
    if manifest.get("short", {}).get("duration") != media_reports["short-hi"]["videoDurationSeconds"]:
        raise DeliveryError("FINAL_MANIFEST.json Short duration differs from the verified Short")
    return {"files": structure["fileCount"], "publishingAuthorized": False}


def validate_staged_package(episode: Path, config_path: Path) -> Dict[str, Any]:
    episode = episode.resolve()
    config_path = config_path.resolve()
    snapshot = _config_snapshot(config_path)
    config, gates = analyze_config(episode, config_path, snapshot=snapshot)
    if gates:
        raise DeliveryError("Config is no longer final-ready: " + ", ".join(gates))
    stage = episode_path(episode, config["stagingDir"], "stagingDir")
    if not stage.is_dir():
        raise DeliveryError(f"Staged package does not exist: {stage}")
    checked = _validate_package_root(episode, config_path, config, snapshot[1], stage)
    return {
        "status": "verified-staging",
        "stage": str(stage),
        "files": checked["files"],
        "publishingAuthorized": False,
    }


def _tree_hashes(root: Path) -> Dict[str, str]:
    return {path: sha256_file(root / path) for path in _package_inventory(root)}


@contextmanager
def _exclusive_promotion_lock(episode: Path):
    lock_path = episode / "work/.hindi-delivery-promote.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DeliveryError("Another final-delivery promotion is already active for this episode") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing even an empty destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes, destination_bytes = os.fsencode(source), os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
    elif hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, 0x1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise DeliveryError("This platform has no supported atomic no-replace rename; refusing promotion")
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise DeliveryError(f"Promotion destination appeared concurrently; refusing overwrite: {destination}")
        raise DeliveryError(f"Atomic no-replace promotion failed: {os.strerror(error)}")


def promote_to_deliverables(episode: Path, config_path: Path) -> Dict[str, Any]:
    episode = episode.resolve()
    config_path = config_path.resolve()
    snapshot = _config_snapshot(config_path)
    config, gates = analyze_config(episode, config_path, snapshot=snapshot)
    if gates:
        raise DeliveryError("Config is no longer final-ready: " + ", ".join(gates))
    stage = episode_path(episode, config["stagingDir"], "stagingDir")
    if not stage.is_dir():
        raise DeliveryError(f"Staged package does not exist: {stage}")
    checked = _validate_package_root(episode, config_path, config, snapshot[1], stage)
    destination = episode / "deliverables"
    with tempfile.TemporaryDirectory(prefix=".hindi-delivery-promote-", dir=str(stage.parent)) as temporary:
        candidate = Path(temporary) / "deliverables"
        shutil.copytree(stage, candidate, copy_function=shutil.copy2)
        if _tree_hashes(candidate) != _tree_hashes(stage):
            raise DeliveryError("Promotion copy hash verification failed")
        candidate_checked = _validate_package_root(episode, config_path, config, snapshot[1], candidate)
        if candidate_checked["files"] != checked["files"]:
            raise DeliveryError("Promotion candidate inventory differs after full validation")
        with _exclusive_promotion_lock(episode):
            if destination.exists() or destination.is_symlink():
                if destination.is_dir() and _tree_hashes(destination) == _tree_hashes(stage):
                    return {
                        "status": "already-promoted-identical",
                        "deliverables": str(destination),
                        "files": checked["files"],
                        "publishingAuthorized": False,
                    }
                raise DeliveryError(
                    "deliverables already exists and is not byte-identical. Refusing overwrite, cleanup, or variant bloat; archive/migrate it in a separately reviewed operation."
                )
            _rename_noreplace(candidate, destination)
    return {"status": "promoted-local-finals", "deliverables": str(destination), "files": checked["files"], "publishingAuthorized": False}


def main() -> int:
    global GPU_FINISHING_PROBE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--gpu-finishing-probe", action="store_true",
                        help="Qualified H.264 MP4 packet preflight plus complete decode/count on the configured GPU worker")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check-config", action="store_true", help="read-only readiness report")
    actions.add_argument("--stage", action="store_true", help="build a new verified package below work/")
    actions.add_argument("--validate-staging", action="store_true", help="revalidate the existing staged package")
    actions.add_argument("--promote-to-deliverables", action="store_true", help="explicitly copy verified staging into a previously absent deliverables/")
    args = parser.parse_args()
    GPU_FINISHING_PROBE = args.gpu_finishing_probe
    episode, config_path = args.episode.resolve(), args.config.resolve()
    try:
        if not _is_below(config_path, episode):
            raise DeliveryError("Config must live inside the episode")
        if args.check_config:
            _config, gates = analyze_config(episode, config_path)
            result = {"status": "ready" if not gates else "blocked", "ready": not gates, "gates": gates, "publishingAuthorized": False}
        elif args.stage:
            result = stage_package(episode, config_path)
        elif args.validate_staging:
            result = validate_staged_package(episode, config_path)
        else:
            result = promote_to_deliverables(episode, config_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ready", True) else 2
    except DeliveryError as exc:
        print(json.dumps({"status": "refused", "error": str(exc), "publishingAuthorized": False}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
