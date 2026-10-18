#!/usr/bin/env python3
"""Author finite H3 continuation manifests from a measured blocked timeline.

This tool is deliberately offline.  It neither imports a cloud SDK nor starts,
stops, uploads to, or contacts an H100 worker.  Its only media mutation is to
decode the exact last displayed frame of an already accepted, silent H3 window.

The builder refuses to run from forecasts or partial review state.  A valid
input must be the non-dry ``timeline-readiness.json`` emitted after every Hindi
narration file and every source-bound approved Grok dialogue take has been
measured.  Remaining rows may be exact H3 frame deficits and, in the explicit
measurement-only mode, a pending score approval. Score approval is preserved
as a delivery gate; it does not change how much picture the speech requires.

One wave authors one bounded continuation per deficient scene. Duration and
expected frame count are explicit provider-contract inputs. After dropping the
repeated conditioning frame, any measured residual remains explicit; a later
wave starts from the reviewed endpoint of the accepted continuation.
That is what makes every link a true delivered-pixel continuation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import canonical_sha256 as _canonical_sha, is_below as _is_below, sha256_file as _sha256
from _common import MediaWorkerRequired as _MediaWorkerRequired, extract_exact_frame as _extract_exact_frame


DELIVERY_FPS = 24
ACCEPTED = {"accept", "accepted", "approve", "approved", "approved-bounded", "use"}
MEASURED_CLOCK_KIND = "measured-speech-clock-not-renderable-edit-plan"
DEFAULT_STATIC_DEADLINE = 4_102_444_800  # 2100-01-01; client still caps itself to 7,000 s.


class DeficitPlanBlocked(RuntimeError):
    """Current evidence cannot safely authorize a continuation manifest."""


@dataclass(frozen=True)
class Evidence:
    path: Path
    sha256: str


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _immutable_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise DeficitPlanBlocked(f"Immutable output would change: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeficitPlanBlocked(f"{label} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeficitPlanBlocked(f"{label} must contain one JSON object: {path}")
    return value


def _normal_verdict(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value") or value.get("status") or ""
    return str(value or "").strip().lower().replace("_", "-")


def _require_sha(value: Any, label: str) -> str:
    digest = str(value or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DeficitPlanBlocked(f"{label} is not an explicit SHA-256 digest")
    return digest


def _resolve(episode: Path, repo: Path, owner: Path, raw: str | Path) -> Path:
    value = Path(raw)
    if value.is_absolute():
        return value.resolve()
    base = owner.parent if str(raw).startswith(("./", "../")) else episode
    return (base / value).resolve()


def _portable(path: Path, episode: Path) -> str:
    try:
        return str(path.resolve().relative_to(episode.resolve()))
    except ValueError:
        return str(path.resolve())


def _scene_number(scene_id: str) -> int:
    match = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*-s(\d+)", scene_id)
    if not match:
        raise DeficitPlanBlocked(f"Invalid scene id: {scene_id}")
    return int(match.group(1))


def _ledger_entries(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = doc.get("entries", doc.get("shots", doc))
    if isinstance(raw, dict):
        return {str(key): row for key, row in raw.items() if isinstance(row, dict)}
    if isinstance(raw, list):
        result: dict[str, dict[str, Any]] = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            key = row.get("sceneId") or row.get("shotId") or row.get("id")
            if key:
                result[str(key)] = row
        return result
    raise DeficitPlanBlocked("H3 review ledger entries must be a mapping or array")


def _verify_evidence_path(
    *, episode: Path, repo: Path, owner: Path, row: dict[str, Any], label: str,
    require_episode: bool = True,
) -> Evidence:
    path = _resolve(episode, repo, owner, row.get("path", ""))
    if require_episode and not _is_below(path, episode):
        raise DeficitPlanBlocked(f"{label} escapes the episode root: {path}")
    if not path.is_file():
        raise DeficitPlanBlocked(f"{label} is missing: {path}")
    declared = _require_sha(row.get("sha256"), f"{label} sha256")
    actual = _sha256(path)
    if actual != declared:
        raise DeficitPlanBlocked(f"{label} changed: {path}")
    return Evidence(path=path, sha256=actual)


def _verify_current_timeline(
    episode: Path,
    repo: Path,
    config_path: Path,
    config: dict[str, Any],
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], Path, Path, Path,
    dict[str, dict[str, Any]], dict[str, dict[str, Any]],
]:
    readiness_path = _resolve(
        episode, repo, config_path, config.get("timelineReadiness", "work/assembly-v1/timeline-readiness.json")
    )
    clock_path = _resolve(
        episode, repo, config_path, config.get("measuredClock", "work/assembly-v1/measured-clock.json")
    )
    timeline_inputs_path = _resolve(
        episode, repo, config_path, config.get("timelineInputs", "work/assembly-v1/timeline-inputs.json")
    )
    for label, path in (
        ("timeline readiness", readiness_path),
        ("measured clock", clock_path),
        ("timeline inputs", timeline_inputs_path),
    ):
        if not _is_below(path, episode):
            raise DeficitPlanBlocked(f"{label} must remain below the episode root: {path}")

    readiness = _read_object(readiness_path, "timeline readiness")
    if readiness.get("dryRun") is not False:
        raise DeficitPlanBlocked("Timeline readiness is dry-run evidence, not a measured production clock")
    if readiness.get("errors"):
        raise DeficitPlanBlocked("Timeline readiness contains invalid inputs")
    narration_count = int(readiness.get("narrationScenes", -1))
    dialogue_count = int(readiness.get("dialogueScenes", -1))
    if narration_count < 1 or dialogue_count < 1:
        raise DeficitPlanBlocked("Measured production must contain narration and Grok dialogue")
    if int(readiness.get("measuredNarration", -1)) != narration_count:
        raise DeficitPlanBlocked("Not every current Hindi narration line has been measured")
    if int(readiness.get("measuredDialogue", -1)) != dialogue_count:
        raise DeficitPlanBlocked("Not every current approved Grok dialogue take has been measured")
    picture_count = int(readiness.get("pictureScenes", 0))
    if int(readiness.get("storyboardScenes", -1)) != narration_count + dialogue_count + picture_count:
        raise DeficitPlanBlocked("Timeline scene counts are internally inconsistent")

    missing = readiness.get("missing")
    if not isinstance(missing, list):
        raise DeficitPlanBlocked("Timeline readiness missing list is invalid")
    allowed_missing = {"h3-coverage-deficit"}
    if readiness.get("mode") == "measurement-only-score-gates-preserved":
        allowed_missing.add("score-admission")
    foreign_missing = [row for row in missing if row.get("type") not in allowed_missing]
    if foreign_missing:
        kinds = sorted({str(row.get("type")) for row in foreign_missing})
        raise DeficitPlanBlocked(
            "Continuation planning waits for all non-picture gates: " + ", ".join(kinds)
        )

    clock_binding = readiness.get("measuredClock")
    if not isinstance(clock_binding, dict):
        raise DeficitPlanBlocked("Readiness has no hash-bound measured clock")
    clock_evidence = _verify_evidence_path(
        episode=episode, repo=repo, owner=readiness_path, row=clock_binding,
        label="measured clock",
    )
    if clock_evidence.path != clock_path:
        raise DeficitPlanBlocked("Configured measured clock is not the readiness-bound clock")
    clock = _read_object(clock_path, "measured clock")
    if clock.get("kind") != MEASURED_CLOCK_KIND or clock.get("renderable") is not False:
        raise DeficitPlanBlocked("Forecast/edit-plan substitution refused; exact measured clock required")
    fingerprint = _require_sha(clock.get("clockFingerprint"), "measured clock fingerprint")
    if clock_binding.get("clockFingerprint") != fingerprint:
        raise DeficitPlanBlocked("Measured clock fingerprint differs from readiness")
    events = clock.get("events")
    if not isinstance(events, list) or len(events) != int(readiness["storyboardScenes"]):
        raise DeficitPlanBlocked("Measured clock does not cover the complete storyboard")
    if int(clock.get("eventCount", -1)) != len(events):
        raise DeficitPlanBlocked("Measured clock event count is stale")
    if sum(row.get("kind") == "narration" for row in events) != narration_count:
        raise DeficitPlanBlocked("Measured clock narration count differs from readiness")
    if sum(row.get("kind") == "picture" for row in events) != picture_count:
        raise DeficitPlanBlocked("Measured clock picture count differs from readiness")
    if sum(row.get("kind") == "dialogue" for row in events) != dialogue_count:
        raise DeficitPlanBlocked("Measured clock dialogue count differs from readiness")

    clock_inputs = clock.get("inputEvidence")
    if not isinstance(clock_inputs, dict):
        raise DeficitPlanBlocked("Measured clock has no input evidence")
    config_evidence = clock_inputs.get("config")
    if not isinstance(config_evidence, dict):
        raise DeficitPlanBlocked("Measured clock does not bind timeline inputs")
    verified_config = _verify_evidence_path(
        episode=episode, repo=repo, owner=clock_path, row=config_evidence,
        label="timeline inputs",
    )
    if verified_config.path != timeline_inputs_path:
        raise DeficitPlanBlocked("Configured timeline inputs differ from measured clock")

    readiness_docs = readiness.get("documents")
    clock_docs = clock_inputs.get("documents")
    if not isinstance(readiness_docs, dict) or readiness_docs != clock_docs:
        raise DeficitPlanBlocked("Readiness and measured clock document bindings differ")
    required_documents = {
        "storyboard", "narrationManifest", "dialogueManifest",
        "h3ReviewLedger", "productionOverrides", "narrationState",
    }
    absent = sorted(required_documents - set(readiness_docs))
    if absent:
        raise DeficitPlanBlocked("Measured timeline lacks required current documents: " + ", ".join(absent))
    document_evidence: dict[str, Evidence] = {}
    for label, row in readiness_docs.items():
        if not isinstance(row, dict):
            raise DeficitPlanBlocked(f"Invalid document binding: {label}")
        document_evidence[label] = _verify_evidence_path(
            episode=episode, repo=repo, owner=readiness_path, row=row,
            label=f"timeline document {label}",
        )

    clock_media = clock_inputs.get("media")
    readiness_media = readiness.get("media")
    if not isinstance(readiness_media, dict) or readiness_media != clock_media:
        raise DeficitPlanBlocked("Readiness and measured clock media bindings differ")
    media_evidence: dict[str, dict[str, Any]] = {}
    for raw_path, row in readiness_media.items():
        if not isinstance(row, dict):
            raise DeficitPlanBlocked(f"Invalid media evidence: {raw_path}")
        path = Path(raw_path).resolve()
        if not _is_below(path, episode) or not path.is_file():
            raise DeficitPlanBlocked(f"Measured media is missing or outside episode: {path}")
        actual = _sha256(path)
        if actual != _require_sha(row.get("sha256"), f"media {path} sha256"):
            raise DeficitPlanBlocked(f"Measured media changed: {path}")
        media_evidence[str(path)] = dict(row)

    event_by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            raise DeficitPlanBlocked("Measured clock event is not an object")
        scene_id = str(event.get("id") or "")
        _scene_number(scene_id)
        if scene_id in event_by_id:
            raise DeficitPlanBlocked(f"Duplicate measured clock scene: {scene_id}")
        event_by_id[scene_id] = event
        if event.get("kind") == "picture":
            if event.get("audio") is not None:
                raise DeficitPlanBlocked(f"Wordless picture event carries voice audio: {scene_id}")
            continue  # no voice to bind; its picture is checked through the ledger
        media_path = _resolve(episode, repo, clock_path, event.get("audio", ""))
        row = media_evidence.get(str(media_path))
        if row is None or row.get("sha256") != event.get("audioSha256"):
            raise DeficitPlanBlocked(f"Event audio is not current measured media: {scene_id}")
        if event.get("kind") == "dialogue":
            if _resolve(episode, repo, clock_path, event.get("picture", "")) != media_path:
                raise DeficitPlanBlocked(f"Grok dialogue picture/audio were decoupled: {scene_id}")
            admission = event.get("admission")
            if not isinstance(admission, dict):
                raise DeficitPlanBlocked(f"Grok dialogue lacks source-bound review admission: {scene_id}")
            for path_field, sha_field in (
                ("sidecar", "sidecarSha256"),
                ("reviewExport", "reviewExportSha256"),
            ):
                evidence_path = _resolve(episode, repo, clock_path, admission.get(path_field, ""))
                if not _is_below(evidence_path, episode) or not evidence_path.is_file():
                    raise DeficitPlanBlocked(f"Dialogue admission evidence missing: {scene_id} {path_field}")
                if _sha256(evidence_path) != _require_sha(
                    admission.get(sha_field), f"{scene_id} {sha_field}"
                ):
                    raise DeficitPlanBlocked(f"Dialogue admission evidence changed: {scene_id} {path_field}")

    timeline_inputs = _read_object(timeline_inputs_path, "timeline inputs")
    ledger_path = document_evidence["h3ReviewLedger"].path
    configured_ledger = _resolve(
        episode, repo, timeline_inputs_path, timeline_inputs.get("h3ReviewLedger", "")
    )
    if configured_ledger != ledger_path:
        raise DeficitPlanBlocked("Timeline inputs and measured clock bind different H3 ledgers")
    ledger = _read_object(ledger_path, "H3 review ledger")
    return (
        readiness, clock, timeline_inputs, readiness_path, clock_path, ledger_path,
        _ledger_entries(ledger), event_by_id,
    )


def _source_interval(
    segment: dict[str, Any], source_frames: int,
) -> tuple[int, int, float]:
    try:
        speed = float(segment.get("speed", 1.0))
        source_in = float(segment.get("in", 0.0))
        source_out = float(segment.get("out", source_frames / DELIVERY_FPS))
    except (TypeError, ValueError) as exc:
        raise DeficitPlanBlocked("Accepted H3 segment has invalid timing") from exc
    if not math.isfinite(speed) or not 0.5 <= speed <= 4.0:
        raise DeficitPlanBlocked("Accepted H3 segment has an invalid explicit speed")
    start = max(0, math.ceil(source_in * DELIVERY_FPS - 1e-7))
    end = min(source_frames, math.floor(source_out * DELIVERY_FPS + 1e-7))
    if end <= start:
        raise DeficitPlanBlocked("Accepted H3 segment has no source frames")
    return start, end, speed


def _subtract_ranges(start: int, end: int, used: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    cursor = start
    result: list[tuple[int, int]] = []
    for left, right in sorted(used):
        left = max(start, left)
        right = min(end, right)
        if right <= left:
            continue
        if left > cursor:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end))
    return result


def _verify_exhausted_accepted_windows(
    *, episode: Path, repo: Path, ledger_path: Path,
    ledger: dict[str, dict[str, Any]], blocks: list[dict[str, Any]],
    deficit_rows: list[dict[str, Any]], media: dict[str, dict[str, Any]],
    require_block_reuse: bool = True,
) -> list[dict[str, Any]]:
    deficit_blocks = {str(row["block"]) for row in deficit_rows}
    block_map = {str(row.get("id")): row for row in blocks if isinstance(row, dict)}
    audit: list[dict[str, Any]] = []
    for block_id in sorted(deficit_blocks):
        block = block_map.get(block_id)
        if block is None or block.get("status") != "blocked-coverage-deficit":
            raise DeficitPlanBlocked(f"Missing exact blocked picture allocation for {block_id}")
        expected_deficit = sum(
            int(row["frames"]) for row in deficit_rows if row["block"] == block_id
        )
        block_deficit = round(float(block.get("deficitSeconds", -1)) * DELIVERY_FPS)
        if expected_deficit != block_deficit:
            raise DeficitPlanBlocked(f"Deficit arithmetic changed for {block_id}")
        candidates = block.get("candidates")
        if not isinstance(candidates, list):
            raise DeficitPlanBlocked(f"{block_id} has no selected-window evidence")
        used_by_key: dict[tuple[str, str, str, str, float], list[tuple[int, int]]] = {}
        for candidate in candidates:
            if candidate.get("status") != "approved" or _normal_verdict(candidate.get("verdict")) not in ACCEPTED:
                raise DeficitPlanBlocked(f"{block_id} contains unapproved picture")
            scene_id = str(candidate.get("sceneId") or "")
            path = _resolve(episode, repo, ledger_path, candidate.get("path", ""))
            digest = _require_sha(candidate.get("sourceSha256"), "selected H3 source sha256")
            role = str(candidate.get("role") or "")
            speed = float(candidate.get("speed", 1.0))
            start = round(float(candidate.get("sourceStart", 0.0)) * DELIVERY_FPS)
            end = round(float(candidate.get("sourceOut", 0.0)) * DELIVERY_FPS)
            if end <= start:
                raise DeficitPlanBlocked(f"{block_id} selected H3 window is empty")
            key = (scene_id, str(path), digest, role, speed)
            used_by_key.setdefault(key, []).append((start, end))

        unused: list[dict[str, Any]] = []
        for scene_id in block.get("events", []):
            if not require_block_reuse and scene_id not in {r["sceneId"] for r in deficit_rows if r["block"] == block_id}:
                continue
            entry = ledger.get(str(scene_id))
            if not isinstance(entry, dict) or entry.get("pictureRoute") != "h3":
                raise DeficitPlanBlocked(f"{block_id} event is not routed to accepted H3: {scene_id}")
            for segment in entry.get("segments", []):
                if _normal_verdict(segment.get("verdict")) not in ACCEPTED:
                    continue
                source = _resolve(episode, repo, ledger_path, segment.get("source", ""))
                evidence = media.get(str(source))
                if evidence is None:
                    raise DeficitPlanBlocked(f"Accepted H3 source lacks measured evidence: {source}")
                digest = _require_sha(segment.get("sourceSha256"), "accepted H3 source sha256")
                if evidence.get("sha256") != digest:
                    raise DeficitPlanBlocked(f"Accepted H3 source hash differs from measured timeline: {source}")
                if int(evidence.get("audioStreams", -1)) != 0:
                    raise DeficitPlanBlocked(f"H3 source still contains audio: {source}")
                if abs(float(evidence.get("videoFps") or 0) - DELIVERY_FPS) > 0.01:
                    raise DeficitPlanBlocked(f"H3 source is not 24fps: {source}")
                source_frames = int(evidence.get("videoFrames") or 0)
                if source_frames < 1:
                    raise DeficitPlanBlocked(f"H3 source has no exact frame count: {source}")
                start, end, speed = _source_interval(segment, source_frames)
                key = (str(scene_id), str(source), digest, str(segment.get("role") or "approved-h3"), speed)
                remaining = _subtract_ranges(start, end, used_by_key.get(key, []))
                for left, right in remaining:
                    unused.append(
                        {
                            "sceneId": scene_id,
                            "source": _portable(source, episode),
                            "sourceSha256": digest,
                            "sourceStartFrame": left,
                            "sourceEndFrame": right,
                            "sourceFrames": right - left,
                            "outputFramesAtExplicitSpeed": math.floor((right - left) / speed + 0.5),
                            "speed": speed,
                            "role": segment.get("role") or "approved-h3",
                        }
                    )
        audit.append(
            {
                "block": block_id,
                "deficitFrames": expected_deficit,
                "deficitSeconds": round(expected_deficit / DELIVERY_FPS, 6),
                "acceptedUnusedWindows": unused,
                "acceptedUnusedFrames": sum(row["outputFramesAtExplicitSpeed"] for row in unused),
            }
        )
        if unused:
            raise DeficitPlanBlocked(
                f"{block_id} still has accepted unused H3 windows; rerun/fix the timeline allocation before generation"
            )
    return audit


def _default_extract_frame(source: Path, frame_index: int, destination: Path) -> None:
    try:
        _extract_exact_frame(source, frame_index, destination)
    except _MediaWorkerRequired as exc:
        raise DeficitPlanBlocked(str(exc)) from exc
    except ValueError as exc:
        raise DeficitPlanBlocked(f"Exact anchor extraction produced no frame: {source} n={frame_index}") from exc


def _action_rows(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = doc.get("items", doc)
    if isinstance(raw, dict):
        return {str(key): row for key, row in raw.items() if isinstance(row, dict)}
    if isinstance(raw, list):
        return {
            str(row.get("id")): row
            for row in raw
            if isinstance(row, dict) and row.get("id")
        }
    raise DeficitPlanBlocked("Action-line document must be a mapping or item array")


def _continuation_prompt(scene_id: str, action_row: dict[str, Any]) -> str:
    prompt = str(action_row.get("prompt") or action_row.get("action") or "").strip()
    if not prompt:
        raise DeficitPlanBlocked(f"No authored continuation direction for {scene_id}")
    # Prompt wording and visual style belong to the author. Source-frame hashes
    # and measured continuity bind this request without a prose template.
    return prompt


def _stage_plan(plan: dict[str, Any], plan_path: Path, target: Path) -> tuple[Path, str]:
    source_root = Path(plan["sourceRoot"]).resolve()
    manifest = {key: value for key, value in plan.items() if key != "sourceRoot"}
    stills = target / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    for item in manifest["items"]:
        source = (source_root / item["source"]).resolve()
        if not _is_below(source, source_root) or not source.is_file():
            raise DeficitPlanBlocked(f"Continuation anchor missing/outside episode: {source}")
        digest = _sha256(source)
        destination = stills / source.name
        if destination.exists() and _sha256(destination) != digest:
            raise DeficitPlanBlocked(f"Staged anchor name collision: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        item["still"] = source.name
        item["sourceSha256"] = digest
        item.pop("source", None)
    manifest_path = target / "manifest.json"
    _immutable_json(manifest_path, manifest)
    return manifest_path, _sha256(manifest_path)


def _batch(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    if not 1 <= size <= 32:
        raise DeficitPlanBlocked("Finite H3 batch size must be between 1 and 32")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build(
    episode: Path,
    config_path: Path,
    *,
    wave: int = 1,
    extractor: Callable[[Path, int, Path], None] = _default_extract_frame,
) -> dict[str, Any]:
    episode = episode.resolve()
    repo = episode
    config_path = config_path.resolve()
    config = _read_object(config_path, "continuation builder config")
    if config.get("schemaVersion") != 1:
        raise DeficitPlanBlocked("Unsupported continuation builder config version")
    episode_id = config.get("episode")
    if not isinstance(episode_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", episode_id):
        raise DeficitPlanBlocked("Continuation contract must bind this exact episode")
    if not isinstance(wave, int) or isinstance(wave, bool) or wave < 1:
        raise DeficitPlanBlocked("Wave must be a positive integer")

    (
        readiness, clock, timeline_inputs, readiness_path, clock_path, ledger_path,
        ledger, event_by_id,
    ) = _verify_current_timeline(episode, repo, config_path, config)
    if readiness.get("episode") != episode_id or clock.get("episode") != episode_id:
        raise DeficitPlanBlocked("Measured evidence belongs to another episode")
    deficit_rows = [
        dict(row) for row in readiness.get("missing", [])
        if row.get("type") == "h3-coverage-deficit"
    ]
    if not deficit_rows:
        return {
            "schemaVersion": 1,
            "episode": episode_id,
            "status": "no-measured-h3-deficit",
            "readyForH100": False,
            "reason": "The measured timeline needs no continuation generation.",
            "manifests": [],
        }
    seen = set()
    for row in deficit_rows:
        key = (str(row.get("block")), str(row.get("sceneId")))
        if key in seen:
            raise DeficitPlanBlocked(f"Duplicate exact deficit row: {key}")
        seen.add(key)
        scene_id = key[1]
        if scene_id not in event_by_id or event_by_id[scene_id].get("kind") != "narration":
            raise DeficitPlanBlocked(f"H3 deficit does not belong to measured narration: {scene_id}")
        frames = row.get("frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise DeficitPlanBlocked(f"Invalid exact deficit frame count: {scene_id}")
        if abs(float(row.get("seconds", -1)) - frames / DELIVERY_FPS) > 1e-6:
            raise DeficitPlanBlocked(f"Deficit seconds are not derived from exact frames: {scene_id}")

    blocks = readiness.get("pictureBlocks")
    if not isinstance(blocks, list):
        raise DeficitPlanBlocked(
            "Timeline readiness predates exact picture-block evidence; rerun the current timeline builder"
        )
    media = {str(Path(path).resolve()): row for path, row in readiness["media"].items()}
    reuse_audit = _verify_exhausted_accepted_windows(
        episode=episode, repo=repo, ledger_path=ledger_path, ledger=ledger,
        blocks=blocks, deficit_rows=deficit_rows, media=media,
        require_block_reuse=timeline_inputs.get("allowCausalBorrowing", True) is not False,
    )

    separate = config.get("separateCoverage", {})
    if (not isinstance(separate, dict) or any(not isinstance(v, str) or not v.strip() for v in separate.values())
            or not set(separate).issubset({r["sceneId"] for r in deficit_rows})):
        raise DeficitPlanBlocked("Separate coverage must name current deficits with explicit reasons")
    deferred = [dict(r, reason=separate[r["sceneId"]]) for r in deficit_rows if r["sceneId"] in separate]

    action_path = _resolve(
        episode, repo, config_path,
        config.get("actionLines", "story/action-lines.json"),
    )
    if not _is_below(action_path, repo) or not action_path.is_file():
        raise DeficitPlanBlocked(f"Authored H3 action lines are missing: {action_path}")
    action_sha = _sha256(action_path)
    actions = _action_rows(_read_object(action_path, "episode action lines"))

    qualified = config.get("qualifiedRequest") or {}
    seconds = qualified.get("seconds", 7)
    expected_frames = qualified.get("expectedFrames", 175)
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not 4 <= seconds <= 15 or isinstance(expected_frames, bool)
            or not isinstance(expected_frames, int) or not 1 < expected_frames <= 400
            or abs(expected_frames / DELIVERY_FPS - seconds) >= 0.75):
        raise DeficitPlanBlocked("Continuation request must fit the explicit H3 duration/frame contract")
    usable_frames = expected_frames - 1
    batch_size = int(config.get("batchSize", 32))
    if not 1 <= batch_size <= 32:
        raise DeficitPlanBlocked("batchSize must be 1..32")
    deadline_at = int(config.get("deadlineAt", DEFAULT_STATIC_DEADLINE))
    if deadline_at < DEFAULT_STATIC_DEADLINE:
        raise DeficitPlanBlocked("Static authored deadline must not expire before handoff")

    clock_fingerprint = _require_sha(clock.get("clockFingerprint"), "clock fingerprint")
    run_id = f"wave-{wave:02d}-{clock_fingerprint[:12]}"
    output_root = _resolve(
        episode, repo, config_path,
        config.get("outputDirectory", "work/h3-continuations-v1/generated"),
    )
    if not _is_below(output_root, episode):
        raise DeficitPlanBlocked("Continuation output directory must stay below the episode root")
    run_root = output_root / run_id
    anchors_root = run_root / "anchors"

    jobs: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    for ordinal, row in enumerate(sorted(deficit_rows, key=lambda item: _scene_number(str(item["sceneId"]))), 1):
        scene_id = str(row["sceneId"])
        if scene_id in separate:
            continue
        entry = ledger.get(scene_id)
        if not isinstance(entry, dict) or entry.get("pictureRoute") != "h3":
            raise DeficitPlanBlocked(f"Deficit scene is not H3-routed in current ledger: {scene_id}")
        accepted = [
            segment for segment in entry.get("segments", [])
            if _normal_verdict(segment.get("verdict")) in ACCEPTED
        ]
        if not accepted:
            raise DeficitPlanBlocked(f"No accepted H3 source can anchor continuation: {scene_id}")
        segment = accepted[-1]
        source = _resolve(episode, repo, ledger_path, segment.get("source", ""))
        source_row = media.get(str(source))
        if source_row is None:
            raise DeficitPlanBlocked(f"Anchor source is absent from measured media: {source}")
        if int(source_row.get("audioStreams", -1)) != 0:
            raise DeficitPlanBlocked(f"Anchor H3 source has not had audio stripped: {source}")
        source_frames = int(source_row.get("videoFrames") or 0)
        start_frame, end_frame, speed = _source_interval(segment, source_frames)
        anchor_frame = end_frame - 1
        digest = _require_sha(segment.get("sourceSha256"), "anchor H3 source sha256")
        if digest != source_row.get("sha256"):
            raise DeficitPlanBlocked(f"Anchor source hash differs from measured timeline: {scene_id}")
        anchor_name = (
            f"{scene_id}-deficit-w{wave:02d}-anchor-f{anchor_frame:06d}-{digest[:12]}.png"
        )
        anchor = anchors_root / anchor_name
        if not anchor.exists():
            extractor(source, anchor_frame, anchor)
        if not anchor.is_file() or anchor.stat().st_size == 0:
            raise DeficitPlanBlocked(f"Exact continuation anchor is empty: {anchor}")
        anchor_sha = _sha256(anchor)
        action_row = actions.get(scene_id)
        if action_row is None:
            raise DeficitPlanBlocked(f"No authored action line for deficit scene: {scene_id}")
        prompt = _continuation_prompt(scene_id, action_row)
        deficit_frames = int(row["frames"])
        planned_frames = min(deficit_frames, usable_frames)
        job_id = f"{scene_id}-deficit-w{wave:02d}-c01"
        seed = int(config.get("seedBase", 437000)) + wave * 1000 + _scene_number(scene_id)
        job = {
            "id": job_id,
            "scene": _scene_number(scene_id),
            "sceneId": scene_id,
            "block": row["block"],
            "source": _portable(anchor, episode),
            "sourceSha256": anchor_sha,
            "prompt": prompt,
            "seconds": seconds,
            "expectedFrames": expected_frames,
            "short_edge": int(qualified.get("shortEdge", 768)),
            "aspect": str(qualified.get("aspect", "auto")),
            "steps": int(qualified.get("steps", 30)),
            "seed": seed,
            "exactMeasuredDeficitFrames": deficit_frames,
            "exactMeasuredDeficitSeconds": round(deficit_frames / DELIVERY_FPS, 6),
            "conditioningFrameRepeatsPrecedingEndpoint": True,
            "dropConditioningFrameAtAcceptedIngest": True,
            "usableExpectedFramesAfterAnchorDrop": usable_frames,
            "plannedCoverageFramesThisWave": planned_frames,
            "residualFramesAfterThisWave": deficit_frames - planned_frames,
            "anchorSource": _portable(source, episode),
            "anchorSourceSha256": digest,
            "anchorSourceFrame": anchor_frame,
            "anchorLedgerRole": segment.get("role") or "approved-h3",
            "anchorLedgerExplicitSpeed": speed,
            "actionLinesSha256": action_sha,
            "coverageMode": "exact-last-frame-continuation",
            "editorialSpeed": 1.0,
            "freezeFrameAllowed": False,
            "duplicateFootageAllowed": False,
            "productionAudioAllowed": False,
            "stripAudioAtIngest": True,
            "audioPolicy": "Raw model audio is provenance only; production ingest must strip the complete audio stream.",
            "creativeVerdict": "pending-independent-moving-picture-review",
        }
        jobs.append(job)
        anchors.append(
            {
                "sceneId": scene_id,
                "block": row["block"],
                "source": _portable(source, episode),
                "sourceSha256": digest,
                "sourceFrame": anchor_frame,
                "sourceWindowEndFrameExclusive": end_frame,
                "anchor": _portable(anchor, episode),
                "anchorSha256": anchor_sha,
                "basis": "last displayed frame of the final accepted ledger segment for this deficient scene",
            }
        )

    bindings = {
        "builderConfig": {"path": _portable(config_path, episode), "sha256": _sha256(config_path)},
        "timelineInputs": {
            "path": _portable(_resolve(episode, repo, config_path, config["timelineInputs"]), episode),
            "sha256": _sha256(_resolve(episode, repo, config_path, config["timelineInputs"])),
        },
        "timelineReadiness": {"path": _portable(readiness_path, episode), "sha256": _sha256(readiness_path)},
        "measuredClock": {"path": _portable(clock_path, episode), "sha256": _sha256(clock_path)},
        "clockFingerprint": clock_fingerprint,
        "h3ReviewLedger": {"path": _portable(ledger_path, episode), "sha256": _sha256(ledger_path)},
        "actionLines": {"path": _portable(action_path, episode), "sha256": action_sha},
        "timelineDocuments": readiness["documents"],
        "timelineMediaFingerprint": _canonical_sha(readiness["media"]),
    }
    plan_index = {
        "schemaVersion": 1,
        "kind": "measured-h3-deficit-continuation-wave",
        "episode": episode_id,
        "runId": run_id,
        "wave": wave,
        "deliveryFps": DELIVERY_FPS,
        "status": "finite-manifests-authored-offline",
        "readyForH100": True,
        "builderPerformedComputeMutation": False,
        "builderPerformedNetworkCall": False,
        "forecastInputsAccepted": False,
        "bindings": bindings,
        "exactDeficits": deficit_rows,
        "deferredDeficits": deferred,
        "preservedApprovalGates": [dict(row) for row in readiness["missing"]
                                   if row.get("type") == "score-admission"],
        "exactDeficitFrames": sum(int(row["frames"]) for row in deficit_rows),
        "exactDeficitSeconds": round(sum(int(row["frames"]) for row in deficit_rows) / DELIVERY_FPS, 6),
        "acceptedWindowReuseAudit": reuse_audit,
        "anchors": anchors,
        "policies": {
            "newGeneration": "Only after every accepted window in each deficient narration block is exhausted.",
            "continuity": "Every job begins from the exact last displayed frame of its accepted scene window.",
            "multiWave": "Residual deficits wait for reviewed acceptance, ledger ingest and a newly measured next wave.",
            "forbidden": [
                "forecast substitution", "still padding", "freeze frame", "duplicate footage",
                "generic loop", "implicit slowdown", "H3 production audio", "unreviewed output ingest",
            ],
            "stitch": "After acceptance, drop generated frame zero at ingest because it repeats the conditioning endpoint.",
        },
        "jobs": jobs,
        "manifests": [],
    }
    _immutable_json(run_root / "plan-index.json", plan_index)
    plan_index_sha = _sha256(run_root / "plan-index.json")

    manifest_rows: list[dict[str, Any]] = []
    for batch_number, batch_items in enumerate(_batch(jobs, batch_size), 1):
        batch_root = run_root / f"batch-{batch_number:03d}"
        authored_plan = {
            "version": 1,
            "route": "owned-h100-h3",
            "generationApiAllowed": False,
            "authorization": (
                f"Finite {episode_id} H100/H3 wave; no owner approval is claimed here. It is bound to measured deficits, "
                "uses one retained worker, and still requires post-generation moving-picture review before ledger ingest."
            ),
            "deadlineAt": deadline_at,
            "deadlinePolicy": "finite_client clamps the actual run to at most 7000 seconds from its first launch",
            "sourceRoot": str(episode),
            "episode": episode_id,
            "purpose": "Exact-last-frame H3 continuation wave for measured Hindi-master picture deficits",
            "runId": run_id,
            "batch": batch_number,
            "planIndexSha256": plan_index_sha,
            "clockFingerprint": clock_fingerprint,
            "h3ReviewLedgerSha256": bindings["h3ReviewLedger"]["sha256"],
            "preservedApprovalGates": plan_index["preservedApprovalGates"],
            "productionAudioAllowed": False,
            "implicitRetimeAllowed": False,
            "freezeFrameAllowed": False,
            "duplicateFootageAllowed": False,
            "audioPolicy": "Strip all H3 audio at ingest; raw output is provenance only.",
            "items": batch_items,
        }
        authored_path = batch_root / "authored-plan.json"
        _immutable_json(authored_path, authored_plan)
        manifest_path, manifest_sha = _stage_plan(
            authored_plan, authored_path, batch_root / "inputs"
        )
        manifest_rows.append(
            {
                "batch": batch_number,
                "jobs": len(batch_items),
                "authoredPlan": _portable(authored_path, episode),
                "authoredPlanSha256": _sha256(authored_path),
                "manifest": _portable(manifest_path, episode),
                "manifestSha256": manifest_sha,
                "stagedStills": len(batch_items),
                "maximumTechnicalAttemptsPerJob": 2,
            }
        )

    handoff = {
        "schemaVersion": 1,
        "kind": "inert-h100-h3-lifecycle-handoff",
        "episode": episode_id,
        "runId": run_id,
        "planIndexSha256": plan_index_sha,
        "manifests": manifest_rows,
        "executionState": "not-started-by-builder",
        "requirements": [
            "Configured GPU host access and an explicitly authorized finite compute budget.",
            "Exactly one qualified four-GPU H3 worker owned for this finite queue.",
            "Remote GPU/process/lock preflight is empty before server launch.",
            "Every local staged manifest hash matches this handoff before transfer.",
        ],
        "lifecycle": [
            "Inspect the configured worker and its ownership; start it only when authorized and stopped.",
            "Copy the hash-bound staged batch plus finite_client.py and serve_owned_finite.sh to a run-specific directory below AGENTIC_H3_ROOT.",
            "Launch serve_owned_finite.sh under exclusive locks and verify /health.",
            "Run batches sequentially with --max-jobs equal to each manifest's exact job count; never blind-resubmit an uncertain task.",
            "Copy outputs/state back, hash-verify them locally, then stop the configured worker and verify it is stopped.",
            "Run review_finite.py on the configured GPU worker, inspect moving picture, admit only accepted silent windows with frame zero dropped, then rerun the timeline and this builder for any residual.",
        ],
        "prohibitions": [
            "Do not use coverage-forecast-v1.json.",
            "Do not run two H3 clients or workers concurrently.",
            "Do not use raw H3 audio.",
            "Do not append a freeze, duplicate endpoint, loop, or undeclared retime.",
            "Do not mutate the H3 ledger before technical verification and independent moving-picture review.",
        ],
    }
    _immutable_json(run_root / "lifecycle-handoff.json", handoff)

    final_index = dict(plan_index)
    final_index["planIndexPreManifestSha256"] = plan_index_sha
    final_index["manifests"] = manifest_rows
    # The original index is an immutable pre-execution contract.  Keep the
    # materialized manifest inventory in a separate, equally hashable file.
    _immutable_json(run_root / "manifest-index.json", final_index)
    result = {
        "schemaVersion": 1,
        "episode": episode_id,
        "status": "ready",
        "readyForH100": True,
        "runId": run_id,
        "runRoot": str(run_root),
        "exactDeficitFrames": final_index["exactDeficitFrames"],
        "exactDeficitSeconds": final_index["exactDeficitSeconds"],
        "jobs": len(jobs),
        "batches": len(manifest_rows),
        "residualFramesAfterWave": sum(row["residualFramesAfterThisWave"] for row in jobs) + sum(row["frames"] for row in deferred),
        "deferredDeficits": deferred,
        "manifestIndex": str(run_root / "manifest-index.json"),
        "manifestIndexSha256": _sha256(run_root / "manifest-index.json"),
        "lifecycleHandoff": str(run_root / "lifecycle-handoff.json"),
        "lifecycleHandoffSha256": _sha256(run_root / "lifecycle-handoff.json"),
        "manifests": manifest_rows,
    }
    return result


def _default_config(episode: Path) -> Path:
    return episode / "work/h3-continuations-v1/config.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--wave", type=int, default=1)
    args = parser.parse_args()
    episode = args.episode.resolve()
    config_path = args.config.resolve() if args.config else _default_config(episode)
    status_path = episode / "work/h3-continuations-v1/builder-readiness.json"
    try:
        result = build(episode, config_path, wave=args.wave)
    except DeficitPlanBlocked as exc:
        result = {
            "schemaVersion": 1,
            "episode": episode.name,
            "status": "blocked",
            "readyForH100": False,
            "reason": str(exc),
            "networkCalls": 0,
            "computeMutations": 0,
            "manifests": [],
        }
        _atomic_json(status_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    _atomic_json(status_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
