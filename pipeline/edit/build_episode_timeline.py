#!/usr/bin/env python3
"""Build a measured Hindi-master timeline without generating or rendering media.

The episode owns the story, approved picture windows, and dialogue verdicts.
This shared tool only measures those inputs and solves a deterministic timeline.
It deliberately refuses to manufacture missing media, freeze frames, or silently
retime speech.

Default input: ``<episode>/work/assembly-v1/timeline-inputs.json``

Outputs, only after every required input is present and coverage is sufficient:

* ``edit-plan.json`` -- input for ``assemble_longform.py``
* ``subtitle-hi-timing.json`` -- Hindi cues on the Hindi-master clock
* ``subtitle-en-timing.json`` -- authored English translation on that clock
* ``timeline-audit.json`` -- hashes, measurements, breath/handle decisions

``--dry-run`` writes only ``timeline-readiness.json`` and is safe before paid
narration/dialogue generation has happened.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import is_below as _is_below, sha256_file as _sha256
from dialogue_provenance import canonical_sha as _canonical_sha, validate_dialogue_request
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subtitle_manuscript import apply_manuscript


DELIVERY_FPS = 24
ACCEPTED = {
    "accept",
    "accepted",
    "approve",
    "approved",
    "approved-bounded",
    "use",
}
MICRO_CUT_ROLE_WORDS = {
    "insert",
    "impact",
    "match",
    "release",
    "strike",
    "flash",
    "cutaway",
}


class TimelineIncomplete(RuntimeError):
    """The timeline cannot be truthfully emitted from the available inputs."""


@dataclass(frozen=True)
class MediaProbe:
    duration: float
    video_duration: float | None
    audio_duration: float | None
    video_frames: int | None
    video_fps: float | None
    video_streams: int
    audio_streams: int
    video_width: int | None = None
    video_height: int | None = None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _rate(raw: str | None) -> float | None:
    if not raw or raw in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(raw))
    except (ValueError, ZeroDivisionError):
        return None


def probe_media(path: Path) -> MediaProbe:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    doc = json.loads(raw)
    # An MP4 cover image is not another moving-picture stream. Keep the source
    # bytes unchanged; count only playable motion streams for the contract.
    videos = [row for row in doc.get("streams", []) if row.get("codec_type") == "video"
              and not row.get('disposition', {}).get('attached_pic', 0)]
    audios = [row for row in doc.get("streams", []) if row.get("codec_type") == "audio"]
    fmt_duration = float(doc.get("format", {}).get("duration") or 0.0)

    video_duration = None
    video_frames = None
    video_fps = None
    if videos:
        video = videos[0]
        video_fps = _rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
        raw_frames = video.get("nb_frames")
        if raw_frames not in (None, "N/A"):
            video_frames = int(raw_frames)
        raw_duration = video.get("duration")
        if raw_duration not in (None, "N/A"):
            video_duration = float(raw_duration)
        elif video_frames is not None and video_fps:
            video_duration = video_frames / video_fps

    audio_duration = None
    if audios:
        raw_duration = audios[0].get("duration")
        if raw_duration not in (None, "N/A"):
            audio_duration = float(raw_duration)

    known = [x for x in (fmt_duration, video_duration, audio_duration) if x is not None]
    duration = max(known) if known else 0.0
    if duration <= 0:
        raise ValueError(f"No positive media duration: {path}")
    return MediaProbe(
        duration=duration,
        video_duration=video_duration,
        audio_duration=audio_duration,
        video_frames=video_frames,
        video_fps=video_fps,
        video_streams=len(videos),
        audio_streams=len(audios),
        video_width=videos[0].get("width") if videos else None,
        video_height=videos[0].get("height") if videos else None,
    )


def _resolve_path(
    episode: Path, repo: Path, owner: Path, raw: str | Path
) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    base = owner.parent if str(raw).startswith(("./", "../")) else episode
    return (base / path).resolve()


# Shared with the subtitle exporter and other tools that accept a moved project.
from _common import portable_path as _portable_path


def _scene_number(scene_id: str) -> int:
    match = re.search(r"s(\d+)$", scene_id)
    if not match:
        raise ValueError(f"Scene ID has no terminal shot number: {scene_id}")
    return int(match.group(1))


def _normal_verdict(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value") or value.get("status") or ""
    return str(value or "").strip().lower().replace("_", "-")


def _ledger_entries(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = doc.get("entries", doc.get("shots", doc))
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items() if isinstance(value, dict)}
    if isinstance(raw, list):
        result = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            key = row.get("shotId") or row.get("sceneId") or row.get("id")
            if key:
                result[str(key)] = row
        return result
    raise ValueError("H3 review ledger entries must be a mapping or list")


def _find_dialogue_output(entry: dict[str, Any], override: Any) -> dict[str, Any] | None:
    if isinstance(override, str):
        return {"path": override}
    if isinstance(override, dict):
        result = dict(override)
        if "source" in result and "path" not in result:
            result["path"] = result["source"]
        return result if result.get("path") else None

    for key in ("acceptedOutput", "accepted_output", "output", "outputPath", "output_path"):
        value = entry.get(key)
        if isinstance(value, str):
            return {"path": value, "verdict": entry.get("status")}
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            result = dict(value)
            result.setdefault("verdict", entry.get("status"))
            return result
    result = entry.get("result")
    if isinstance(result, dict):
        for key in ("path", "output", "file"):
            if isinstance(result.get(key), str):
                return {"path": result[key], "verdict": entry.get("status")}
    return None


def _speech_window(entry: dict[str, Any], output: dict[str, Any]) -> tuple[float, float] | None:
    candidate = output.get("speechWindow") or output.get("speech_window")
    if candidate is None:
        candidate = entry.get("speechWindow") or entry.get("speech_window")
    if isinstance(candidate, dict):
        start = candidate.get("start", candidate.get("speechStart"))
        end = candidate.get("end", candidate.get("speechEnd"))
        if start is not None and end is not None:
            return float(start), float(end)
    start = output.get("speechStart", output.get("speech_start"))
    end = output.get("speechEnd", output.get("speech_end"))
    if start is not None and end is not None:
        return float(start), float(end)
    return None


def _dialogue_entries(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = doc.get("entries") or doc.get("items") or doc.get("jobs") or []
    result = {}
    for row in rows:
        key = row.get("scene_id") or row.get("sceneId") or row.get("id")
        if key:
            result[str(key)] = row
    return result


def _narration_entries(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in doc.get("items", []):
        key = row.get("shotId")
        if not key:
            item_id = str(row.get("id", ""))
            match = re.match(r"(.+-s\d+)(?:-hi)?$", item_id)
            key = match.group(1) if match else None
        if key:
            result[str(key)] = row
    return result


def _spoken_dialogue(text: str) -> str:
    """Strip an authored transcript label without altering the spoken words."""
    value = str(text).strip()
    marker = ": “"
    if marker in value and value.endswith("”"):
        return value.split(marker, 1)[1][:-1].strip()
    # Some source-grounded scripts use a Hindi attribution instead of NAME:.
    # Only remove that explicit attribution, never an arbitrary quoted phrase.
    if re.match(r"^[^“”]+ ने कहा,?\s*“[^“”]+”$", value):
        return value.split("“", 1)[1][:-1].strip()
    return value


def _dialogue_subtitle_text(hi: str, en: str, mode: str) -> dict[str, str]:
    """Optional performed-only captions; source scripts/paid text stay frozen."""
    if mode == 'source':
        return {'hi': hi, 'en': en}
    if mode != 'spoken-only':
        raise ValueError('Unknown dialogue subtitle mode')
    return {'hi': _spoken_dialogue(hi), 'en': _spoken_dialogue(en)}


def _production_overrides(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Accept the two existing frozen episode schemas without rewriting them."""
    if "overrides" in doc and "items" in doc:
        raise ValueError("Ambiguous production overrides: both overrides and items")
    raw = doc.get("overrides", doc.get("items", []))
    if isinstance(raw, dict):
        if any(not key or not isinstance(row, dict) or row.get("id", key) != key
               for key, row in raw.items()):
            raise ValueError("Invalid keyed production override")
        return dict(raw)
    if not isinstance(raw, list) or any(not isinstance(row, dict) or not row.get("id") for row in raw):
        raise ValueError("Production overrides must be keyed items or rows with IDs")
    result = {str(row["id"]): row for row in raw}
    if len(result) != len(raw):
        raise ValueError("Duplicate production override ID")
    return result


from _common import state_path_matches as _state_path_matches


def _read_json_evidence(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _require_hex_sha(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} must be an explicit SHA-256 digest")
    return text


def _review_approval(
    *, episode: Path, repo: Path, owner: Path, evidence: dict[str, Any],
    review_type: str, item_id: str, source: Path, source_sha: str,
    sidecar: Path, sidecar_sha: str, required_bindings: list[Path],
) -> dict[str, Any]:
    required = {
        "reviewExport", "reviewExportSha256", "reviewFingerprint"
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise ValueError(f"{item_id}: review evidence is missing {missing}")
    review_path = _resolve_path(episode, repo, owner, evidence["reviewExport"])
    if not _is_below(review_path, episode):
        raise ValueError(f"{item_id}: review export must be ingested below the episode root")
    review_sha = _require_hex_sha(evidence["reviewExportSha256"], "reviewExportSha256")
    if not review_path.is_file() or _sha256(review_path) != review_sha:
        raise ValueError(f"{item_id}: review export is missing or its hash changed")
    review = _read_json_evidence(review_path, f"{review_type} review export")
    fingerprint = _require_hex_sha(evidence["reviewFingerprint"], "reviewFingerprint")
    if (review.get("schemaVersion") != 2 or review.get("reviewType") != review_type
            or review.get("reviewFingerprint") != fingerprint):
        raise ValueError(f"{item_id}: review type/fingerprint does not match admission evidence")
    bindings = review.get("sourceBindings")
    if not isinstance(bindings, list):
        raise ValueError(f"{item_id}: review export has no source bindings")
    for required_path in required_bindings:
        actual_sha = _sha256(required_path)
        found = False
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            bound_path = _resolve_path(episode, repo, review_path, binding.get("path", ""))
            if bound_path == required_path.resolve() and binding.get("sha256") == actual_sha:
                found = True
                break
        if not found:
            raise ValueError(
                f"{item_id}: review export is stale; missing current binding for {required_path}"
            )
    matches = []
    for row in review.get("reviews", []):
        if not isinstance(row, dict) or row.get("itemId") != item_id:
            continue
        reviewed_source = row.get("source") or {}
        reviewed_path = _resolve_path(
            episode, repo, review_path, reviewed_source.get("path", "")
        )
        reviewed_sidecar = _resolve_path(
            episode, repo, review_path, reviewed_source.get("sidecarPath", "")
        )
        if (reviewed_path == source.resolve()
                and reviewed_source.get("sha256") == source_sha
                and reviewed_sidecar == sidecar.resolve()
                and reviewed_source.get("sidecarSha256") == sidecar_sha):
            matches.append(row)
    if len(matches) != 1 or _normal_verdict(matches[0].get("decision")) not in ACCEPTED:
        raise ValueError(f"{item_id}: exactly one source-bound approved review decision is required")
    return {
        "reviewExport": _portable_path(review_path, episode),
        "reviewExportSha256": review_sha,
        "reviewFingerprint": fingerprint,
        "reviewedAt": matches[0].get("reviewedAt"),
    }


def _is_micro_role(role: str, configured: set[str]) -> bool:
    words = set(re.findall(r"[a-z]+", role.lower()))
    return bool(words & configured)


def _allocate_windows(
    candidates: list[dict[str, Any]], target_frames: int, min_frames: int,
    micro_words: set[str], warnings: list[dict[str, Any]], event_id: str,
) -> list[dict[str, Any]]:
    if target_frames <= 0:
        return []
    chosen: list[dict[str, Any]] = []
    remaining = target_frames
    for source in candidates:
        if remaining <= 0:
            break
        take = min(int(source["availableFrames"]), remaining)
        if take <= 0:
            continue
        row = dict(source)
        row["useFrames"] = take
        chosen.append(row)
        remaining -= take
    if remaining:
        raise TimelineIncomplete(
            f"Internal allocation error for {event_id}: {remaining} frames unavailable"
        )

    if len(chosen) > 1:
        last = chosen[-1]
        deliberate = _is_micro_role(str(last.get("role", "")), micro_words)
        # A last-frame continuation is still the same shot, not a new tiny
        # cut. Lengthening it by trimming its parent skips the anchor frame.
        if last["useFrames"] < min_frames and not deliberate and not last.get("continuationOf"):
            need = min_frames - last["useFrames"]
            donor = None
            for index in range(len(chosen)-2, -1, -1):
                previous = chosen[index]
                if chosen[index+1].get("continuationOf"):
                    continue  # This parent's exact endpoint owns the next shot's anchor.
                if previous["useFrames"] - need >= min_frames:
                    donor = previous
                    break
            if donor is not None and last["availableFrames"] >= min_frames:
                donor["useFrames"] -= need
                last["useFrames"] += need
            else:
                warnings.append(
                    {
                        "type": "unavoidable-short-picture-window",
                        "sceneId": event_id,
                        "frames": last["useFrames"],
                        "seconds": round(last["useFrames"] / DELIVERY_FPS, 6),
                        "role": last.get("role"),
                        "reason": "No earlier selected window could donate enough frames",
                    }
                )
    return chosen


def _candidate_output(row: dict[str, Any], episode: Path) -> dict[str, Any]:
    start_frame = int(row["sourceStartFrame"])
    use_frames = int(row["useFrames"])
    speed = float(row.get("speed", 1.0))
    if use_frames == int(row["availableFrames"]):
        source_duration = int(row["availableSourceFrames"]) / DELIVERY_FPS
    else:
        source_duration = min(
            int(row["availableSourceFrames"]) / DELIVERY_FPS,
            use_frames * speed / DELIVERY_FPS,
        )
    result = {
        "shot": _scene_number(str(row["sceneId"])),
        "sceneId": row["sceneId"],
        "role": row["role"],
        "path": _portable_path(row["path"], episode),
        "sourceStart": round(start_frame / DELIVERY_FPS, 6),
        "sourceDuration": round(source_duration, 6),
        "sourceOut": round(start_frame / DELIVERY_FPS + source_duration, 6),
        "duration": round(use_frames / DELIVERY_FPS, 6),
        "frames": use_frames,
        "speed": speed,
        "sourceSha256": row["sha256"],
        "status": "approved",
        "verdict": row["verdict"],
    }
    if row.get("borrowedFor"):
        result["borrowedFor"] = row["borrowedFor"]
    if row.get("microCutJustification"):
        result["microCutJustification"] = row["microCutJustification"]
    if row.get("continuationOf"):
        result["continuationOf"] = row["continuationOf"]
    return result


def _subtitle_doc(language: str, events: list[dict[str, Any]], fingerprint: str) -> dict[str, Any]:
    shots = []
    for event in events:
        text = event["subtitleTextHi" if language == "hi" else "subtitleTextEn"]
        shots.append(
            {
                "id": event["id"],
                "start": event["start"],
                "dur": event["duration"],
                "audioAt": event["subtitleAudioAt"],
                "audioDuration": event["subtitleAudioDuration"],
                "dialogue": event["kind"] == "dialogue",
                "speaker": event.get("speaker"),
                "text": text,
                "sourceLanguage": "hi",
            }
        )
        if event.get('subtitleCuePairs'):
            shots[-1].update(cueParts=[p[language] for p in event['subtitleCuePairs']],
                             cueWeights=event['subtitleCueWeights'])
    return {
        "schemaVersion": 1,
        "kind": "hindi-master-subtitle-timing-input",
        "language": language,
        "masterLanguage": "hi",
        "duration": events[-1]["end"] if events else 0.0,
        "timelineFingerprint": fingerprint,
        "timingPolicy": (
            "Measured complete source-audio windows on the Hindi-master clock; "
            "English is an authored translation, never a second voice/master"
        ),
        "shots": shots,
    }


def build_timeline(
    episode: Path,
    config_path: Path,
    *,
    dry_run: bool = False,
    measure_pending_score: bool = False,
    probe: Callable[[Path], MediaProbe] = probe_media,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    episode = episode.resolve()
    repo = episode
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = _resolve_path(
        episode, repo, config_path, config.get("outputDirectory", "work/assembly-v1")
    )
    if not out_dir.is_relative_to(episode / "work"):
        raise ValueError("Timeline outputDirectory must stay inside project/work")
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    measured: dict[str, dict[str, Any]] = {}
    probe_cache: dict[Path, MediaProbe] = {}

    def measured_probe(path: Path, purpose: str) -> MediaProbe | None:
        if not path.is_file():
            missing.append({"type": "media", "purpose": purpose, "path": str(path)})
            return None
        try:
            if path not in probe_cache:
                probe_cache[path] = probe(path)
            value = probe_cache[path]
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            errors.append(
                {"type": "media-probe", "purpose": purpose, "path": str(path), "error": str(exc)}
            )
            return None
        measured[str(path)] = {
            "sha256": _sha256(path),
            "duration": round(value.duration, 6),
            "videoDuration": None if value.video_duration is None else round(value.video_duration, 6),
            "audioDuration": None if value.audio_duration is None else round(value.audio_duration, 6),
            "videoFrames": value.video_frames,
            "videoFps": value.video_fps,
            "videoStreams": value.video_streams,
            "audioStreams": value.audio_streams,
        }
        return value

    required_docs = {
        "storyboard": config.get("storyboard", "story/storyboard_plan.json"),
        "narrationManifest": config.get(
            "narrationManifest", "work/narration-hi-v1/manifest.json"
        ),
        "dialogueManifest": config.get(
            "dialogueManifest", "work/grok-dialogue-v1/draft-manifest.json"
        ),
        "h3ReviewLedger": config.get("h3ReviewLedger", "work/h3-review-ledger-v1.json"),
    }
    if config.get("productionOverrides"):
        required_docs["productionOverrides"] = config["productionOverrides"]
    if config.get('subtitleManuscript'):
        required_docs['subtitleManuscript'] = config['subtitleManuscript']
    docs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for label, raw in required_docs.items():
        path = _resolve_path(episode, repo, config_path, raw)
        if not path.is_file():
            missing.append({"type": "document", "purpose": label, "path": str(path)})
            # The review director may still be writing the H3 ledger while an
            # offline dry-run inventories the independent speech lane.
            if dry_run and label == "h3ReviewLedger":
                docs[label] = (path, {})
            continue
        try:
            docs[label] = (path, json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"type": "document", "purpose": label, "path": str(path), "error": str(exc)})

    readiness_path = out_dir / "timeline-readiness.json"
    if len(docs) != len(required_docs):
        readiness = {
            "episode": episode.name,
            "ready": False,
            "dryRun": dry_run,
            "missing": missing,
            "errors": errors,
            "warnings": warnings,
            "outputsWritten": [str(readiness_path)],
        }
        _atomic_json(readiness_path, readiness)
        if dry_run:
            return readiness, None
        raise TimelineIncomplete("Required timeline documents are missing or invalid")

    storyboard_path, storyboard = docs["storyboard"]
    narration_path, narration_doc = docs["narrationManifest"]
    dialogue_path, dialogue_doc = docs["dialogueManifest"]
    ledger_path, ledger_doc = docs["h3ReviewLedger"]
    ledger_available = ledger_path.is_file()
    try:
        overrides = _production_overrides(docs.get("productionOverrides", (config_path, {}))[1])
    except ValueError as exc:
        errors.append({"type": "production-overrides", "error": str(exc)})
        overrides = {}
    rows = storyboard.get("stills", [])
    if not rows:
        errors.append({"type": "storyboard", "error": "No storyboard stills"})
    scene_ids = [str(row.get("still_id", "")) for row in rows]
    if len(scene_ids) != len(set(scene_ids)) or any(not item for item in scene_ids):
        errors.append({"type": "storyboard", "error": "Scene IDs are empty or duplicated"})

    narration = _narration_entries(narration_doc)
    dialogues = _dialogue_entries(dialogue_doc)
    ledger = _ledger_entries(ledger_doc)
    dialogue_outputs = config.get("dialogueOutputs", {})
    if not isinstance(dialogue_outputs, dict):
        errors.append({"type": "config", "error": "dialogueOutputs must be a mapping"})
        dialogue_outputs = {}
    score_placements_raw = config.get("scorePlacements", [])
    if not isinstance(score_placements_raw, list):
        errors.append({"type": "config", "error": "scorePlacements must be an array"})
        score_placements_raw = []
    if config.get('requireApprovedScore') and not score_placements_raw:
        missing.append({'type':'score-admission', 'reason':'This episode requires a reviewed score; no score placements are admitted.'})

    production_path = docs.get("productionOverrides", (None, {}))[0]
    production_sha = _sha256(production_path) if production_path and production_path.is_file() else None
    for label, manifest_doc in (
        ("narrationManifest", narration_doc), ("dialogueManifest", dialogue_doc)
    ):
        declared = manifest_doc.get("productionOverridesSha256")
        if production_sha and declared is not None and declared != production_sha:
            errors.append({
                "type": "manifest-production-override-hash",
                "purpose": label,
                "declared": declared,
                "actual": production_sha,
            })

    narration_state_path = _resolve_path(
        episode,
        repo,
        config_path,
        config.get(
            "narrationState",
            str(narration_path.with_suffix(".eleven-state.json")),
        ),
    )
    narration_state: dict[str, Any] | None = None
    if narration_state_path.is_file():
        try:
            narration_state = _read_json_evidence(narration_state_path, "narration finite state")
            if narration_state.get("manifestSha256") != _sha256(narration_path):
                raise ValueError("narration finite state does not bind the current manifest")
        except ValueError as exc:
            errors.append({"type": "narration-provenance", "error": str(exc)})
            narration_state = None

    effective_text: dict[str, dict[str, str]] = {}
    for row in rows:
        scene_id = str(row["still_id"])
        override = overrides.get(scene_id, {})
        if row.get("speaker"):
            effective_hi = str(override.get("narration_hi") or row["narration_hi"])
            source_scene_id = dialogue_outputs.get(scene_id, {}).get("sourceSceneId", scene_id) if isinstance(dialogue_outputs.get(scene_id), dict) else scene_id
            dialogue_row = dialogues.get(source_scene_id, {})
            submitted_hi = dialogue_row.get("hindi") or dialogue_row.get("exact_hindi_line")
            if (override.get("narration_hi") and submitted_hi
                    and _spoken_dialogue(str(submitted_hi)) != _spoken_dialogue(effective_hi)):
                errors.append(
                    {
                        "type": "production-override-drift",
                        "sceneId": scene_id,
                        "field": "dialogue_hi",
                        "override": _spoken_dialogue(effective_hi),
                        "dialogueManifest": _spoken_dialogue(str(submitted_hi)),
                    }
                )
            effective_text[scene_id] = _dialogue_subtitle_text(
                effective_hi, str(override.get("narration_en") or row["narration_en"]),
                config.get('dialogueSubtitleMode', 'source'))
            continue
        narration_row = narration.get(scene_id, {})
        submitted_hi = str(narration_row.get("text") or row["narration_hi"])
        if override.get("narration_hi") and override["narration_hi"] != submitted_hi:
            errors.append(
                {
                    "type": "production-override-drift",
                    "sceneId": scene_id,
                    "field": "narration_hi",
                    "override": override["narration_hi"],
                    "narrationManifest": submitted_hi,
                }
            )
        effective_text[scene_id] = {
            "hi": submitted_hi,
            "en": str(override.get("narration_en") or row["narration_en"]),
        }

    if 'subtitleManuscript' in docs:
        try:
            effective_text = apply_manuscript(episode, docs['subtitleManuscript'][0],
                config.get('subtitleManuscriptSha256'), effective_text)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({'type':'subtitle-manuscript', 'error':str(exc)})

    h3_by_scene: dict[str, list[dict[str, Any]]] = {}
    routes: dict[str, str] = {}
    for row in rows:
        scene_id = str(row["still_id"])
        if not ledger_available:
            continue
        entry = ledger.get(scene_id)
        if entry is None:
            missing.append({"type": "h3-verdict", "sceneId": scene_id})
            continue
        route = str(entry.get("pictureRoute", "")).strip().lower()
        routes[scene_id] = route
        expected_route = "grok-dialogue" if row.get("speaker") else "h3"
        if route not in {"h3", "grok-dialogue", "omit"}:
            errors.append({"type": "h3-route", "sceneId": scene_id, "route": route})
            continue
        if row.get("speaker") and route != "grok-dialogue":
            errors.append(
                {
                    "type": "route-mismatch",
                    "sceneId": scene_id,
                    "expected": expected_route,
                    "actual": route,
                }
            )
        if not row.get("speaker") and route == "grok-dialogue":
            errors.append(
                {
                    "type": "route-mismatch",
                    "sceneId": scene_id,
                    "expected": "h3-or-omit",
                    "actual": route,
                }
            )
        candidates = []
        for index, segment in enumerate(entry.get("segments", [])):
            verdict = _normal_verdict(segment.get("verdict"))
            if verdict not in ACCEPTED:
                continue
            raw_source = segment.get("source")
            if not raw_source:
                errors.append(
                    {"type": "h3-segment", "sceneId": scene_id, "index": index, "error": "missing source"}
                )
                continue
            source = _resolve_path(episode, repo, ledger_path, raw_source)
            info = measured_probe(source, f"H3 {scene_id} segment {index}")
            if info is None:
                continue
            declared_sha = segment.get("sourceSha256")
            actual_sha = measured[str(source)]["sha256"]
            if declared_sha and str(declared_sha).lower() != actual_sha:
                errors.append(
                    {
                        "type": "h3-source-hash-mismatch",
                        "sceneId": scene_id,
                        "source": str(source),
                        "declared": declared_sha,
                        "actual": actual_sha,
                    }
                )
                continue
            if info.video_streams != 1 or info.audio_streams != 0:
                errors.append(
                    {
                        "type": "h3-stream-contract",
                        "sceneId": scene_id,
                        "source": str(source),
                        "videoStreams": info.video_streams,
                        "audioStreams": info.audio_streams,
                    }
                )
                continue
            if info.video_fps is None or abs(info.video_fps - DELIVERY_FPS) > 0.01:
                errors.append(
                    {
                        "type": "h3-fps-contract",
                        "sceneId": scene_id,
                        "source": str(source),
                        "fps": info.video_fps,
                    }
                )
                continue
            natural_frames = info.video_frames or math.floor((info.video_duration or info.duration) * DELIVERY_FPS + 1e-6)
            source_in = float(segment.get("in", 0.0))
            source_out = float(segment.get("out", natural_frames / DELIVERY_FPS))
            try:
                speed = float(segment.get("speed", 1.0))
            except (TypeError, ValueError):
                speed = float("nan")
            if not math.isfinite(speed) or not 0.5 <= speed <= 4.0:
                errors.append(
                    {
                        "type": "h3-speed-contract",
                        "sceneId": scene_id,
                        "source": str(source),
                        "speed": segment.get("speed", 1.0),
                        "allowed": [0.5, 4.0],
                        "reason": "H3 retiming must be explicit, positive, and bounded",
                    }
                )
                continue
            start_frame = max(0, math.ceil(source_in * DELIVERY_FPS - 1e-7))
            end_frame = min(natural_frames, math.floor(source_out * DELIVERY_FPS + 1e-7))
            if end_frame <= start_frame:
                errors.append(
                    {
                        "type": "h3-window",
                        "sceneId": scene_id,
                        "source": str(source),
                        "in": source_in,
                        "out": source_out,
                    }
                )
                continue
            role = str(segment.get("role") or "approved-h3")
            available_source_frames = end_frame - start_frame
            available_frames = math.floor(available_source_frames / speed + 0.5)
            if available_frames < 1:
                errors.append(
                    {
                        "type": "h3-speed-window-too-short",
                        "sceneId": scene_id,
                        "source": str(source),
                        "sourceFrames": available_source_frames,
                        "speed": speed,
                    }
                )
                continue
            candidate = {
                "sceneId": scene_id,
                "path": source,
                "sha256": actual_sha,
                "sourceStartFrame": start_frame,
                "sourceEndFrame": end_frame,
                "availableSourceFrames": available_source_frames,
                "availableFrames": available_frames,
                "speed": speed,
                "role": role,
                "verdict": verdict,
            }
            if segment.get("continuationOf") is not None:
                try:
                    link = segment['continuationOf']
                    previous_id = scene_ids[scene_ids.index(scene_id)-1] if scene_ids.index(scene_id) > 0 else None
                    preceding = candidates or h3_by_scene.get(previous_id, [])
                    if not isinstance(link, dict):
                        raise ValueError('continuation needs its immediate accepted predecessor')
                    # One continuous shot can span two consecutive narration
                    # utterances. The renderer rechecks the actual selected
                    # endpoint after allocation; skipped/dialogue rows cannot join.
                    if preceding:
                        parent = preceding[-1]
                    elif link.get('parentKind') == 'dialogue' and previous_id in dialogue_outputs:
                        # Opt-in native Grok endpoint -> silent H3 continuation.
                        # Paid-source admission and selected endpoint are checked
                        # again after the complete dialogue event is resolved.
                        choice = dialogue_outputs[previous_id]
                        parent_path = _resolve_path(episode, repo, config_path, choice.get('path', ''))
                        parent_info = measured_probe(parent_path, 'dialogue continuation parent')
                        if (parent_info is None or parent_info.video_fps != DELIVERY_FPS
                                or not parent_info.video_frames):
                            raise ValueError('dialogue continuation requires measured native frames')
                        parent = dict(path=parent_path, sha256=measured[str(parent_path)]['sha256'],
                                      sourceEndFrame=parent_info.video_frames, speed=1.0)
                    else:
                        raise ValueError('continuation needs its immediate accepted predecessor')
                    linked_source = _resolve_path(episode, repo, ledger_path, link.get('source', ''))
                    endpoint = link.get('endFrameExclusive')
                    if isinstance(endpoint, bool) or not isinstance(endpoint, int):
                        raise ValueError('continuation endpoint must be an exact integer source frame')
                    if (linked_source != parent['path']
                            or link.get('sourceSha256') != parent['sha256']
                            or endpoint != parent['sourceEndFrame']):
                        raise ValueError('continuation source/hash/endpoint differs from its predecessor')
                    if start_frame != 1:
                        raise ValueError('true continuation must drop only conditioning frame0')
                    if speed != 1.0 or parent['speed'] != 1.0:
                        raise ValueError('retimed continuation joins need a separate explicit moving-picture review')
                    candidate['continuationOf'] = dict(source=_portable_path(linked_source,episode),
                        sourceSha256=parent['sha256'],endFrameExclusive=endpoint)
                    if link.get('parentKind') == 'dialogue':
                        candidate['continuationOf']['parentKind'] = 'dialogue'
                except (ValueError, TypeError) as exc:
                    errors.append(dict(type='h3-continuation-contract',sceneId=scene_id,
                                       source=str(source),error=str(exc)))
                    continue
            if available_frames < round(float(config.get("minimumPictureSegmentSeconds", 1.3)) * DELIVERY_FPS):
                if _is_micro_role(role, set(config.get("microCutRoleWords", MICRO_CUT_ROLE_WORDS))):
                    candidate["microCutJustification"] = role
                else:
                    warnings.append(
                        {
                            "type": "short-approved-h3-window",
                            "sceneId": scene_id,
                            "role": role,
                            "seconds": round(available_frames / DELIVERY_FPS, 6),
                        }
                    )
            candidates.append(candidate)
        h3_by_scene[scene_id] = candidates
        if route == "h3" and not candidates and dry_run:
            missing.append({"type": "approved-h3-coverage", "sceneId": scene_id})

    # Resolve and measure the complete speech inventory before writing timings.
    narration_media: dict[str, tuple[Path, MediaProbe]] = {}
    dialogue_media: dict[str, tuple[Path, MediaProbe, dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        scene_id = str(row["still_id"])
        if row.get("speaker"):
            selection = dialogue_outputs.get(scene_id)
            source_scene_id = selection.get("sourceSceneId", scene_id) if isinstance(selection, dict) else scene_id
            entry = dialogues.get(source_scene_id)
            if entry is None:
                missing.append({"type": "dialogue-manifest-entry", "sceneId": scene_id})
                continue
            if source_scene_id != scene_id:
                submitted = entry.get("hindi") or entry.get("exact_hindi_line") or ""
                if (entry.get("speaker") != row.get("speaker")
                        or _spoken_dialogue(submitted) != _spoken_dialogue(effective_text[scene_id]["hi"])
                        or not selection.get("reuseReason")):
                    errors.append({"type": "dialogue-reuse-contract", "sceneId": scene_id,
                                   "error": "Reused native performance requires the same speaker, exact current spoken text, and an authored reason"})
                    continue
            output = _find_dialogue_output(entry, selection)
            if output is None:
                missing.append({"type": "approved-dialogue-output", "sceneId": scene_id})
                continue
            verdict = _normal_verdict(output.get("verdict", entry.get("status")))
            if config.get("requireApprovedDialogue", True) and verdict not in ACCEPTED:
                missing.append(
                    {
                        "type": "approved-dialogue-verdict",
                        "sceneId": scene_id,
                        "actual": verdict or "unset",
                    }
                )
                continue
            source = _resolve_path(episode, repo, dialogue_path, output["path"])
            if not _is_below(source, episode):
                errors.append({
                    "type": "dialogue-path-containment", "sceneId": scene_id,
                    "source": str(source),
                })
                continue
            info = measured_probe(source, f"Grok dialogue {scene_id}")
            if info is None:
                continue
            if info.video_streams != 1 or info.audio_streams < 1:
                errors.append(
                    {
                        "type": "dialogue-stream-contract",
                        "sceneId": scene_id,
                        "source": str(source),
                        "videoStreams": info.video_streams,
                        "audioStreams": info.audio_streams,
                    }
                )
                continue
            try:
                actual_sha = measured[str(source)]["sha256"]
                declared_sha = _require_hex_sha(output.get("sha256"), "dialogue sha256")
                if declared_sha != actual_sha:
                    raise ValueError(f"{scene_id}: dialogue media hash changed")
                sidecar = _resolve_path(
                    episode, repo, dialogue_path, output.get("sidecar", "")
                )
                if not _is_below(sidecar, episode):
                    raise ValueError(f"{scene_id}: dialogue sidecar escapes episode root")
                sidecar_sha = _require_hex_sha(
                    output.get("sidecarSha256"), "dialogue sidecarSha256"
                )
                if not sidecar.is_file() or _sha256(sidecar) != sidecar_sha:
                    raise ValueError(f"{scene_id}: dialogue sidecar is missing or changed")
                source_doc = _read_json_evidence(sidecar, "dialogue source sidecar")
                execution_path, execution_doc, execution_entry = dialogue_path, dialogue_doc, entry
                if output.get('executionManifest'):
                    execution_path = _resolve_path(episode, repo, config_path, output['executionManifest'])
                    if not _is_below(execution_path, episode):
                        raise ValueError('correction execution manifest escapes episode')
                    if _sha256(execution_path) != output.get('executionManifestSha256'):
                        raise ValueError('correction execution manifest changed')
                    execution_doc = _read_json_evidence(execution_path, 'correction execution manifest')
                    execution_entry = _dialogue_entries(execution_doc).get(source_scene_id, {})
                    if execution_entry.get('hindi') != entry.get('hindi'):
                        raise ValueError('correction changes current spoken dialogue')
                    for key in ('episode', 'model', 'resolution', 'aspectRatio', 'language', 'productionOverridesSha256'):
                        if execution_doc.get(key) != dialogue_doc.get(key):
                            raise ValueError('correction changes dialogue contract: ' + key)
                    docs['dialogueCorrection:'+scene_id] = (execution_path, execution_doc)
                for field in ('source', 'prompt'):
                    input_path = _resolve_path(episode, repo, execution_path, execution_entry[field])
                    if not _is_below(input_path, episode) or not input_path.is_file():
                        raise ValueError('dialogue input missing or escapes episode: ' + field)
                    if _sha256(input_path) != execution_entry[field+'Sha256']:
                        raise ValueError('dialogue input bytes changed: ' + field)
                prompt_path = _resolve_path(episode, repo, execution_path, execution_entry['prompt'])
                execution_request = {
                    "version": 1,
                    "jobId": source_scene_id,
                    "executionManifestSha256": _sha256(execution_path),
                    "model": execution_doc.get("model"),
                    "sourceSha256": execution_entry.get("sourceSha256"),
                    "promptSha256": execution_entry.get("promptSha256"),
                    "durationSeconds": execution_entry.get("durationSeconds"),
                    "resolution": execution_doc.get("resolution"),
                    "aspectRatio": execution_doc.get("aspectRatio"),
                    "audio": True,
                    "outputStem": execution_entry.get("outputStem"),
                }
                references = [ref.get("sha256") for ref in execution_entry.get("references") or []]
                if references:  # mirrors kie_dialogue_finite.execution_request
                    execution_request.update(version=2, referenceSha256s=references)
                expected_sidecar = {
                    "jobId": source_scene_id,
                    "executionManifestSha256": _sha256(execution_path),
                    "executionRequestSha256": _canonical_sha(execution_request),
                    "sourceSha256": execution_entry.get("sourceSha256"),
                    "promptSha256": execution_entry.get("promptSha256"),
                    "outputSha256": actual_sha,
                    "sha256": actual_sha,
                    "model": execution_doc.get("model"),
                    "durationSeconds": execution_entry.get("durationSeconds"),
                    "resolution": execution_doc.get("resolution"),
                    "aspectRatio": execution_doc.get("aspectRatio"),
                    "audio": True,
                }
                for field, expected in expected_sidecar.items():
                    if not expected or source_doc.get(field) != expected:
                        raise ValueError(f"{scene_id}: dialogue sidecar mismatch: {field}")
                request = source_doc.get("request")
                request_sha = _require_hex_sha(
                    source_doc.get("requestSha256"), "dialogue requestSha256"
                )
                if not isinstance(request, dict) or _canonical_sha(request) != request_sha:
                    raise ValueError(f"{scene_id}: dialogue request fingerprint is invalid")
                if source_doc.get("requestFingerprint") != request_sha:
                    raise ValueError(f"{scene_id}: dialogue request digest aliases disagree")
                validate_dialogue_request(source_doc, {**expected_sidecar, "referenceSha256s": references},
                                          prompt_path.read_text(encoding='utf-8'))
                if not isinstance(source_doc.get("taskId"), str) or not source_doc["taskId"]:
                    raise ValueError(f"{scene_id}: dialogue sidecar has no paid taskId")
                required_bindings = [dialogue_path]
                if execution_path != dialogue_path:
                    required_bindings.append(execution_path)
                if production_path:
                    required_bindings.append(production_path)
                review = _review_approval(
                    episode=episode, repo=repo, owner=config_path, evidence=output,
                    review_type="dialogue", item_id=source_scene_id, source=source,
                    source_sha=actual_sha, sidecar=sidecar, sidecar_sha=sidecar_sha,
                    required_bindings=required_bindings,
                )
                admitted = dict(output)
                if output.get("pictureCropFrom") is not None and output.get("pictureCrop") is None:
                    raise ValueError("A timed dialogue crop requires an authored crop")
                if output.get("pictureCrop") is not None:
                    crop = output["pictureCrop"]
                    if (not isinstance(crop, dict) or set(crop) != {"x", "y", "width", "height"}
                            or any(isinstance(v, bool) or not isinstance(v, int) or v % 2 for v in crop.values())
                            or crop["x"] < 0 or crop["y"] < 0 or crop["width"] < 128 or crop["height"] < 72
                            or not info.video_width or not info.video_height
                            or crop["x"] + crop["width"] > info.video_width
                            or crop["y"] + crop["height"] > info.video_height
                            or not output.get("pictureCropReason")):
                        raise ValueError("Invalid authored dialogue crop or missing reason")
                    if output.get("pictureCropFrom") is not None:
                        crop_from = output["pictureCropFrom"]
                        if (isinstance(crop_from, bool) or not isinstance(crop_from, (int, float))
                                or not math.isfinite(crop_from) or not 0 < crop_from < info.duration
                                or abs(crop_from * DELIVERY_FPS - round(crop_from * DELIVERY_FPS)) > 1e-6):
                            raise ValueError("Timed dialogue crop must start at an interior native frame boundary")
                admitted.update({
                    "sourceSceneId": source_scene_id,
                    "sha256": actual_sha,
                    "sidecar": _portable_path(sidecar, episode),
                    "sidecarSha256": sidecar_sha,
                    "executionManifestSha256": _sha256(execution_path),
                    **review,
                })
            except ValueError as exc:
                errors.append({
                    "type": "dialogue-admission", "sceneId": scene_id,
                    "error": str(exc),
                })
                continue
            dialogue_media[scene_id] = (source, info, execution_entry, admitted)
        else:
            entry = narration.get(scene_id)
            if entry is None:
                missing.append({"type": "narration-manifest-entry", "sceneId": scene_id})
                continue
            source = _resolve_path(episode, repo, narration_path, entry["output"])
            if not _is_below(source, episode):
                errors.append({
                    "type": "narration-path-containment", "sceneId": scene_id,
                    "source": str(source),
                })
                continue
            info = measured_probe(source, f"Hindi narration {scene_id}")
            if info is None:
                continue
            if info.audio_streams < 1:
                errors.append(
                    {"type": "narration-stream-contract", "sceneId": scene_id, "source": str(source)}
                )
                continue
            if narration_state is None:
                errors.append({
                    "type": "narration-provenance", "sceneId": scene_id,
                    "error": "generated narration exists without its canonical finite state",
                })
                continue
            record = (narration_state.get("items") or {}).get(str(entry.get("id")), {})
            actual_sha = measured[str(source)]["sha256"]
            if (record.get("status") != "COMPLETED"
                    or record.get("sha256") != actual_sha
                    or record.get("outputSha256") != actual_sha
                    or not _state_path_matches(record.get("output"), source, episode,
                                               config.get("narrationStateEpisodeRoot"))):
                errors.append({
                    "type": "narration-admission", "sceneId": scene_id,
                    "error": "finite state is not COMPLETED and hash/path bound to this output",
                })
                continue
            narration_media[scene_id] = (source, info)

    validated_score_placements: list[dict[str, Any]] = []
    score_documents: dict[str, dict[str, Any]] = {}
    if score_placements_raw:
        score_manifest_path = _resolve_path(
            episode, repo, config_path,
            config.get("scoreManifest", "work/music-score-v1/execution-manifest.json"),
        )
        score_state_path = score_manifest_path.with_suffix(".state.json")
        try:
            if not _is_below(score_manifest_path, episode):
                raise ValueError("score manifest escapes episode root")
            score_doc = _read_json_evidence(score_manifest_path, "score execution manifest")
            score_state = _read_json_evidence(score_state_path, "score finite state")
            score_manifest_sha = _sha256(score_manifest_path)
            if score_state.get("executionManifestSha256") != score_manifest_sha:
                raise ValueError("score finite state does not bind the current execution manifest")
            # Creation receipts remain immutable when instrumental music is
            # adopted into an edit. The source-bound review below must bind the
            # current script; creation metadata must agree with its own state.
            score_creation_production_sha = score_doc.get("productionOverridesSha256")
            if score_creation_production_sha is not None:
                _require_hex_sha(score_creation_production_sha, "score creation production hash")
            if score_state.get("productionOverridesSha256") != score_creation_production_sha:
                raise ValueError("score state changed its creation production binding")
            score_documents = {
                "scoreManifest": {"path": str(score_manifest_path), "sha256": score_manifest_sha},
                "scoreState": {"path": str(score_state_path), "sha256": _sha256(score_state_path)},
            }
            cues = {str(row.get("id")): row for row in score_doc.get("cues", [])}
            seen_score_placements: set[str] = set()
            for pos, placement in enumerate(score_placements_raw):
                if not isinstance(placement, dict):
                    raise ValueError(f"score placement {pos} must be an object")
                movement = str(placement.get("movement") or "")
                placement_id = placement.get("id", movement)
                if (not movement or movement not in cues
                        or not isinstance(placement_id, str)
                        or not re.fullmatch(r"[A-Za-z0-9_.-]+", placement_id)
                        or placement_id in seen_score_placements):
                    raise ValueError(f"score placement {pos} has duplicate/invalid identity or unknown movement {movement}")
                seen_score_placements.add(placement_id)
                take = placement.get("take")
                if not isinstance(take, int) or isinstance(take, bool) or take <= 0:
                    raise ValueError(f"{movement}: take must be a positive integer")
                start = placement.get("timelineStart")
                duration = placement.get("timelineDuration")
                if (isinstance(start, bool) or not isinstance(start, (int, float))
                        or float(start) < 0 or isinstance(duration, bool)
                        or not isinstance(duration, (int, float)) or float(duration) <= 0):
                    raise ValueError(f"{movement}: timelineStart/duration are invalid")
                track = _resolve_path(episode, repo, config_path, placement.get("track", ""))
                if not _is_below(track, episode):
                    raise ValueError(f"{movement}: score track escapes episode root")
                info = measured_probe(track, f"Score {movement} take {take}")
                if info is None:
                    continue
                if info.audio_streams < 1 or info.video_streams:
                    raise ValueError(f"{movement}: score track must contain audio only")
                track_sha = _require_hex_sha(placement.get("trackSha256"), "trackSha256")
                if track_sha != measured[str(track)]["sha256"]:
                    raise ValueError(f"{movement}: score track hash changed")
                sidecar = _resolve_path(
                    episode, repo, config_path, placement.get("sidecar", "")
                )
                if not _is_below(sidecar, episode):
                    raise ValueError(f"{movement}: score sidecar escapes episode root")
                sidecar_sha = _require_hex_sha(
                    placement.get("sidecarSha256"), "score sidecarSha256"
                )
                if not sidecar.is_file() or _sha256(sidecar) != sidecar_sha:
                    raise ValueError(f"{movement}: score sidecar is missing or changed")
                source_doc = _read_json_evidence(sidecar, "score source sidecar")
                cue = cues[movement]
                expected_source = {
                    "cueId": movement,
                    "candidateIndex": take,
                    "executionManifestSha256": score_manifest_sha,
                    "sourceDraftSha256": score_doc.get("sourceDraftSha256"),
                    "requestSha256": _canonical_sha(cue.get("payload")),
                    "sha256": track_sha,
                }
                if source_doc.get("productionOverridesSha256") != score_creation_production_sha:
                    raise ValueError(f"{movement}: score sidecar changed its creation production binding")
                for field, expected in expected_source.items():
                    if expected is None or source_doc.get(field) != expected:
                        raise ValueError(f"{movement}: score sidecar mismatch: {field}")
                state_item = (score_state.get("items") or {}).get(movement, {})
                state_matches = [
                    row for row in state_item.get("candidates", [])
                    if (row.get("index") == take and row.get("status") == "COMPLETE"
                        and _state_path_matches(row.get("file"), track, episode,
                                                config.get("scoreStateEpisodeRoot"))
                        and row.get("sha256") == track_sha
                        and _state_path_matches(row.get("sidecar"), sidecar, episode,
                                                config.get("scoreStateEpisodeRoot"))
                        and row.get("sidecarSha256") == sidecar_sha)
                ]
                if state_item.get("status") != "DOWNLOADED" or len(state_matches) != 1:
                    raise ValueError(f"{movement}: score finite state does not admit this candidate")
                required_bindings = [score_manifest_path, narration_path, dialogue_path]
                if production_path:
                    required_bindings.append(production_path)
                review = _review_approval(
                    episode=episode, repo=repo, owner=config_path, evidence=placement,
                    review_type="score", item_id=movement, source=track,
                    source_sha=track_sha, sidecar=sidecar, sidecar_sha=sidecar_sha,
                    required_bindings=required_bindings,
                )
                normalized = dict(placement)
                normalized.update({
                    "track": _portable_path(track, episode),
                    "trackSha256": track_sha,
                    "sidecar": _portable_path(sidecar, episode),
                    "sidecarSha256": sidecar_sha,
                    "executionManifest": _portable_path(score_manifest_path, episode),
                    "executionManifestSha256": score_manifest_sha,
                    **review,
                })
                validated_score_placements.append(normalized)
        except ValueError as exc:
            errors.append({"type": "score-admission", "error": str(exc)})

    readiness = {
        "schemaVersion": 1,
        "episode": storyboard.get("episode", episode.name),
        "ready": not missing and not errors,
        "dryRun": dry_run,
        "storyboardScenes": len(rows),
        "narrationScenes": sum(not bool(row.get("speaker")) for row in rows),
        "dialogueScenes": sum(bool(row.get("speaker")) for row in rows),
        "measuredNarration": len(narration_media),
        "measuredDialogue": len(dialogue_media),
        "approvedH3Scenes": sum(bool(value) for value in h3_by_scene.values()),
        "missing": missing,
        "errors": errors,
        "warnings": warnings,
        "documents": {
            label: {
                "path": str(path),
                "sha256": _sha256(path) if path.is_file() else None,
            }
            for label, (path, _) in docs.items()
        },
        "media": measured,
        "outputsWritten": [str(readiness_path)],
    }
    if narration_state_path.is_file():
        readiness["documents"]["narrationState"] = {
            "path": str(narration_state_path), "sha256": _sha256(narration_state_path)
        }
    readiness["documents"].update(score_documents)
    _atomic_json(readiness_path, readiness)
    if dry_run:
        return readiness, None
    blocking_missing = [row for row in missing if not (
        measure_pending_score and row.get('type') == 'score-admission')]
    if blocking_missing or errors:
        raise TimelineIncomplete(
            f"Timeline inputs incomplete: {len(missing)} missing, {len(errors)} invalid; see {readiness_path}"
        )

    breathing = config.get("narrationBreathing", {})
    same_beat_target = float(breathing.get("sameBeatSeconds", 0.40))
    beat_end_target = float(breathing.get("beatEndSeconds", 0.50))
    before_dialogue_target = float(breathing.get("beforeDialogueSeconds", 0.35))
    minimum_breath = float(breathing.get("minimumSeconds", 0.35))
    maximum_breath = float(breathing.get("maximumSeconds", 0.55))
    breath_overrides = config.get("narrationBreathOverrides", {})
    if not isinstance(breath_overrides, dict):
        raise TimelineIncomplete("narrationBreathOverrides must map scene IDs to seconds and reason")
    narration_ids = {str(row['still_id']) for row in rows if not row.get('speaker')}
    for scene_id, override in breath_overrides.items():
        if scene_id not in narration_ids or not isinstance(override, dict):
            raise TimelineIncomplete(f"Narration breath override has no narration scene: {scene_id}")
        seconds = override.get('seconds')
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < minimum_breath
                or not isinstance(override.get('reason'), str) or not override['reason'].strip()):
            raise TimelineIncomplete(f"Narration breath override needs finite seconds and authored reason: {scene_id}")
    handles = config.get("dialogueHandles", {})
    dialogue_pre = float(handles.get("preSeconds", 0.70))
    dialogue_post = float(handles.get("postSeconds", 0.90))
    preserve_dialogue = config.get("preserveCompleteDialogueClips", False)
    if type(preserve_dialogue) is not bool:
        raise TimelineIncomplete("preserveCompleteDialogueClips must be a boolean")

    events: list[dict[str, Any]] = []
    cursor_frames = 0
    for index, row in enumerate(rows):
        scene_id = str(row["still_id"])
        start = cursor_frames / DELIVERY_FPS
        speaker = row.get("speaker")
        if speaker:
            source, info, entry, output = dialogue_media[scene_id]
            natural_frames = info.video_frames or math.floor((info.video_duration or info.duration) * DELIVERY_FPS + 1e-6)
            if natural_frames <= 0:
                raise TimelineIncomplete(f"Dialogue has no video frames: {scene_id}")
            speech = _speech_window(entry, output)
            if speech is None:
                source_start_frame = 0
                source_end_frame = natural_frames
                speech_start = 0.0
                speech_end = min(
                    natural_frames / DELIVERY_FPS,
                    info.audio_duration if info.audio_duration is not None else natural_frames / DELIVERY_FPS,
                )
                handle_basis = "full-native-clip-conservative"
                warnings.append(
                    {
                        "type": "dialogue-speech-window-unmeasured",
                        "sceneId": scene_id,
                        "policy": "Keep the complete native clip; refine subtitle onset after measured dialogue QA",
                    }
                )
            else:
                speech_start, speech_end = speech
                natural_seconds = natural_frames / DELIVERY_FPS
                if not 0 <= speech_start < speech_end <= natural_seconds + 1 / DELIVERY_FPS:
                    raise TimelineIncomplete(
                        f"Invalid dialogue speech window for {scene_id}: {speech_start}..{speech_end}"
                    )
                source_start_frame = max(0, math.floor((speech_start - dialogue_pre) * DELIVERY_FPS + 1e-7))
                source_end_frame = min(
                    natural_frames,
                    math.ceil((speech_end + dialogue_post) * DELIVERY_FPS - 1e-7),
                )
                handle_basis = "measured-full-speech-window-plus-authored-handles"
                if preserve_dialogue:
                    source_start_frame = 0
                    source_end_frame = natural_frames
                    handle_basis = "measured-subtitle-window-complete-native-performance"
            duration_frames = source_end_frame - source_start_frame
            duration = duration_frames / DELIVERY_FPS
            source_start = source_start_frame / DELIVERY_FPS
            subtitle_at = max(0.0, speech_start - source_start)
            subtitle_duration = min(speech_end - speech_start, duration - subtitle_at)
            if subtitle_duration <= 0:
                raise TimelineIncomplete(f"Dialogue subtitle window vanished: {scene_id}")
            audio_duration = info.audio_duration or duration
            if speech is None and audio_duration > duration + 1 / DELIVERY_FPS + 0.01:
                raise TimelineIncomplete(
                    f"Dialogue audio outlasts picture for {scene_id}: {audio_duration:.3f}s > {duration:.3f}s"
                )
            event = {
                "id": scene_id,
                "shot": _scene_number(scene_id),
                "kind": "dialogue",
                "speaker": speaker,
                "sourceSceneId": output.get("sourceSceneId", scene_id),
                **({"pictureCrop": output["pictureCrop"], "pictureCropReason": output["pictureCropReason"]}
                   if output.get("pictureCrop") else {}),
                **({"pictureCropFrom": output["pictureCropFrom"]}
                   if output.get("pictureCropFrom") is not None else {}),
                "audio": _portable_path(source, episode),
                "picture": _portable_path(source, episode),
                "audioSha256": measured[str(source)]["sha256"],
                "start": round(start, 6),
                "duration": round(duration, 6),
                "end": round(start + duration, 6),
                "sourceStart": round(source_start, 6),
                "audioSourceStart": round(source_start, 6),
                "audioSourceDuration": round(duration, 6),
                "subtitleAudioAt": round(subtitle_at, 6),
                "subtitleAudioDuration": round(subtitle_duration, 6),
                "dialogueHandles": {
                    "basis": handle_basis,
                    "preSeconds": round(subtitle_at, 6),
                    "postSeconds": round(duration - subtitle_at - subtitle_duration, 6),
                },
                "admission": {
                    "sidecar": output["sidecar"],
                    "sidecarSha256": output["sidecarSha256"],
                    "reviewExport": output["reviewExport"],
                    "reviewExportSha256": output["reviewExportSha256"],
                    "reviewFingerprint": output["reviewFingerprint"],
                    "executionManifestSha256": output.get('executionManifestSha256', _sha256(dialogue_path)),
                    "executionManifest": output.get('executionManifest', _portable_path(dialogue_path, episode)),
                },
                "text": entry.get("hindi") or entry.get("exact_hindi_line") or row["narration_hi"],
                "subtitleTextHi": effective_text[scene_id]["hi"],
                "subtitleTextEn": effective_text[scene_id]["en"],
            }
            cursor_frames += duration_frames
        else:
            source, info = narration_media[scene_id]
            audio_duration = info.audio_duration or info.duration
            next_row = rows[index + 1] if index + 1 < len(rows) else None
            if next_row is None or next_row.get("beat", {}).get("number") != row.get("beat", {}).get("number"):
                breath_target = beat_end_target
                breath_reason = "beat-boundary"
            elif next_row.get("speaker"):
                breath_target = before_dialogue_target
                breath_reason = "dialogue-entry"
            else:
                breath_target = same_beat_target
                breath_reason = "intra-beat"
            override = breath_overrides.get(scene_id)
            if override:
                breath_target = float(override['seconds'])
                breath_reason = 'authored-motion-breath: ' + override['reason'].strip()
            duration_frames = math.ceil((audio_duration + breath_target) * DELIVERY_FPS - 1e-9)
            duration = duration_frames / DELIVERY_FPS
            actual_breath = duration - audio_duration
            breath_ceiling = breath_target + 1 / DELIVERY_FPS if override else maximum_breath
            if not minimum_breath - 1e-6 <= actual_breath <= breath_ceiling + 1e-6:
                raise TimelineIncomplete(
                    f"Frame-aligned narration breath outside contract for {scene_id}: {actual_breath:.6f}s"
                )
            event = {
                "id": scene_id,
                "shot": _scene_number(scene_id),
                "kind": "narration",
                "speaker": None,
                "audio": _portable_path(source, episode),
                "picture": None,
                "audioSha256": measured[str(source)]["sha256"],
                "start": round(start, 6),
                "duration": round(duration, 6),
                "end": round(start + duration, 6),
                "audioSourceStart": 0.0,
                "audioSourceDuration": round(audio_duration, 6),
                "subtitleAudioAt": 0.0,
                "subtitleAudioDuration": round(audio_duration, 6),
                "narrationBreath": {
                    "targetSeconds": breath_target,
                    "actualSeconds": round(actual_breath, 6),
                    "reason": breath_reason,
                },
                "text": effective_text[scene_id]["hi"],
                "subtitleTextHi": effective_text[scene_id]["hi"],
                "subtitleTextEn": effective_text[scene_id]["en"],
            }
            cursor_frames += duration_frames
        if effective_text[scene_id].get('cuePairs'):
            event['subtitleCuePairs'] = effective_text[scene_id]['cuePairs']
            event['subtitleCueWeights'] = effective_text[scene_id]['cueWeights']
        events.append(event)

    for event_index, event in enumerate(events):
        for candidate_index, candidate in enumerate(h3_by_scene.get(event['id'], [])):
            link = candidate.get('continuationOf', {})
            if link.get('parentKind') != 'dialogue':
                continue
            parent = events[event_index-1] if event_index else {}
            selected_endpoint = (parent.get('sourceStart', 0) + parent.get('duration', 0)) * DELIVERY_FPS
            if (candidate_index != 0 or parent.get('kind') != 'dialogue'
                    or parent.get('picture') != link['source']
                    or parent.get('audioSha256') != link['sourceSha256']
                    or abs(selected_endpoint-link['endFrameExclusive']) > 1e-3):
                errors.append(dict(type='h3-continuation-contract',sceneId=event['id'],
                                   error='Selected preceding dialogue no longer ends on the bound native endpoint'))

    minimum_picture_frames = round(float(config.get("minimumPictureSegmentSeconds", 1.3)) * DELIVERY_FPS)
    micro_words = set(config.get("microCutRoleWords", MICRO_CUT_ROLE_WORDS))
    blocks: list[dict[str, Any]] = []
    coverage_borrowing: list[dict[str, Any]] = []
    block_number = 0
    index = 0
    while index < len(events):
        block_number += 1
        event = events[index]
        if event["kind"] == "dialogue":
            duration_frames = round(event["duration"] * DELIVERY_FPS)
            source_start_frame = round(event["sourceStart"] * DELIVERY_FPS)
            windows = [(0, duration_frames, event.get("pictureCrop"))]
            if event.get("pictureCropFrom") is not None:
                cut_frame = round(event["pictureCropFrom"] * DELIVERY_FPS) - source_start_frame
                if not 0 < cut_frame < duration_frames:
                    raise TimelineIncomplete(f"Timed crop falls outside the kept dialogue window: {event['id']}")
                windows = [(0, cut_frame, None), (cut_frame, duration_frames, event["pictureCrop"])]
            for part, (left, right, crop) in enumerate(windows, 1):
                part_frames = right - left
                part_duration = round(part_frames / DELIVERY_FPS, 6)
                blocks.append(
                {
                    "id": f"b{block_number:03d}" + (f"-p{part}" if len(windows) > 1 else ""),
                    "kind": "dialogue-fixed-sync",
                    "start": round(event["start"] + left / DELIVERY_FPS, 6),
                    "duration": part_duration,
                    "events": [event["id"]],
                    "candidates": [
                        {
                            "path": event["picture"],
                            "sourceStart": round((source_start_frame + left) / DELIVERY_FPS, 6),
                            "duration": part_duration,
                            "frames": part_frames,
                            "role": "dialogue-fixed-sync",
                            "status": "approved",
                            "sourceSha256": event["audioSha256"],
                            **({"crop": crop} if crop else {}),
                        }
                    ],
                    "availableSeconds": part_duration,
                    "selectedSeconds": part_duration,
                    "status": "ready-native-picture-audio-sync",
                }
                )
            index += 1
            continue

        end_index = index
        while end_index < len(events) and events[end_index]["kind"] == "narration":
            end_index += 1
        block_events = events[index:end_index]
        target_by_scene = {
            row["id"]: round(float(row["duration"]) * DELIVERY_FPS) for row in block_events
        }
        allocation_targets: dict[str, int] = {}
        deficits: dict[str, int] = {}
        surplus: dict[str, int] = {}
        for block_event in block_events:
            scene_id = block_event["id"]
            available = sum(int(row["availableFrames"]) for row in h3_by_scene.get(scene_id, []))
            target = target_by_scene[scene_id]
            allocation_targets[scene_id] = min(available, target)
            deficits[scene_id] = max(0, target - available)
            surplus[scene_id] = max(0, available - target)

        # Borrow only from the nearest causal neighbor, while preserving the
        # source order in the final flattened candidate list.
        unresolved = sum(deficits.values())
        for receiver_index, receiver in enumerate(block_events):
            if config.get("allowCausalBorrowing", True) is False:
                break
            receiver_id = receiver["id"]
            need = deficits[receiver_id]
            if need <= 0:
                continue
            donor_order = sorted(
                range(len(block_events)), key=lambda item: (abs(item - receiver_index), item)
            )
            for donor_index in donor_order:
                donor_id = block_events[donor_index]["id"]
                give = min(need, surplus[donor_id])
                if give <= 0:
                    continue
                allocation_targets[donor_id] += give
                surplus[donor_id] -= give
                need -= give
                unresolved -= give
                coverage_borrowing.append(
                    {
                        "block": f"b{block_number:03d}",
                        "donorScene": donor_id,
                        "receiverScene": receiver_id,
                        "frames": give,
                        "seconds": round(give / DELIVERY_FPS, 6),
                        "reason": "nearest causally adjacent accepted H3 coverage",
                    }
                )
                if need == 0:
                    break
            deficits[receiver_id] = need

        if unresolved:
            for scene_id, frames in deficits.items():
                if frames:
                    missing.append(
                        {
                            "type": "h3-coverage-deficit",
                            "block": f"b{block_number:03d}",
                            "sceneId": scene_id,
                            "frames": frames,
                            "seconds": round(frames / DELIVERY_FPS, 6),
                        }
                    )
        chosen: list[dict[str, Any]] = []
        for block_event in block_events:
            scene_id = block_event["id"]
            target = allocation_targets[scene_id]
            if target <= 0:
                continue
            selected = _allocate_windows(
                h3_by_scene.get(scene_id, []),
                target,
                minimum_picture_frames,
                micro_words,
                warnings,
                scene_id,
            )
            chosen.extend(_candidate_output(row, episode) for row in selected)
        block_frames = sum(target_by_scene.values())
        selected_frames = sum(row["frames"] for row in chosen)
        available_frames = sum(
            row["availableFrames"]
            for block_event in block_events
            for row in h3_by_scene.get(block_event["id"], [])
        )
        blocks.append(
            {
                "id": f"b{block_number:03d}",
                "kind": "narration-fluid-picture",
                "start": block_events[0]["start"],
                "duration": round(block_frames / DELIVERY_FPS, 6),
                "events": [row["id"] for row in block_events],
                "candidates": chosen,
                "availableSeconds": round(available_frames / DELIVERY_FPS, 6),
                "selectedSeconds": round(selected_frames / DELIVERY_FPS, 6),
                "surplusSeconds": round(max(0, available_frames - block_frames) / DELIVERY_FPS, 6),
                "deficitSeconds": round(max(0, block_frames - selected_frames) / DELIVERY_FPS, 6),
                "status": "ready" if selected_frames == block_frames else "blocked-coverage-deficit",
            }
        )
        index = end_index

    # Preserve the exact allocation evidence even when picture coverage is
    # short.  A continuation planner must be able to prove which accepted
    # source frames were consumed before it commissions another H3 frame; a
    # block-level duration summary is not strong enough for that decision.
    # This remains readiness evidence, not a renderable edit plan.
    readiness["pictureBlocks"] = blocks
    readiness["exactCoverageDeficits"] = [
        dict(row)
        for row in missing
        if row.get("type") == "h3-coverage-deficit"
    ]

    # Speech measurement is valuable even when picture coverage is short.  A
    # previous handoff discarded this clock whenever H3 ran out, which
    # made it impossible to derive exact score ranges or author only the
    # continuations actually needed.  This document is deliberately *not* an
    # edit plan and no renderer accepts it; it is a hash-bound deficit-planning
    # clock that can safely survive a blocked coverage pass.
    measured_clock_path = out_dir / "measured-clock.json"
    clock_inputs = {
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "documents": readiness["documents"],
        "media": measured,
        "events": events,
    }
    measured_clock = {
        "schemaVersion": 1,
        "kind": "measured-speech-clock-not-renderable-edit-plan",
        "episode": storyboard.get("episode", episode.name),
        "language": "hi",
        "deliveryFps": DELIVERY_FPS,
        "clockFingerprint": _canonical_sha(clock_inputs),
        "runtimeSeconds": round(cursor_frames / DELIVERY_FPS, 6),
        "eventCount": len(events),
        "events": events,
        "inputEvidence": clock_inputs,
        "renderable": False,
        "reason": (
            "This clock measures complete admitted speech. Picture allocation, "
            "score admission and the final timeline fingerprint remain separate gates."
        ),
    }
    _atomic_json(measured_clock_path, measured_clock)
    readiness["measuredClock"] = {
        "path": str(measured_clock_path),
        "sha256": _sha256(measured_clock_path),
        "clockFingerprint": measured_clock["clockFingerprint"],
        "runtimeSeconds": measured_clock["runtimeSeconds"],
        "renderable": False,
    }
    if str(measured_clock_path) not in readiness["outputsWritten"]:
        readiness["outputsWritten"].append(str(measured_clock_path))

    if measure_pending_score:
        readiness['ready'] = False
        readiness['mode'] = 'measurement-only-score-gates-preserved'
        readiness['missing'] = missing
        readiness['errors'] = errors
        readiness['warnings'] = warnings
        _atomic_json(readiness_path, readiness)
        return readiness, None

    if missing or errors:
        readiness["ready"] = False
        readiness["missing"] = missing
        readiness["warnings"] = warnings
        readiness["coverageBorrowing"] = coverage_borrowing
        _atomic_json(readiness_path, readiness)
        raise TimelineIncomplete(
            f"Measured speech exposed picture deficits or invalid picture contracts; see {readiness_path}"
        )

    runtime = cursor_frames / DELIVERY_FPS
    inputs_for_fingerprint = {
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "documents": readiness["documents"],
        "media": measured,
        "events": events,
        "blocks": blocks,
    }
    fingerprint = _canonical_sha(inputs_for_fingerprint)
    edit_plan = {
        "schemaVersion": 2,
        "kind": "renderable-edit-plan",
        "renderable": True,
        "renderScope": config.get('reviewScope', 'source-admitted-render; final watch separate'),
        "finalApproved": False,
        "episode": storyboard.get("episode", episode.name),
        "version": int(config.get("version", 1)),
        "language": "hi",
        "timelineFingerprint": fingerprint,
        "runtimeSeconds": round(runtime, 6),
        "runtimeMinutes": round(runtime / 60, 3),
        "deliveryFps": DELIVERY_FPS,
        "picturePolicy": (
            "accepted generated motion only; H3 picture is silent; Grok dialogue picture/audio stay coupled; "
            "no still padding, dead holds, implicit retiming, or invented media; an H3 speed is legal only "
            "when explicitly authored on its approved ledger segment"
        ),
        "timingPolicy": (
            "measured complete Hindi speech; default frame-aligned 0.35-0.55 second narration breaths "
            "with explicit reasoned, motion-covered per-scene overrides where authored; "
            "native dialogue clips retain the full source or measured full-speech windows plus longer handles"
        ),
        "events": events,
        "pictureBlocks": blocks,
        "pictureOwnershipContracts": ledger_doc.get("pictureOwnershipContracts", []),
        "scorePlacements": validated_score_placements,
        "musicBreaths": config.get("musicBreaths", []),
        "sfxManifest": config.get("sfxManifest"),
    }
    subtitle_hi = _subtitle_doc("hi", events, fingerprint)
    subtitle_en = _subtitle_doc("en", events, fingerprint)
    explicit_retimes = [
        {
            "block": block["id"],
            "sceneId": candidate["sceneId"],
            "source": candidate["path"],
            "sourceStart": candidate["sourceStart"],
            "sourceOut": candidate["sourceOut"],
            "sourceDuration": candidate["sourceDuration"],
            "speed": candidate["speed"],
            "outputFrames": candidate["frames"],
            "outputDuration": candidate["duration"],
            "role": candidate["role"],
            "verdict": candidate["verdict"],
        }
        for block in blocks
        for candidate in block["candidates"]
        if isinstance(candidate, dict) and abs(float(candidate.get("speed", 1.0)) - 1.0) > 1e-9
    ]
    plan_path = out_dir / "edit-plan.json"
    hi_path = out_dir / "subtitle-hi-timing.json"
    en_path = out_dir / "subtitle-en-timing.json"
    audit_path = out_dir / "timeline-audit.json"
    audit = {
        "schemaVersion": 1,
        "episode": edit_plan["episode"],
        "timelineFingerprint": fingerprint,
        "runtimeSeconds": edit_plan["runtimeSeconds"],
        "eventCount": len(events),
        "narrationCount": sum(row["kind"] == "narration" for row in events),
        "dialogueCount": sum(row["kind"] == "dialogue" for row in events),
        "pictureBlockCount": len(blocks),
        "coverageBorrowing": coverage_borrowing,
        "explicitH3Retimes": explicit_retimes,
        "warnings": warnings,
        "inputEvidence": inputs_for_fingerprint,
        "outputs": {
            "editPlan": str(plan_path),
            "subtitleHi": str(hi_path),
            "subtitleEn": str(en_path),
        },
    }
    _atomic_json(plan_path, edit_plan)
    _atomic_json(hi_path, subtitle_hi)
    _atomic_json(en_path, subtitle_en)
    _atomic_json(audit_path, audit)
    readiness.update(
        {
            "ready": True,
            "warnings": warnings,
            "timelineFingerprint": fingerprint,
            "runtimeSeconds": edit_plan["runtimeSeconds"],
            "outputsWritten": [
                str(readiness_path),
                str(plan_path),
                str(hi_path),
                str(en_path),
                str(audit_path),
            ],
        }
    )
    _atomic_json(readiness_path, readiness)
    return readiness, edit_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        help="Timeline-input JSON (default: EP/work/assembly-v1/timeline-inputs.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure what exists and write readiness only; missing inputs do not fail",
    )
    parser.add_argument('--measure-pending-score', action='store_true',
                        help='Measure full speech/picture allocation while score admission is pending; never emit an edit plan')
    args = parser.parse_args()
    episode = args.episode.resolve()
    config_path = args.input.resolve() if args.input else episode / "work/assembly-v1/timeline-inputs.json"
    try:
        readiness, plan = build_timeline(episode, config_path, dry_run=args.dry_run,
                                        measure_pending_score=args.measure_pending_score)
    except TimelineIncomplete as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "ready": readiness["ready"],
                "missing": len(readiness.get("missing", [])),
                "errors": len(readiness.get("errors", [])),
                "warnings": len(readiness.get("warnings", [])),
                "runtimeSeconds": plan.get("runtimeSeconds") if plan else None,
                "outputs": readiness["outputsWritten"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
