#!/usr/bin/env python3
"""Offline, advisory score-candidate analysis and ranking.

The analyzer deliberately measures only acoustic properties: dynamics, spectral
weight, onset regularity, breathing, structural contrast, and ending behavior.
It maps an authored semantic wave onto transparent *technical proxies*.  It
does not infer emotion, musical meaning, Indian musical quality, or approval.

Default operation is a zero-write inventory/dry run.  ``--analyze`` decodes
locally with ffmpeg and writes one hash-bound report below the episode's work
directory.  There is no network code in this module.

Decoding runs on the qualified GPU worker.  ``--analyze --audition`` decodes on
any host with ffmpeg instead; that report is written beside the normal one as
``*-audition-not-final.json``, records the real host and ffmpeg build, and is
labelled inadmissible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


ANALYZER_VERSION = 1
SEMANTIC_DIMENSIONS = (
    "awe", "dread", "devotion", "grief", "hope", "scale", "kinetic",
    "intimacy",
)
TRAJECTORY_FEATURES = (
    "rmsEnergy", "lowFrequency", "spectralBrightness", "spectralSpread",
    "onsetActivity",
)
GLOBAL_FEATURES = (
    "dynamicRange", "tempoConfidence", "breathing", "sectionContrast",
)
ENDING_FEATURES = ("tailDrop", "terminalSilence", "abruptness")
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
TAKE_SIDECAR = re.compile(
    r"(?P<cue>[a-z0-9][a-z0-9._-]*)-take-(?P<take>[1-9][0-9]*)\.source\.json\Z"
)


class AnalysisError(ValueError):
    """A fail-closed plan, provenance, or media error."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must be a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def resolved(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"{label} must be a non-empty path")
    return (base / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()


def child_path(base: Path, value: Any, label: str) -> Path:
    """Resolve a write/input directory while keeping it below the manifest."""
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise AnalysisError(f"{label} must stay below the analysis manifest")
    return (base / relative).resolve()


def checked_binding(base: Path, spec: dict[str, Any], path_key: str,
                    hash_key: str, label: str) -> tuple[Path, dict[str, Any], str]:
    path = resolved(base, spec.get(path_key), label)
    if not path.is_file():
        raise AnalysisError(f"{label} is missing: {path}")
    actual_hash = sha_file(path)
    expected_hash = spec.get(hash_key)
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise AnalysisError(
            f"{label} changed after the analysis manifest was frozen: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return path, read_json(path, label), actual_hash


def scene_number(shot_id: Any) -> int:
    match = re.search(r"-s([0-9]+)\Z", str(shot_id))
    if not match:
        raise AnalysisError(f"malformed semantic shot id: {shot_id}")
    return int(match.group(1))


def validate_points(points: Any, label: str) -> list[list[float]]:
    if not isinstance(points, list) or len(points) < 2:
        raise AnalysisError(f"{label} needs at least two [position,value] points")
    result: list[list[float]] = []
    previous = -1.0
    for point in points:
        if (not isinstance(point, list) or len(point) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           for x in point)):
            raise AnalysisError(f"{label} contains a malformed control point")
        position, value = float(point[0]), float(point[1])
        if not 0.0 <= position <= 1.0 or not 0.0 <= value <= 1.0:
            raise AnalysisError(f"{label} control points must remain in 0..1")
        if position <= previous:
            raise AnalysisError(f"{label} positions must be strictly increasing")
        previous = position
        result.append([position, value])
    if result[0][0] != 0.0 or result[-1][0] != 1.0:
        raise AnalysisError(f"{label} must explicitly bind positions 0 and 1")
    return result


def load_plan(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path, "analysis manifest")
    if manifest.get("version") != 1:
        raise AnalysisError("analysis manifest version must be 1")
    base = manifest_path.parent
    creative_path, creative, creative_hash = checked_binding(
        base, manifest, "creativeDraft", "creativeDraftSha256", "creative draft"
    )
    anchors_path, anchors, anchors_hash = checked_binding(
        base, manifest, "semanticAnchors", "semanticAnchorsSha256",
        "semantic anchors",
    )
    execution_path = resolved(base, manifest.get("executionManifest"),
                              "execution manifest")
    execution = read_json(execution_path, "execution manifest")
    execution_hash = sha_file(execution_path)

    # The finite Suno compiler also accepts unversioned drafts. Preserve their
    # paid-request hashes; never rewrite an executed draft to add a version.
    unversioned = ("version" not in creative and manifest.get("creativeDraftSchema")
                   == "finite-suno-unversioned-v1")
    if (creative.get("version") != 1 and not unversioned) or anchors.get("version") != 1:
        raise AnalysisError("creative draft and semantic anchors must be version 1")
    if anchors.get("kind") != "authored-semantic-target-not-audio-measurement":
        raise AnalysisError("semantic anchor kind does not preserve the advisory boundary")
    if tuple(anchors.get("dimensions") or ()) != SEMANTIC_DIMENSIONS:
        raise AnalysisError("semantic dimensions/order differ from the analyzer contract")

    cues = creative.get("cues")
    common = creative.get("commonPayload")
    if not isinstance(cues, list) or not cues or not isinstance(common, dict):
        raise AnalysisError("creative draft lacks cues or commonPayload")
    cue_ids: list[str] = []
    for cue in cues:
        cue_id = cue.get("id") if isinstance(cue, dict) else None
        shots = cue.get("shots") if isinstance(cue, dict) else None
        if (not isinstance(cue_id, str) or not SAFE_ID.fullmatch(cue_id)
                or cue_id in cue_ids):
            raise AnalysisError(f"duplicate or malformed movement id: {cue_id}")
        if (not isinstance(shots, list) or len(shots) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in shots)
                or shots[0] > shots[1]):
            raise AnalysisError(f"{cue_id}: shots must be an ascending integer pair")
        cue_ids.append(cue_id)

    anchor_rows = anchors.get("anchors")
    if not isinstance(anchor_rows, list) or not anchor_rows:
        raise AnalysisError("semantic anchors are empty")
    parsed_anchors: list[dict[str, Any]] = []
    previous_scene = -1
    for row in anchor_rows:
        number = scene_number(row.get("shotId") if isinstance(row, dict) else None)
        values = row.get("values") if isinstance(row, dict) else None
        if number <= previous_scene:
            raise AnalysisError("semantic anchors must be strictly ordered by scene")
        previous_scene = number
        if not isinstance(values, dict) or tuple(values.keys()) != SEMANTIC_DIMENSIONS:
            raise AnalysisError(f"scene {number}: semantic value schema changed")
        if any(not isinstance(values[name], (int, float)) or isinstance(values[name], bool)
               or not 0 <= values[name] <= 100 for name in SEMANTIC_DIMENSIONS):
            raise AnalysisError(f"scene {number}: semantic values must be 0..100")
        parsed_anchors.append({"scene": number, "values": values})
    hard_transitions = anchors.get("hardTransitions") or []
    if (not isinstance(hard_transitions, list)
            or any(not isinstance(value, int) or isinstance(value, bool)
                   for value in hard_transitions)
            or hard_transitions != sorted(set(hard_transitions))):
        raise AnalysisError("hardTransitions must be sorted unique scene integers")

    materialized = [
        {**cue, "payload": {**common, **cue.get("payload", {})}}
        for cue in cues
    ]
    if execution.get("sourceDraftSha256") != creative_hash:
        raise AnalysisError("execution manifest is not bound to this creative draft")
    if execution.get("cues") != materialized:
        raise AnalysisError("execution manifest cues differ from the creative draft")

    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise AnalysisError("sampling settings are missing")
    expected_sampling = {"sampleRate", "frameSize", "hopSize", "trajectoryBins",
                         "sectionBins", "lowFrequencyHz"}
    if set(sampling) != expected_sampling:
        raise AnalysisError(f"sampling keys must be exactly {sorted(expected_sampling)}")
    sample_rate = sampling["sampleRate"]
    frame_size = sampling["frameSize"]
    hop_size = sampling["hopSize"]
    trajectory_bins = sampling["trajectoryBins"]
    section_bins = sampling["sectionBins"]
    low_hz = sampling["lowFrequencyHz"]
    integer_values = (sample_rate, frame_size, hop_size, trajectory_bins, section_bins)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in integer_values):
        raise AnalysisError("sampling integer fields must be integers")
    if not (8000 <= sample_rate <= 48000 and 256 <= frame_size <= 8192
            and 64 <= hop_size <= frame_size and 8 <= trajectory_bins <= 64
            and 4 <= section_bins <= 24
            and isinstance(low_hz, (int, float)) and 40 <= low_hz <= 300):
        raise AnalysisError("sampling settings are outside the finite safe ranges")

    profiles = manifest.get("movementProfiles")
    if not isinstance(profiles, dict) or set(profiles) != set(cue_ids):
        raise AnalysisError("movementProfiles must bind every and only creative cue id")
    for cue_id, profile in profiles.items():
        if not isinstance(profile, dict):
            raise AnalysisError(f"{cue_id}: movement profile must be an object")
        blend = profile.get("promptBlend")
        if not isinstance(blend, (int, float)) or isinstance(blend, bool) or not 0 <= blend <= 1:
            raise AnalysisError(f"{cue_id}: promptBlend must be 0..1")
        curves = profile.get("curves")
        if not isinstance(curves, dict) or not set(curves).issubset(TRAJECTORY_FEATURES):
            raise AnalysisError(f"{cue_id}: unknown proxy trajectory")
        for name, points in curves.items():
            validate_points(points, f"{cue_id}.{name}")
        globals_target = profile.get("globals")
        ending_target = profile.get("ending")
        if not isinstance(globals_target, dict) or set(globals_target) != set(GLOBAL_FEATURES):
            raise AnalysisError(f"{cue_id}: global targets must bind {GLOBAL_FEATURES}")
        if not isinstance(ending_target, dict) or set(ending_target) != set(ENDING_FEATURES):
            raise AnalysisError(f"{cue_id}: ending targets must bind {ENDING_FEATURES}")
        for family, values in (("globals", globals_target), ("ending", ending_target)):
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not 0 <= value <= 1 for value in values.values()):
                raise AnalysisError(f"{cue_id}.{family}: targets must be normalized 0..1")

    score_weights = manifest.get("scoreWeights")
    expected_weights = set(TRAJECTORY_FEATURES + GLOBAL_FEATURES + ENDING_FEATURES)
    if (not isinstance(score_weights, dict) or set(score_weights) != expected_weights
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or value <= 0 for value in score_weights.values())):
        raise AnalysisError("scoreWeights must give every measurable proxy a positive weight")

    candidate_directory = child_path(base, manifest.get("candidateDirectory"),
                                     "candidateDirectory")
    report_path = child_path(base, manifest.get("reportPath"), "reportPath")
    if "deliverables" in {part.lower() for part in report_path.parts}:
        raise AnalysisError("the advisory analyzer may never write below deliverables")
    if report_path.suffix.lower() != ".json":
        raise AnalysisError("reportPath must be a JSON file")

    return {
        "manifestPath": manifest_path,
        "manifestHash": sha_file(manifest_path),
        "manifest": manifest,
        "creativePath": creative_path,
        "creative": creative,
        "creativeHash": creative_hash,
        "anchorsPath": anchors_path,
        "anchors": anchors,
        "anchorsHash": anchors_hash,
        "parsedAnchors": parsed_anchors,
        "hardTransitions": hard_transitions,
        "executionPath": execution_path,
        "execution": execution,
        "executionHash": execution_hash,
        "cues": cues,
        "materializedCues": materialized,
        "cueIds": cue_ids,
        "candidateDirectory": candidate_directory,
        "reportPath": report_path,
        "sampling": sampling,
        "profiles": profiles,
        "scoreWeights": score_weights,
    }


def discover_candidates(plan: dict[str, Any]) -> dict[str, Any]:
    directory: Path = plan["candidateDirectory"]
    found: list[dict[str, Any]] = []
    orphans: list[str] = []
    seen: set[tuple[str, int]] = set()
    if directory.is_dir():
        sidecars = sorted(directory.glob("*.source.json"))
        sidecar_media = {
            path.name.removesuffix(".source.json") for path in sidecars
        }
        for media in sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"}
        ):
            if media.stem not in sidecar_media:
                orphans.append(str(media))
    else:
        sidecars = []

    materialized_by_id = {cue["id"]: cue for cue in plan["materializedCues"]}
    for sidecar_path in sidecars:
        match = TAKE_SIDECAR.fullmatch(sidecar_path.name)
        if not match:
            raise AnalysisError(f"malformed candidate sidecar name: {sidecar_path}")
        cue_id = match.group("cue")
        take = int(match.group("take"))
        if cue_id not in materialized_by_id:
            raise AnalysisError(f"candidate sidecar names unknown cue: {cue_id}")
        if (cue_id, take) in seen:
            raise AnalysisError(f"duplicate candidate identity: {cue_id} take {take}")
        seen.add((cue_id, take))
        sidecar = read_json(sidecar_path, "candidate sidecar")
        if sidecar.get("version") != 1 or sidecar.get("episode") != plan["manifest"].get("episode"):
            raise AnalysisError(f"candidate episode/version mismatch: {sidecar_path}")
        if sidecar.get("cueId") != cue_id or sidecar.get("candidateIndex") != take:
            raise AnalysisError(f"candidate identity mismatch: {sidecar_path}")
        if sidecar.get("sourceDraftSha256") != plan["creativeHash"]:
            raise AnalysisError(f"candidate uses another creative draft: {sidecar_path}")
        if sidecar.get("executionManifestSha256") != plan["executionHash"]:
            raise AnalysisError(f"candidate uses another execution manifest: {sidecar_path}")
        expected_request = materialized_by_id[cue_id]["payload"]
        if sidecar.get("request") != expected_request:
            raise AnalysisError(f"candidate request differs from frozen cue: {sidecar_path}")
        if sidecar.get("requestSha256") != sha_bytes(canonical_bytes(expected_request)):
            raise AnalysisError(f"candidate request hash mismatch: {sidecar_path}")
        media_path = resolved(sidecar_path.parent, sidecar.get("file"), "candidate file")
        try:
            media_path.relative_to(directory)
        except ValueError as exc:
            raise AnalysisError(f"candidate escapes its source directory: {media_path}") from exc
        if not media_path.is_file():
            raise AnalysisError(f"candidate file is missing: {media_path}")
        actual_hash = sha_file(media_path)
        if sidecar.get("sha256") != actual_hash:
            raise AnalysisError(f"candidate media hash mismatch: {media_path}")
        found.append({
            "cueId": cue_id,
            "take": take,
            "id": f"{cue_id}-take-{take}",
            "file": str(media_path),
            "sha256": actual_hash,
            "sidecar": str(sidecar_path),
            "sidecarSha256": sha_file(sidecar_path),
        })
    found.sort(key=lambda item: (plan["cueIds"].index(item["cueId"]), item["take"]))
    count_by_cue = {cue_id: 0 for cue_id in plan["cueIds"]}
    for item in found:
        count_by_cue[item["cueId"]] += 1
    expected = int(plan["execution"].get("expectedCandidatesPerRequest") or 1)
    return {
        "found": found,
        "countByMovement": count_by_cue,
        "missingMovements": [key for key, value in count_by_cue.items() if value == 0],
        "partialMovements": [key for key, value in count_by_cue.items()
                             if 0 < value < expected],
        "expectedCandidatesPerMovement": expected,
        "orphans": orphans,
    }


def interp_points(points: list[list[float]], positions: np.ndarray) -> np.ndarray:
    pairs = np.asarray(points, dtype=np.float64)
    return np.interp(positions, pairs[:, 0], pairs[:, 1])


def interpolate_with_hard_barriers(plan: dict[str, Any], name: str,
                                   positions: np.ndarray) -> np.ndarray:
    """Interpolate only inside a semantic continuity partition.

    A hard-transition scene belongs to the new/right partition.  Where a
    partition has no anchor at its exact edge, the nearest anchor *inside that
    partition* is extended flat.  This prevents, for example, scene 27's
    generosity from bleeding across the scene-31 boundary into pursuit.
    """
    anchors = plan["parsedAnchors"]
    cuts = plan["hardTransitions"]
    result = np.empty(len(positions), dtype=np.float64)
    minimum = anchors[0]["scene"]
    maximum = anchors[-1]["scene"] + 1
    boundaries = [minimum] + [cut for cut in cuts if minimum < cut < maximum] + [maximum]
    for index, position in enumerate(positions):
        partition = max(0, min(
            len(boundaries) - 2,
            int(np.searchsorted(boundaries, position, side="right") - 1),
        ))
        left, right = boundaries[partition], boundaries[partition + 1]
        local = [row for row in anchors if left <= row["scene"] < right]
        if not local:
            raise AnalysisError(
                f"semantic partition {left}..{right - 1} has no anchors for {name}"
            )
        scenes = np.asarray([row["scene"] for row in local], dtype=np.float64)
        values = np.asarray([row["values"][name] / 100.0 for row in local],
                            dtype=np.float64)
        result[index] = float(np.interp(position, scenes, values))
    return result


def semantic_target(plan: dict[str, Any], cue: dict[str, Any]) -> dict[str, Any]:
    bins = int(plan["sampling"]["trajectoryBins"])
    start, end = cue["shots"]
    scene_positions = np.linspace(start, end, bins)
    absolute = plan["anchors"].get("absoluteTimes")
    if absolute is not None:
        if not isinstance(absolute, list):
            raise AnalysisError("absoluteTimes must contain measured scene intervals")
        selected = [row for row in absolute if start <= scene_number(row["shotId"]) <= end]
        if not selected:
            raise AnalysisError(f"{cue['id']}: no current measured scene intervals")
        previous_end = -1.0
        centers, scenes = [], []
        for row in selected:
            left, right = float(row["start"]), float(row["end"])
            if not math.isfinite(left + right) or left < previous_end - 1e-5 or right <= left:
                raise AnalysisError("Measured score intervals overlap or are invalid")
            previous_end = right
            centers.append((left + right) / 2)
            scenes.append(scene_number(row["shotId"]))
        measured_positions = np.linspace(float(selected[0]["start"]), float(selected[-1]["end"]), bins)
        scene_positions = np.interp(measured_positions, centers, scenes)
    semantics: dict[str, np.ndarray] = {}
    for name in SEMANTIC_DIMENSIONS:
        semantics[name] = interpolate_with_hard_barriers(plan, name, scene_positions)

    # These are explicit editing proxies, not claims that an emotion has a
    # unique or detectable acoustic signature.
    proxies = {
        "rmsEnergy": np.clip(
            0.12 + 0.35 * semantics["scale"] + 0.20 * semantics["awe"]
            + 0.20 * semantics["dread"] + 0.10 * semantics["hope"]
            + 0.08 * semantics["grief"] - 0.13 * semantics["intimacy"],
            0.04, 1,
        ),
        "lowFrequency": np.clip(
            0.50 * semantics["scale"] + 0.30 * semantics["dread"]
            + 0.20 * (1.0 - semantics["intimacy"]), 0, 1
        ),
        "spectralBrightness": np.clip(
            0.25 + 0.35 * semantics["hope"] + 0.20 * semantics["awe"]
            + 0.15 * semantics["devotion"] - 0.25 * semantics["dread"]
            - 0.15 * semantics["grief"], 0, 1
        ),
        "spectralSpread": np.clip(
            0.45 * semantics["awe"] + 0.35 * semantics["scale"]
            + 0.20 * (1.0 - semantics["intimacy"]), 0, 1
        ),
        "onsetActivity": np.clip(
            0.70 * semantics["kinetic"] + 0.18 * semantics["dread"]
            + 0.08 * semantics["hope"] - 0.18 * semantics["intimacy"]
            - 0.12 * semantics["grief"], 0, 1
        ),
    }
    profile = plan["profiles"][cue["id"]]
    blend = float(profile["promptBlend"])
    normalized_positions = np.linspace(0.0, 1.0, bins)
    for name, points in profile["curves"].items():
        prompt_curve = interp_points(validate_points(points, f"{cue['id']}.{name}"),
                                     normalized_positions)
        proxies[name] = np.clip((1.0 - blend) * proxies[name] + blend * prompt_curve,
                                0, 1)
    return {
        "scenePositions": scene_positions,
        "semantics": semantics,
        "proxies": proxies,
        "globals": {name: float(profile["globals"][name]) for name in GLOBAL_FEATURES},
        "ending": {name: float(profile["ending"][name]) for name in ENDING_FEATURES},
    }


def audition_report_path(path: Path) -> Path:
    return path.with_name(path.stem + "-audition-not-final" + path.suffix)


def audition_record() -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _common
    return {**_common.preview_host_record(None), "audition": True, "admissible": False,
            "scope": "Decoded off the qualified worker for listening and comparison only; "
                     "never a production stem, master or selection record"}


def decode_audio(path: Path, sample_rate: int, audition: bool = False) -> np.ndarray:
    if not audition and (sys.platform != "linux" or not shutil.which("nvidia-smi")):
        raise AnalysisError("Audio processing must run on the qualified GPU worker, not the Mac")
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
        "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise AnalysisError("ffmpeg is required but was not found") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise AnalysisError(f"audio decode failed for {path}: {detail}") from exc
    audio = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float64)
    if len(audio) < sample_rate or not np.all(np.isfinite(audio)):
        raise AnalysisError(f"candidate has less than one finite second of audio: {path}")
    return audio


def binned_mean(values: np.ndarray, bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(bins, dtype=np.float64)
    edges = np.linspace(0, len(values), bins + 1).astype(int)
    result = np.empty(bins, dtype=np.float64)
    for index in range(bins):
        lo, hi = edges[index], edges[index + 1]
        if hi <= lo:
            result[index] = values[min(lo, len(values) - 1)]
        else:
            result[index] = float(np.mean(values[lo:hi]))
    return result


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def tempo_from_onsets(envelope: np.ndarray, hop_seconds: float,
                      onset_count: int) -> tuple[float | None, float]:
    if onset_count < 4 or len(envelope) < 16:
        return None, 0.0
    centered = envelope - np.mean(envelope)
    denominator = float(np.dot(centered, centered)) + 1e-12
    min_lag = max(1, int(round(60.0 / (180.0 * hop_seconds))))
    max_lag = min(len(centered) - 2,
                  int(round(60.0 / (45.0 * hop_seconds))))
    if max_lag <= min_lag:
        return None, 0.0
    correlations = []
    for lag in range(min_lag, max_lag + 1):
        correlations.append(float(np.dot(centered[:-lag], centered[lag:]) / denominator))
    correlations_array = np.asarray(correlations)
    best_offset = int(np.argmax(correlations_array))
    best_lag = min_lag + best_offset
    confidence = float(np.clip(correlations_array[best_offset], 0, 1))
    density_factor = min(1.0, onset_count / 12.0)
    return 60.0 / (best_lag * hop_seconds), confidence * density_factor


def extract_features(audio: np.ndarray, sample_rate: int, frame_size: int,
                     hop_size: int, trajectory_bins: int, section_bins: int,
                     low_frequency_hz: float) -> dict[str, Any]:
    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))
    starts = np.arange(0, len(audio) - frame_size + 1, hop_size, dtype=np.int64)
    if len(starts) < 2:
        starts = np.asarray([0, max(0, len(audio) - frame_size)], dtype=np.int64)
    count = len(starts)
    rms = np.empty(count, dtype=np.float64)
    low_ratio = np.empty(count, dtype=np.float64)
    centroid = np.empty(count, dtype=np.float64)
    spread = np.empty(count, dtype=np.float64)
    spectral_flux = np.zeros(count, dtype=np.float64)
    window = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
    low_mask = (frequencies >= 20.0) & (frequencies < low_frequency_hz)
    audible_mask = (frequencies >= 20.0) & (frequencies <= min(10000, sample_rate / 2))
    previous_shape: np.ndarray | None = None
    indices = np.arange(frame_size, dtype=np.int64)
    block_size = 256
    for block_start in range(0, count, block_size):
        block_starts = starts[block_start:block_start + block_size]
        frames = audio[block_starts[:, None] + indices[None, :]]
        rms_block = np.sqrt(np.mean(frames * frames, axis=1) + 1e-20)
        spectra = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
        power = spectra * spectra
        audible_power = power[:, audible_mask]
        audible_frequencies = frequencies[audible_mask]
        totals = np.sum(audible_power, axis=1) + 1e-20
        centers = np.sum(audible_power * audible_frequencies[None, :], axis=1) / totals
        deviations = audible_frequencies[None, :] - centers[:, None]
        spreads = np.sqrt(np.sum(audible_power * deviations * deviations, axis=1) / totals)
        lows = np.sum(power[:, low_mask], axis=1) / totals
        shapes = audible_power / totals[:, None]
        combined = shapes if previous_shape is None else np.vstack([previous_shape, shapes])
        differences = np.maximum(0.0, np.diff(combined, axis=0))
        flux_block = np.sqrt(np.sum(differences * differences, axis=1))
        if previous_shape is None:
            spectral_flux[block_start] = 0.0
            spectral_flux[block_start + 1:block_start + len(block_starts)] = flux_block
        else:
            spectral_flux[block_start:block_start + len(block_starts)] = flux_block
        previous_shape = shapes[-1:]
        stop = block_start + len(block_starts)
        rms[block_start:stop] = rms_block
        low_ratio[block_start:stop] = lows
        centroid[block_start:stop] = centers
        spread[block_start:stop] = spreads

    rms_db = 20.0 * np.log10(rms + 1e-12)
    energy_rise = np.maximum(0.0, np.diff(rms_db, prepend=rms_db[0]))
    flux_scale = np.percentile(spectral_flux, 95) + 1e-12
    rise_scale = np.percentile(energy_rise, 95) + 1e-12
    onset_envelope = (0.65 * np.clip(spectral_flux / flux_scale, 0, 1)
                      + 0.35 * np.clip(energy_rise / rise_scale, 0, 1))
    median = float(np.median(onset_envelope))
    mad = float(np.median(np.abs(onset_envelope - median)))
    threshold = max(0.10, median + 1.75 * 1.4826 * mad)
    refractory = max(1, int(round(0.12 * sample_rate / hop_size)))
    candidates = np.flatnonzero(
        (onset_envelope[1:-1] >= onset_envelope[:-2])
        & (onset_envelope[1:-1] > onset_envelope[2:])
        & (onset_envelope[1:-1] >= threshold)
    ) + 1
    selected: list[int] = []
    for index in candidates:
        if not selected or index - selected[-1] >= refractory:
            selected.append(int(index))
        elif onset_envelope[index] > onset_envelope[selected[-1]]:
            selected[-1] = int(index)
    onset_mask = np.zeros(count, dtype=np.float64)
    onset_mask[selected] = 1.0

    duration = len(audio) / sample_rate
    hop_seconds = hop_size / sample_rate
    onset_density = len(selected) / max(duration, 1e-9)
    tempo_bpm, tempo_confidence = tempo_from_onsets(
        onset_envelope, hop_seconds, len(selected)
    )

    peak_rms_db = float(np.percentile(rms_db, 95))
    silence_threshold = min(-55.0, peak_rms_db - 40.0)
    silence_mask = rms_db < silence_threshold
    breathing_threshold = max(silence_threshold, float(np.percentile(rms_db, 65)) - 10.0)
    breathing_mask = rms_db < breathing_threshold
    minimum_breath_frames = max(1, int(round(0.25 / hop_seconds)))
    breathing_runs = [run for run in contiguous_runs(breathing_mask)
                      if run[1] - run[0] >= minimum_breath_frames]
    breathing_frames = sum(end - start for start, end in breathing_runs)

    # Fixed physical scaling preserves flat tracks and cross-candidate dynamic
    # differences.  No per-track percentile stretch or compression occurs.
    energy_frame = np.clip((rms_db + 60.0) / 54.0, 0, 1)
    energy_curve = binned_mean(energy_frame, trajectory_bins)
    low_curve = np.clip(binned_mean(low_ratio, trajectory_bins) / 0.55, 0, 1)
    brightness_frame = np.clip(
        (np.log2(np.maximum(centroid, 150.0)) - math.log2(150.0))
        / (math.log2(6000.0) - math.log2(150.0)), 0, 1
    )
    brightness_curve = binned_mean(brightness_frame, trajectory_bins)
    spread_frame = np.clip(spread / 5000.0, 0, 1)
    spread_curve = binned_mean(spread_frame, trajectory_bins)
    frames_per_bin = max(1.0, count / trajectory_bins)
    seconds_per_bin = frames_per_bin * hop_seconds
    onset_curve = np.clip(
        binned_mean(onset_mask, trajectory_bins) / max(hop_seconds, 1e-9) / 4.0,
        0, 1,
    )

    section_vectors = np.stack([
        binned_mean(energy_frame, section_bins),
        np.clip(binned_mean(low_ratio, section_bins) / 0.55, 0, 1),
        binned_mean(brightness_frame, section_bins),
        binned_mean(spread_frame, section_bins),
        np.clip(binned_mean(onset_mask, section_bins)
                / max(hop_seconds, 1e-9) / 4.0, 0, 1),
    ], axis=1)
    if len(section_vectors) > 1:
        section_contrast_raw = float(np.mean(np.sqrt(
            np.mean(np.diff(section_vectors, axis=0) ** 2, axis=1)
        )))
    else:
        section_contrast_raw = 0.0

    tail_seconds = min(10.0, max(3.0, duration * 0.10))
    tail_frames = max(2, min(count, int(round(tail_seconds / hop_seconds))))
    final_frames = max(1, min(count, int(round(1.0 / hop_seconds))))
    tail_db = rms_db[-tail_frames:]
    previous_lo = max(0, count - 2 * tail_frames)
    previous_hi = max(previous_lo + 1, count - tail_frames)
    previous_db = rms_db[previous_lo:previous_hi]
    tail_drop_db = float(np.median(previous_db) - np.median(rms_db[-final_frames:]))
    time_axis = np.arange(len(tail_db), dtype=np.float64) * hop_seconds
    if len(tail_db) > 1:
        slope = float(np.polyfit(time_axis, tail_db, 1)[0])
    else:
        slope = 0.0
    terminal_silence_frames = 0
    for is_silent in silence_mask[::-1]:
        if not is_silent:
            break
        terminal_silence_frames += 1
    terminal_silence_seconds = terminal_silence_frames * hop_seconds
    final_relative_db = float(np.mean(rms_db[-final_frames:]) - peak_rms_db)
    abruptness = float(np.clip((final_relative_db + 24.0) / 24.0, 0, 1))

    dynamic_range_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 20))
    global_rms = float(np.sqrt(np.mean(audio * audio) + 1e-20))
    crest_db = float(20.0 * np.log10((np.max(np.abs(audio)) + 1e-12)
                                     / (global_rms + 1e-12)))
    breathing_fraction = breathing_frames / max(count, 1)
    measured_normalized = {
        "dynamicRange": float(np.clip(dynamic_range_db / 30.0, 0, 1)),
        "tempoConfidence": float(np.clip(tempo_confidence, 0, 1)),
        "breathing": float(np.clip(breathing_fraction / 0.50, 0, 1)),
        "sectionContrast": float(np.clip(section_contrast_raw / 0.45, 0, 1)),
        "tailDrop": float(np.clip(max(0.0, tail_drop_db) / 30.0, 0, 1)),
        "terminalSilence": float(np.clip(terminal_silence_seconds / 3.0, 0, 1)),
        "abruptness": abruptness,
    }
    return {
        "durationSeconds": duration,
        "frameCount": count,
        "sampleRate": sample_rate,
        "frameSize": frame_size,
        "hopSize": hop_size,
        "trajectory": {
            "rmsEnergy": energy_curve,
            "lowFrequency": low_curve,
            "spectralBrightness": brightness_curve,
            "spectralSpread": spread_curve,
            "onsetActivity": onset_curve,
        },
        "summary": {
            "rmsMeanDbfs": float(20.0 * np.log10(global_rms + 1e-12)),
            "rmsDynamicRangeDb": dynamic_range_db,
            "crestFactorDb": crest_db,
            "lowFrequencyEnergyRatioMean": float(np.mean(low_ratio)),
            "spectralCentroidHzMean": float(np.mean(centroid)),
            "spectralCentroidHzStd": float(np.std(centroid)),
            "spectralSpreadHzMean": float(np.mean(spread)),
            "onsetCount": len(selected),
            "onsetDensityPerSecond": onset_density,
            "tempoBpmCandidate": tempo_bpm,
            "tempoConfidence01": tempo_confidence,
            "silenceThresholdDbfs": silence_threshold,
            "silenceFraction": float(np.mean(silence_mask)),
            "breathingThresholdDbfs": breathing_threshold,
            "breathingFraction": breathing_fraction,
            "breathEvents": len(breathing_runs),
            "breathEventsPerMinute": len(breathing_runs) * 60.0 / duration,
            "sectionContrastRaw": section_contrast_raw,
            "endingTailWindowSeconds": tail_seconds,
            "endingTailDropDb": tail_drop_db,
            "endingTailSlopeDbPerSecond": slope,
            "terminalSilenceSeconds": terminal_silence_seconds,
            "endingAbruptness01": abruptness,
            "trajectoryBinSecondsApprox": seconds_per_bin,
        },
        "normalized": measured_normalized,
    }


def round_value(value: Any, digits: int = 6) -> Any:
    if isinstance(value, np.ndarray):
        return [round(float(item), digits) for item in value]
    if isinstance(value, (np.floating, float)):
        return round(float(value), digits)
    if isinstance(value, dict):
        return {key: round_value(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_value(item, digits) for item in value]
    return value


def fit_candidate(features: dict[str, Any], target: dict[str, Any],
                  weights: dict[str, float]) -> dict[str, Any]:
    component_scores: dict[str, float] = {}
    weighted_error = 0.0
    total_weight = 0.0
    for name in TRAJECTORY_FEATURES:
        measured = np.asarray(features["trajectory"][name])
        intended = np.asarray(target["proxies"][name])
        error = float(np.mean(np.abs(measured - intended)))
        component_scores[name] = 100.0 * (1.0 - error)
        weighted_error += weights[name] * error
        total_weight += weights[name]
    for name in GLOBAL_FEATURES:
        error = abs(float(features["normalized"][name]) - target["globals"][name])
        component_scores[name] = 100.0 * (1.0 - error)
        weighted_error += weights[name] * error
        total_weight += weights[name]
    for name in ENDING_FEATURES:
        error = abs(float(features["normalized"][name]) - target["ending"][name])
        component_scores[name] = 100.0 * (1.0 - error)
        weighted_error += weights[name] * error
        total_weight += weights[name]
    score = 100.0 * (1.0 - weighted_error / total_weight)
    return {"technicalFitScore": float(np.clip(score, 0, 100)),
            "componentScores": component_scores}


def diversity_vector(candidate: dict[str, Any]) -> np.ndarray:
    features = candidate["features"]
    trajectory = features["trajectory"]
    vector = [
        features["normalized"]["dynamicRange"],
        float(np.mean(trajectory["lowFrequency"])),
        float(np.mean(trajectory["spectralBrightness"])),
        float(np.mean(trajectory["spectralSpread"])),
        float(np.mean(trajectory["onsetActivity"])),
        features["normalized"]["tempoConfidence"],
        float(np.clip(features["summary"]["silenceFraction"] / 0.25, 0, 1)),
        features["normalized"]["breathing"],
        features["normalized"]["sectionContrast"],
        features["normalized"]["tailDrop"],
        features["normalized"]["abruptness"],
    ]
    return np.asarray(vector, dtype=np.float64)


def trajectory_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    similarities = []
    for name in TRAJECTORY_FEATURES:
        a = np.asarray(left["features"]["trajectory"][name], dtype=np.float64)
        b = np.asarray(right["features"]["trajectory"][name], dtype=np.float64)
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            similarity = 1.0 - float(np.mean(np.abs(a - b)))
        else:
            similarity = (float(np.corrcoef(a, b)[0, 1]) + 1.0) / 2.0
        similarities.append(float(np.clip(similarity, 0, 1)))
    return float(np.mean(similarities))


def apply_diversity(candidates: list[dict[str, Any]],
                    thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    global_distance_threshold = float(thresholds.get("globalDistance", 0.14))
    trajectory_threshold = float(thresholds.get("trajectorySimilarity", 0.93))
    diversity_scale = float(thresholds.get("diversityScale", 0.42))
    vectors = [diversity_vector(candidate) for candidate in candidates]
    flags: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        cross_distances = []
        candidate["nearestDifferentMovement"] = None
        for other_index, other in enumerate(candidates):
            if other_index <= index:
                continue
            distance = float(np.sqrt(np.mean((vectors[index] - vectors[other_index]) ** 2)))
            similarity = trajectory_similarity(candidate, other)
            kind = "same-movement" if candidate["cueId"] == other["cueId"] else "cross-movement"
            if distance < global_distance_threshold or (
                distance < global_distance_threshold * 1.5
                and similarity >= trajectory_threshold
            ):
                flags.append({
                    "left": candidate["id"], "right": other["id"], "kind": kind,
                    "globalFeatureDistance": distance,
                    "trajectorySimilarity": similarity,
                    "reason": "acoustically similar measurable profile; audition for homogeneity",
                })
        for other_index, other in enumerate(candidates):
            if index == other_index or candidate["cueId"] == other["cueId"]:
                continue
            distance = float(np.sqrt(np.mean((vectors[index] - vectors[other_index]) ** 2)))
            cross_distances.append((distance, other["id"]))
        if cross_distances:
            nearest_distance, nearest_id = min(cross_distances)
            candidate["nearestDifferentMovement"] = {
                "candidateId": nearest_id, "globalFeatureDistance": nearest_distance,
            }
            candidate["diversityScore"] = float(
                100.0 * np.clip(nearest_distance / diversity_scale, 0, 1)
            )
            candidate["advisoryCompositeScore"] = (
                0.85 * candidate["technicalFitScore"]
                + 0.15 * candidate["diversityScore"]
            )
        else:
            candidate["diversityScore"] = None
            candidate["advisoryCompositeScore"] = candidate["technicalFitScore"]
    return flags


def inventory_result(plan: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    provisional = plan["anchors"].get("absoluteTimes") is None
    return {
        "version": ANALYZER_VERSION,
        "operation": "DRY_RUN",
        "status": "CANDIDATES_READY" if not inventory["missingMovements"] else "MISSING_CANDIDATES",
        "networkCalls": 0,
        "writes": 0,
        "advisoryOnly": True,
        "timingProvisional": provisional,
        "timingBasis": (
            "Scene-normalized authored targets; re-run after measured final-edit timing exists."
            if provisional else "Measured absolute semantic timing supplied."
        ),
        "candidateDirectory": str(plan["candidateDirectory"]),
        "reportPath": str(plan["reportPath"]),
        "candidatesFound": len(inventory["found"]),
        "countByMovement": inventory["countByMovement"],
        "missingMovements": inventory["missingMovements"],
        "partialMovements": inventory["partialMovements"],
        "orphanMediaIgnored": inventory["orphans"],
        "bindings": {
            "analyzerSource": str(Path(__file__).resolve()),
            "analyzerSourceSha256": sha_file(Path(__file__).resolve()),
            "analysisManifest": str(plan["manifestPath"]),
            "analysisManifestSha256": plan["manifestHash"],
            "creativeDraft": str(plan["creativePath"]),
            "creativeDraftSha256": plan["creativeHash"],
            "semanticAnchors": str(plan["anchorsPath"]),
            "semanticAnchorsSha256": plan["anchorsHash"],
            "executionManifest": str(plan["executionPath"]),
            "executionManifestSha256": plan["executionHash"],
        },
        "candidateBindings": inventory["found"],
        "message": (
            "No candidates are present yet; this is expected before the finite Suno lane returns."
            if not inventory["found"] else
            "Candidate provenance is valid; use --analyze for local feature extraction."
        ),
    }


def analyze(plan: dict[str, Any], inventory: dict[str, Any],
            audition: bool = False) -> dict[str, Any]:
    if not inventory["found"]:
        result = inventory_result(plan, inventory)
        result["operation"] = "ANALYZE"
        result["status"] = "NO_CANDIDATES"
        result["reportWritten"] = False
        return result
    sampling = plan["sampling"]
    cue_by_id = {cue["id"]: cue for cue in plan["cues"]}
    analyzed: list[dict[str, Any]] = []
    targets: dict[str, dict[str, Any]] = {}
    for cue in plan["cues"]:
        targets[cue["id"]] = semantic_target(plan, cue)
    for binding in inventory["found"]:
        audio = (decode_audio(Path(binding["file"]), int(sampling["sampleRate"]), audition=True)
                 if audition else decode_audio(Path(binding["file"]), int(sampling["sampleRate"])))
        features = extract_features(
            audio, int(sampling["sampleRate"]), int(sampling["frameSize"]),
            int(sampling["hopSize"]), int(sampling["trajectoryBins"]),
            int(sampling["sectionBins"]), float(sampling["lowFrequencyHz"]),
        )
        fit = fit_candidate(features, targets[binding["cueId"]], plan["scoreWeights"])
        requested_duration = float(
            {item["id"]: item for item in plan["materializedCues"]}
            [binding["cueId"]]["payload"]["duration"]
        )
        analyzed.append({
            **binding,
            "requestedDurationSeconds": requested_duration,
            "durationRatio": features["durationSeconds"] / requested_duration,
            "durationFlag": (
                "SHORTER_THAN_85_PERCENT_OF_REQUEST"
                if features["durationSeconds"] < 0.85 * requested_duration else "OK"
            ),
            "features": features,
            **fit,
            "selectionVerdict": "unreviewed; technical audition aid only",
        })
    thresholds = plan["manifest"].get("homogeneityThresholds") or {}
    homogeneity_flags = apply_diversity(analyzed, thresholds)

    rankings: dict[str, Any] = {}
    for cue_id in plan["cueIds"]:
        movement = [candidate for candidate in analyzed if candidate["cueId"] == cue_id]
        movement.sort(key=lambda candidate: (-candidate["advisoryCompositeScore"],
                                              candidate["take"]))
        previous_score: float | None = None
        previous_rank = 0
        for index, candidate in enumerate(movement, 1):
            score = candidate["advisoryCompositeScore"]
            rank = previous_rank if previous_score is not None and abs(score - previous_score) < 1e-9 else index
            candidate["advisoryRankWithinMovement"] = rank
            previous_score, previous_rank = score, rank
        ordered = [
            {
                "rank": candidate["advisoryRankWithinMovement"],
                "candidateId": candidate["id"],
                "technicalFitScore": candidate["technicalFitScore"],
                "diversityScore": candidate["diversityScore"],
                "advisoryCompositeScore": candidate["advisoryCompositeScore"],
                "selectionVerdict": candidate["selectionVerdict"],
            }
            for candidate in movement
        ]
        if len(movement) < 2:
            lead = None
            confidence = "indeterminate"
            rationale = "Fewer than two candidates; no comparative structural lead."
        else:
            delta = movement[0]["advisoryCompositeScore"] - movement[1]["advisoryCompositeScore"]
            if delta < 3.0:
                lead = None
                confidence = "indeterminate"
                rationale = f"Top-two technical delta is only {delta:.3f} points."
            else:
                lead = movement[0]["id"]
                confidence = "directional" if delta <= 6.0 else "strong_structural"
                rationale = (
                    f"Technical/diversity lead by {delta:.3f} points; listening verdict still required."
                )
        rankings[cue_id] = {
            "orderedCandidates": ordered,
            "structuralLead": lead,
            "confidence": confidence,
            "rationale": rationale,
            "humanAuditionRequired": True,
        }

    target_report = {}
    for cue_id, target in targets.items():
        target_report[cue_id] = {
            "scenePositions": target["scenePositions"],
            "authoredSemanticWave": target["semantics"],
            "technicalProxyTrajectories": target["proxies"],
            "globalTechnicalTargets": target["globals"],
            "endingTechnicalTargets": target["ending"],
        }
    report = {
        "version": ANALYZER_VERSION,
        "episode": plan["manifest"].get("episode"),
        "status": (
            "INCOMPLETE" if inventory["missingMovements"] else
            "PROVISIONAL_TECHNICAL_ANALYSIS_COMPLETE"
            if plan["anchors"].get("absoluteTimes") is None else
            "TECHNICAL_ANALYSIS_COMPLETE"
        ),
        "generatedBy": "pipeline/music_candidate_ranker.py",
        "advisory": {
            "scope": "Ranks measurable technical fit and cross-movement acoustic diversity.",
            "limitations": [
                "No waveform feature proves an emotion, story meaning, originality, Indian flavor, or musical quality.",
                "The eight authored semantic dimensions are converted to transparent editing proxies; they are not inferred back from audio.",
                "A high score cannot approve a take. Internal creative audition and the reviewer's listening verdict remain decisive.",
                "Tempo confidence measures periodic recurrence, not whether a rhythm is aesthetically right.",
            ],
        },
        "networkCalls": 0,
        "timingProvisional": plan["anchors"].get("absoluteTimes") is None,
        "timingBasis": plan["anchors"].get("timingRule"),
        "hardTransitionPolicy": {
            "scenes": plan["hardTransitions"],
            "rule": "A transition scene belongs to the new/right partition; edge values extend only from an anchor within that partition.",
        },
        "bindings": inventory_result(plan, inventory)["bindings"],
        "sampling": sampling,
        "normalization": {
            "rmsEnergy": "fixed physical RMS mapping from -60..-6 dBFS; no percentile stretch or compression",
            "lowFrequency": f"power below {sampling['lowFrequencyHz']} Hz divided by 0.55",
            "spectralBrightness": "log spectral centroid mapped from 150..6000 Hz",
            "spectralSpread": "spectral standard deviation divided by 5000 Hz",
            "onsetActivity": "detected onsets per second divided by 4",
            "dynamicRange": "RMS p95-p20 divided by 30 dB",
            "breathing": "fraction in >=250ms relative low-energy runs divided by 0.5",
            "sectionContrast": "adjacent section-vector distance divided by 0.45",
            "tailDrop": "positive last-tail drop divided by 30 dB",
            "terminalSilence": "contiguous terminal silence divided by 3 seconds",
        },
        "missingMovements": inventory["missingMovements"],
        "partialMovements": inventory["partialMovements"],
        "orphanMediaIgnored": inventory["orphans"],
        "targets": target_report,
        "candidates": analyzed,
        "rankings": rankings,
        "homogeneityFlags": homogeneity_flags,
    }
    report_path = plan["reportPath"]
    if audition:
        report_path = audition_report_path(report_path)
        report["auditionRecord"] = audition_record()
    rounded = round_value(report)
    atomic_json(report_path, rounded)
    result = {
        "version": ANALYZER_VERSION,
        "operation": "ANALYZE",
        "status": rounded["status"],
        "networkCalls": 0,
        "candidatesAnalyzed": len(analyzed),
        "missingMovements": inventory["missingMovements"],
        "homogeneityFlags": len(homogeneity_flags),
        "reportWritten": True,
        "report": str(report_path),
        "reportSha256": sha_file(report_path),
        "advisoryOnly": True,
    }
    if audition:
        result["audition"] = True
    return result


def execute(manifest_path: Path, do_analyze: bool = False,
            audition: bool = False) -> dict[str, Any]:
    plan = load_plan(manifest_path)
    inventory = discover_candidates(plan)
    if audition and not do_analyze:
        raise AnalysisError("--audition applies to --analyze")
    if not do_analyze:
        return inventory_result(plan, inventory)
    return analyze(plan, inventory, audition=True) if audition else analyze(plan, inventory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--analyze", action="store_true",
                        help="decode candidates and atomically write the advisory report")
    parser.add_argument("--audition", action="store_true",
                        help="with --analyze: decode on this host; the report is labelled an inadmissible audition")
    args = parser.parse_args(argv)
    try:
        result = execute(args.manifest, do_analyze=args.analyze, audition=args.audition)
    except AnalysisError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(round_value(result), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
