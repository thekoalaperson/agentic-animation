#!/usr/bin/env python3
"""Validate and execute the Stable-Audio absolute-SFX long-form mix.

The default operation is a read-only preflight. ``--write-contract`` records a
hash-bound mixer contract below the episode ``work/`` tree. ``--execute`` is a
separate explicit local action and accepts only that unchanged contract. The
script never writes to ``deliverables/``, never invokes a provider, never uses
``-shortest`` and never treats technical success as creative approval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np


SR = 48_000
FPS = 24
CHANNELS = 2
SHA = re.compile(r"[0-9a-f]{64}\Z")
MIXER_VERSION = "stable-audio-absolute-longform-mixer-v2"


class MixError(RuntimeError):
    """A mixer input or execution artifact is incomplete, stale or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MixError(f"Cannot read {label}: {path}: {exc}") from exc


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise MixError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MixError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise MixError(f"{label} must be finite")
    return number


def episode_path(
    episode: Path, raw: str | Path, label: str, *, must_exist: bool = True
) -> Path:
    value = Path(raw)
    path = value.resolve() if value.is_absolute() else (episode / value).resolve()
    try:
        relative = path.relative_to(episode.resolve())
    except ValueError as exc:
        raise MixError(f"{label} escapes episode root: {path}") from exc
    if "deliverables" in relative.parts:
        raise MixError(f"{label} may not use deliverables/: {path}")
    if must_exist and not path.is_file():
        raise MixError(f"{label} is missing: {path}")
    return path


def work_output(episode: Path, raw: str | Path, label: str) -> Path:
    path = episode_path(episode, raw, label, must_exist=False)
    try:
        path.relative_to((episode / "work").resolve())
    except ValueError as exc:
        raise MixError(f"{label} must remain below episode work/: {path}") from exc
    return path


def portable(path: Path, episode: Path) -> str:
    return path.resolve().relative_to(episode.resolve()).as_posix()


def verify_hash(path: Path, expected: Any, label: str) -> str:
    declared = str(expected or "").lower()
    if not SHA.fullmatch(declared):
        raise MixError(f"{label} has no valid SHA-256")
    actual = sha256_file(path)
    if actual != declared:
        raise MixError(f"{label} changed: expected {declared}, got {actual}: {path}")
    return actual


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(command: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MixError(f"Required executable is unavailable: {command[0]}") from exc
    if result.returncode:
        tail = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise MixError(f"Command failed ({result.returncode}): {' '.join(command)}\n{tail}")
    return result


def probe_media(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    raw = json.loads(result.stdout.decode("utf-8"))
    streams = raw.get("streams", [])
    audio = [row for row in streams if row.get("codec_type") == "audio"]
    video = [row for row in streams if row.get("codec_type") == "video"]
    duration = finite(raw.get("format", {}).get("duration"), f"{path} duration")
    normalized: dict[str, Any] = {"durationSeconds": duration, "audio": [], "video": []}
    for row in audio:
        normalized["audio"].append(
            {
                "codec": row.get("codec_name"),
                "sampleRate": int(row.get("sample_rate", 0)),
                "channels": int(row.get("channels", 0)),
                "durationSeconds": finite(row.get("duration", duration), "audio duration"),
            }
        )
    for row in video:
        frames = row.get("nb_read_frames", row.get("nb_frames"))
        normalized["video"].append(
            {
                "codec": row.get("codec_name"),
                "width": int(row.get("width", 0)),
                "height": int(row.get("height", 0)),
                "frameRate": str(row.get("r_frame_rate", "0/1")),
                "frames": int(frames) if str(frames or "").isdigit() else None,
                "durationSeconds": finite(row.get("duration", duration), "video duration"),
            }
        )
    return normalized


def verify_audio_probe(
    info: dict[str, Any], runtime: float, expected_frames: int, label: str, *, codec: str | None = None
) -> dict[str, Any]:
    if info.get("video") or len(info.get("audio", [])) != 1:
        raise MixError(f"{label} must contain exactly one audio stream and no video")
    row = info["audio"][0]
    if row.get("sampleRate") != SR or row.get("channels") != CHANNELS:
        raise MixError(f"{label} must be 48 kHz stereo")
    if codec is not None and row.get("codec") != codec:
        raise MixError(f"{label} codec must be {codec}, got {row.get('codec')}")
    duration = finite(row.get("durationSeconds"), f"{label} duration")
    if abs(duration * SR - expected_frames) > 1.01:
        raise MixError(f"{label} does not span the measured timeline exactly")
    if abs(finite(info.get("durationSeconds"), f"{label} format duration") - runtime) > 1 / SR:
        raise MixError(f"{label} container duration differs from the measured timeline")
    return row


def verify_picture_probe(
    info: dict[str, Any], runtime: float, expected_frames: int, height: int
) -> dict[str, Any]:
    if info.get("audio") or len(info.get("video", [])) != 1:
        raise MixError("silent picture must contain one video stream and no audio")
    row = info["video"][0]
    expected_width = round(height * 16 / 9 / 2) * 2
    try:
        rate = Fraction(str(row.get("frameRate", "0/1")))
    except (ValueError, ZeroDivisionError) as exc:
        raise MixError("picture frame rate is malformed") from exc
    if (
        row.get("width") != expected_width
        or row.get("height") != height
        or rate != Fraction(FPS)
        or row.get("frames") != expected_frames
    ):
        raise MixError("picture geometry, cadence or exact frame count is wrong")
    if abs(finite(info.get("durationSeconds"), "picture duration") - runtime) > 1 / FPS:
        raise MixError("picture duration differs from the measured timeline")
    return row


def load_sfx_adapter() -> Any:
    path = Path(__file__).resolve().with_name("validate_absolute_sfx_mix_input.py")
    spec = importlib.util.spec_from_file_location("absolute_sfx_mix_revalidator", path)
    if not spec or not spec.loader:
        raise MixError(f"Cannot load SFX input validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def revalidate_sfx_contract(episode: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    evidence = loaded.get("inputEvidence")
    if not isinstance(evidence, dict):
        raise MixError("SFX mix-input has no inputEvidence")
    if loaded.get("schemaVersion") != 2:
        raise MixError("Mixing requires the source-bound designed-SFX v2 handoff")
    needed = ("absoluteDesign", "renderAudit", "editPlan", "pictureSegments")
    if "contextReview" in evidence:
        needed += ("contextReview",)
    paths: dict[str, Path] = {}
    for key in needed:
        row = evidence.get(key)
        if not isinstance(row, dict):
            raise MixError(f"SFX mix-input lacks {key} evidence")
        paths[key] = episode_path(episode, row.get("path", ""), f"SFX {key}")
        verify_hash(paths[key], row.get("sha256"), f"SFX {key}")
    adapter = load_sfx_adapter()
    try:
        current = adapter.build_contract(
            episode,
            paths["absoluteDesign"],
            paths["renderAudit"],
            paths["editPlan"],
            paths["pictureSegments"],
            context_review_path=paths.get("contextReview"),
        )
    except adapter.ContractError as exc:
        raise MixError(f"SFX final input no longer validates: {exc}") from exc
    if loaded != current:
        raise MixError("SFX mix-input JSON differs from the freshly rebuilt contract")
    return current


def validate_events(plan: dict[str, Any], runtime: float) -> None:
    events = plan.get("events")
    if not isinstance(events, list) or not events:
        raise MixError("edit plan must contain its complete measured event list")
    cursor = 0.0
    ids, shots = set(), set()
    for event in events:
        if (not isinstance(event, dict) or not isinstance(event.get("id"), str) or not event["id"]
                or event["id"] in ids or isinstance(event.get("shot"), bool)
                or not isinstance(event.get("shot"), (int, str)) or event["shot"] in shots
                or not isinstance(event.get("kind"), str) or not event["kind"]):
            raise MixError("edit-plan event identities are missing or duplicated")
        ids.add(event["id"])
        shots.add(event["shot"])
        start = finite(event.get("start"), "event start")
        end = finite(event.get("end"), "event end")
        duration = finite(event.get("duration"), "event duration")
        if abs(start - cursor) > 1e-5 or end <= start or abs(duration - (end - start)) > 1e-5:
            raise MixError(f"edit-plan event clock is not contiguous at {event.get('id')}")
        cursor = end
    if abs(cursor - runtime) > 1e-5:
        raise MixError("edit-plan events do not end at runtime")


def silent_placement(row: Any) -> bool:
    """An authored stretch of no score: it names no track and binds no media."""
    return isinstance(row, dict) and row.get("silence") is True


def validate_score_placements(episode: Path, plan: dict[str, Any], runtime: float) -> None:
    """Placements are ordered, never overlap and stay inside the measured clock.

    A gap between placements is silence, so a film may score only some of its
    events. An empty list is accepted only from a plan that declares itself
    unscored, so a forgotten score is still caught. Every placement that names
    a track keeps its full hash binding.
    """
    rows = plan.get("scorePlacements")
    if not isinstance(rows, list):
        raise MixError("the source-bound score placement list is required")
    unscored = plan.get("unscored", False)
    if type(unscored) is not bool:
        raise MixError("unscored must be a boolean")
    if not rows and not unscored:
        raise MixError("the source-bound score placement list is required; "
                       "a film with no score declares unscored: true")
    if unscored and any(not silent_placement(row) for row in rows):
        raise MixError("an unscored film may carry only silence placements")
    cursor = 0.0
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise MixError("score placements must be objects")
        movement = str(row.get("movement", ""))
        if not movement or movement in seen:
            raise MixError("score movement names are missing or duplicated")
        seen.add(movement)
        start = finite(row.get("timelineStart"), f"{movement} timelineStart")
        duration = finite(row.get("timelineDuration"), f"{movement} timelineDuration")
        if start < cursor - 1e-5 or duration <= 0:
            raise MixError(f"score placements must be ordered and must not overlap: {movement}")
        cursor = start + duration
        if cursor > runtime + 1e-5:
            raise MixError(f"score placement {movement} runs past the measured runtime")
        if silent_placement(row):
            if any(key in row for key in ("track", "trackSha256", "sidecar", "sidecarSha256")):
                raise MixError(f"silence placement {movement} must not name a score track")
            continue
        for path_key, sha_key in (
            ("track", "trackSha256"),
            ("sidecar", "sidecarSha256"),
            ("reviewExport", "reviewExportSha256"),
            ("executionManifest", "executionManifestSha256"),
        ):
            if path_key == "reviewExport" and path_key not in row and sha_key not in row:
                continue
            path = episode_path(episode, row.get(path_key, ""), f"{movement} {path_key}")
            verify_hash(path, row.get(sha_key), f"{movement} {path_key}")


def validate_render_runs(
    picture_segments: list[dict[str, Any]], render_runs: list[dict[str, Any]], expected_frames: int
) -> None:
    expected_ids = [str(row.get("id", "")) for row in picture_segments]
    members = [
        str(member)
        for run in render_runs
        for member in run.get("memberSegmentIds", [])
    ]
    if not expected_ids or members != expected_ids or len(set(members)) != len(members):
        raise MixError("picture render runs do not preserve every semantic segment exactly once")
    frames = sum(int(run.get("frames", -1)) for run in render_runs)
    if frames != expected_frames:
        raise MixError("picture render runs do not cover the exact delivery frame count")
    for run in render_runs:
        member_ids = run.get("memberSegmentIds", [])
        if run.get("role") == "dissolve-transition-run":
            # Two or more sources overlapped by authored dissolves: the sources'
            # frames, less each overlap, must be exactly the run's frames.
            sources, dissolves = run.get("sources"), run.get("dissolves")
            if (not isinstance(sources, list) or len(sources) < 2 or not isinstance(dissolves, list)
                    or len(dissolves) != len(sources) - 1
                    or sum(int(row.get("frames", -1)) for row in sources) - sum(dissolves) != int(run.get("frames", -1))
                    or sum(int(row.get("shownFrames", -1)) for row in sources) != int(run.get("frames", -1))):
                raise MixError("dissolve picture run does not account for its sources' frames")
            if len(run.get("memberRoles", [])) != len(member_ids):
                raise MixError("coalesced picture run lost semantic role evidence")
        elif len(member_ids) > 1:
            if run.get("role") != "source-contiguous-role-preserving-render-run":
                raise MixError("coalesced picture run lacks its continuity declaration")
            if len(run.get("memberRoles", [])) != len(member_ids):
                raise MixError("coalesced picture run lost semantic role evidence")


def validate_assembly_evidence(
    episode: Path,
    plan: dict[str, Any],
    runtime: float,
    expected_frames: int,
    paths: dict[str, Path],
) -> None:
    levels = read_json(paths["voiceLevels"], "voice levels")
    if not isinstance(levels, list) or [row.get("event") for row in levels] != [
        event["id"] for event in plan["events"]
    ]:
        raise MixError("voice-level audit does not cover the measured events in order")
    conform = read_json(paths["musicConform"], "music conform")
    placements = plan["scorePlacements"]
    if not isinstance(conform, list) or len(conform) != len(placements):
        raise MixError("music-conform audit does not cover all score movements")
    for decision, placement in zip(conform, placements):
        if silent_placement(placement):
            if (decision.get("movement") != placement["movement"] or decision.get("silence") is not True
                    or abs(finite(decision.get("timelineDuration"), "conform timelineDuration")
                           - finite(placement.get("timelineDuration"), "placement timelineDuration")) > 1e-5):
                raise MixError("music-conform audit differs from the authored silence placements")
            continue
        source = episode_path(episode, placement["track"], "score track")
        if (
            decision.get("movement") != placement["movement"]
            or decision.get("take") != placement["take"]
            or abs(
                finite(decision.get("timelineDuration"), "conform timelineDuration")
                - finite(placement.get("timelineDuration"), "placement timelineDuration")
            )
            > 1e-5
            or episode_path(episode, str(decision.get("source", "")), "conformed score source") != source
        ):
            raise MixError("music-conform audit differs from the selected score placements")
    wave_audit = read_json(paths["musicWaveAudit"], "music wave audit")
    targets = wave_audit.get("eventTargets") if isinstance(wave_audit, dict) else None
    if not isinstance(targets, list) or [row.get("shot") for row in targets] != [event["shot"] for event in plan["events"]]:
        raise MixError("music-wave audit does not cover every measured event in order")
    wave = read_json(paths["dramaticWave"], "dramatic wave")
    if (
        not isinstance(wave, dict)
        or wave.get("clockSha256") != sha256_file(paths["editPlan"])
    ):
        raise MixError("dramatic wave does not bind the measured edit")
    shots = wave.get("shots")
    if not isinstance(shots, list) or [row.get("shot") for row in shots] != [
        event["shot"] for event in plan["events"]
    ]:
        raise MixError("dramatic wave does not cover the measured events in order")
    segments = read_json(paths["pictureSegments"], "picture segments")
    runs = read_json(paths["pictureRenderRuns"], "picture render runs")
    if not isinstance(segments, list) or not isinstance(runs, list):
        raise MixError("picture segment/run evidence must be arrays")
    validate_render_runs(segments, runs, expected_frames)


def validate_notches(rows: Any, runtime: float) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise MixError("scoreNotchAutomation must be an array")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MixError(f"score notch {index} is not an object")
        cue = str(row.get("cueId", ""))
        if not cue or cue in ids:
            raise MixError("score notch cue IDs are missing or duplicated")
        ids.add(cue)
        start = finite(row.get("startSeconds"), f"{cue} notch start")
        end = finite(row.get("endSeconds"), f"{cue} notch end")
        gain = finite(row.get("scoreGainDb"), f"{cue} notch gain")
        attack = finite(row.get("attackSeconds"), f"{cue} notch attack")
        release = finite(row.get("releaseSeconds"), f"{cue} notch release")
        if not (0 <= start < end <= runtime) or not (-18 <= gain < 0):
            raise MixError(f"{cue} score notch has unsafe time or gain")
        if attack <= 0 or release <= 0 or max(attack, release) > end - start + 1e-6:
            raise MixError(f"{cue} score notch has unsafe attack/release")
        result.append(
            {
                **row,
                "startSeconds": start,
                "endSeconds": end,
                "scoreGainDb": gain,
                "attackSeconds": attack,
                "releaseSeconds": release,
            }
        )
    return result


def notch_volume_expression(rows: list[dict[str, Any]]) -> str:
    expressions = []
    for row in rows:
        start = row["startSeconds"]
        end = row["endSeconds"]
        attack = row["attackSeconds"]
        release = row["releaseSeconds"]
        shape = (
            f"max(0,min(1,min((t-{start:.9f})/{attack:.9f},"
            f"({end:.9f}-t)/{release:.9f})))"
        )
        expressions.append(
            f"if(between(t,{start:.9f},{end:.9f}),"
            f"{row['scoreGainDb']:.9f}*{shape},0)"
        )
    db = "0" if not expressions else "min(0," + ",".join(expressions) + ")"
    # ``min`` is binary in FFmpeg expressions; nest it for more than one row.
    if len(expressions) > 1:
        combined = expressions[0]
        for expression in expressions[1:]:
            combined = f"min({combined},{expression})"
        db = f"min(0,{combined})"
    return f"pow(10,({db})/20)"


def default_settings() -> dict[str, Any]:
    return {
        "musicGainDb": 0.0,
        "musicDuck": {"threshold": 0.018, "ratio": 9.0, "attackMs": 12, "releaseMs": 520},
        "sfxDuck": {"threshold": 0.018, "ratio": 7.0, "attackMs": 8, "releaseMs": 360},
        "targetIntegratedLufs": -14.0,
        "targetTruePeakDbtp": -1.5,
        "targetLra": 11.0,
    }


def check_bus_settings(settings: dict[str, Any], *, exact: bool = True) -> None:
    """The score gain and sidechain bounds every mix in this repository honours."""
    gain = finite(settings.get("musicGainDb"), "musicGainDb")
    if not -6 <= gain <= 6:
        raise MixError("musicGainDb must lie within -6..6 dB")
    names = {"threshold", "ratio", "attackMs", "releaseMs"}
    for key in ("musicDuck", "sfxDuck"):
        duck = settings.get(key)
        if not isinstance(duck, dict) or not names <= set(duck) or (exact and set(duck) != names):
            raise MixError(f"{key} needs threshold, ratio, attackMs and releaseMs")
        values = {name: finite(value, f"{key} {name}") for name, value in duck.items()}
        if (not .005 <= values["threshold"] <= .1 or not 1 <= values["ratio"] <= 12
                or not 1 <= values["attackMs"] <= 100 or not 100 <= values["releaseMs"] <= 1500):
            raise MixError(f"{key} is outside threshold .005-.1, ratio 1-12, attack 1-100 ms, release 100-1500 ms")


def validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Version 1 is exactly ``default_settings()``; version 2 is authored within bounds.

    A version 2 file names ``settingsVersion: 2`` and only the values it changes.
    Every other value keeps today's default, and the contract records the whole
    effective set, so its fingerprint binds every chosen value.
    """
    if settings == default_settings():
        return settings
    if not isinstance(settings, dict) or settings.get("settingsVersion") != 2:
        raise MixError(
            "Mixer settings differ from the reviewed v1 contract; author settingsVersion 2 instead"
        )
    unknown = set(settings) - set(default_settings()) - {"settingsVersion", "refuseDynamicMastering", "reason"}
    if unknown:
        raise MixError(f"unknown mixer settings: {', '.join(sorted(unknown))}")
    result = {**default_settings(), "refuseDynamicMastering": False, **copy.deepcopy(settings)}
    check_bus_settings(result)
    if not -27 <= finite(result["targetIntegratedLufs"], "targetIntegratedLufs") <= -12:
        raise MixError("targetIntegratedLufs must lie within -27..-12")
    if not -3 <= finite(result["targetTruePeakDbtp"], "targetTruePeakDbtp") <= -1:
        raise MixError("targetTruePeakDbtp must lie within -3..-1")
    lra = result["targetLra"]
    if lra != "measured" and (isinstance(lra, (bool, str)) or not 1 <= finite(lra, "targetLra") <= 50):
        raise MixError('targetLra must be a number within 1..50 or "measured"')
    if type(result["refuseDynamicMastering"]) is not bool:
        raise MixError("refuseDynamicMastering must be a boolean")
    if "reason" in result and (not isinstance(result["reason"], str) or not result["reason"].strip()):
        raise MixError("a settings reason must be text")
    return result


def measurement_lra(settings: dict[str, Any]) -> float:
    """A first pass only measures; its LRA target does not change what it reports."""
    return 11.0 if settings["targetLra"] == "measured" else settings["targetLra"]


def second_pass_lra(settings: dict[str, Any], first_pass: dict[str, Any]) -> float:
    """``measured`` asks loudnorm for the film's own range, within loudnorm's 1..50."""
    if settings["targetLra"] != "measured":
        return settings["targetLra"]
    return min(50.0, max(1.0, finite(first_pass.get("input_lra"), "measured input LRA")))


def build_mix_contract(
    episode: Path,
    *,
    sfx_input_path: Path,
    edit_plan_path: Path,
    picture_path: Path,
    voice_path: Path,
    music_path: Path,
    height: int = 720,
    probe: Callable[[Path], dict[str, Any]] = probe_media,
    sfx_revalidator: Callable[[Path, dict[str, Any]], dict[str, Any]] = revalidate_sfx_contract,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    episode = episode.resolve()
    if height != 720:
        raise MixError("This body-mix contract is intentionally limited to the reviewed 720p stage")
    assembly = episode_path(episode, edit_plan_path, "edit plan").parent
    paths = {
        "sfxInput": episode_path(episode, sfx_input_path, "SFX final-mix input"),
        "editPlan": episode_path(episode, edit_plan_path, "edit plan"),
        "picture": episode_path(episode, picture_path, "silent picture"),
        "voice": episode_path(episode, voice_path, "voice stem"),
        "music": episode_path(episode, music_path, "selected score stem"),
        "voiceLevels": episode_path(episode, assembly / "voice-levels.json", "voice-level audit"),
        "musicConform": episode_path(episode, assembly / "music-conform.json", "music-conform audit"),
        "musicWaveAudit": episode_path(episode, assembly / "music-wave-audit.json", "music-wave audit"),
        "pictureRenderRuns": episode_path(episode, assembly / "picture-render-runs.json", "picture render runs"),
    }
    loaded_sfx = read_json(paths["sfxInput"], "SFX final-mix input")
    if not isinstance(loaded_sfx, dict):
        raise MixError("SFX final-mix input must be an object")
    sfx = sfx_revalidator(episode, loaded_sfx)
    if sfx.get("schemaVersion") != 2 or sfx.get("kind") != "stable-audio-absolute-sfx-final-mix-input":
        raise MixError("wrong SFX final-mix contract kind")
    for key in ("pictureSegments", "dramaticWave"):
        row = sfx.get("inputEvidence", {}).get(key, {})
        paths[key] = episode_path(episode, row.get("path", ""), f"SFX-bound {key}")
        verify_hash(paths[key], row.get("sha256"), f"SFX-bound {key}")
    reviewed_context = [sfx.get("inputEvidence", {}).get("reviewed" + key.title())
                        for key in ("picture", "voice", "music")]
    review_matches = (all(row.get("sha256") == sha256_file(paths[key])
                          for key, row in zip(("picture", "voice", "music"), reviewed_context))
                      if all(isinstance(row, dict) for row in reviewed_context) else None)
    plan = read_json(paths["editPlan"], "edit plan")
    if not isinstance(plan, dict) or plan.get("schemaVersion") != 2:
        raise MixError("edit plan must be the final schemaVersion 2 plan")
    if (not isinstance(plan.get("episode"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", plan["episode"])
            or not isinstance(plan.get("language"), str) or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]+)*", plan["language"])
            or plan["episode"] != sfx.get("episode") or plan["language"] != sfx.get("language")):
        raise MixError("SFX and edit plan must declare the same valid project identity and language")
    timeline = str(plan.get("timelineFingerprint", ""))
    if not SHA.fullmatch(timeline) or timeline != sfx.get("timelineFingerprint"):
        raise MixError("SFX and edit-plan timeline fingerprints differ")
    runtime = finite(plan.get("runtimeSeconds"), "edit-plan runtime")
    if runtime <= 0 or abs(runtime - finite(sfx.get("runtimeSeconds"), "SFX runtime")) > 1e-6:
        raise MixError("SFX and edit-plan runtimes differ")
    if verify_hash(
        paths["editPlan"],
        sfx.get("inputEvidence", {}).get("editPlan", {}).get("sha256"),
        "SFX-bound edit plan",
    ) != sha256_file(paths["editPlan"]):
        raise AssertionError("unreachable")
    validate_events(plan, runtime)
    validate_score_placements(episode, plan, runtime)
    expected_video_frames = round(runtime * FPS)
    expected_audio_frames = round(runtime * SR)
    validate_assembly_evidence(episode, plan, runtime, expected_video_frames, paths)
    verify_picture_probe(probe(paths["picture"]), runtime, expected_video_frames, height)
    verify_audio_probe(probe(paths["voice"]), runtime, expected_audio_frames, "voice stem")
    verify_audio_probe(probe(paths["music"]), runtime, expected_audio_frames, "selected score stem")
    stem = episode_path(episode, sfx.get("sfxStem", {}).get("path", ""), "SFX stem")
    verify_hash(stem, sfx.get("sfxStem", {}).get("sha256"), "SFX stem")
    verify_audio_probe(probe(stem), runtime, expected_audio_frames, "SFX stem", codec="pcm_f32le")
    notches = validate_notches(sfx.get("scoreNotchAutomation"), runtime)
    settings = validate_settings(default_settings() if settings is None else settings)
    evidence = {
        key: {"path": portable(path, episode), "sha256": sha256_file(path)}
        for key, path in paths.items()
    }
    evidence["sfxStem"] = {"path": portable(stem, episode), "sha256": sha256_file(stem)}
    contract: dict[str, Any] = {
        "schemaVersion": 2,
        "mixerVersion": MIXER_VERSION,
        "kind": "stable-audio-absolute-longform-mix-contract",
        "status": "validated-ready-for-explicit-local-execution-not-creatively-approved",
        "episode": plan["episode"],
        "language": plan["language"],
        "timelineFingerprint": timeline,
        "runtimeSeconds": runtime,
        "deliveryFps": FPS,
        "expectedVideoFrames": expected_video_frames,
        "sampleRate": SR,
        "channels": CHANNELS,
        "expectedAudioFrames": expected_audio_frames,
        "inputEvidence": evidence,
        "contextReview": sfx.get("contextReview"),
        "reviewContextMatchesMixInputs": review_matches,
        "scoreNotchAutomation": notches,
        "scoreNotchEnvelope": {
            "combination": "most-negative-active-gain-wins; never sum overlapping dips",
            "ffmpegVolumeExpression": notch_volume_expression(notches),
        },
        "settings": settings,
        "outputs": {
            "premaster": portable(assembly / f"mix-{plan['language']}-premaster-48kf32.wav", episode),
            "bodyMaster": portable(assembly / f"{plan['episode']}_{plan['language'].upper()}_720P_BODY.mp4", episode),
            "busAudit": portable(assembly / "absolute-mix-bus-audit.json", episode),
            "masterAudit": portable(assembly / "body-master-audit-720p.json", episode),
        },
        "safety": {
            "paidOrNetworkCalls": False,
            "writesDeliverables": False,
            "usesShortest": False,
            "pictureCodecCopyRequired": True,
            "picturePacketEqualityRequired": True,
            "creativeApprovalGrantedByThisContract": False,
        },
    }
    contract["contractFingerprint"] = canonical_sha(contract)
    return contract


def verify_saved_contract(
    episode: Path,
    contract_path: Path,
    *,
    probe: Callable[[Path], dict[str, Any]] = probe_media,
    sfx_revalidator: Callable[[Path, dict[str, Any]], dict[str, Any]] = revalidate_sfx_contract,
) -> dict[str, Any]:
    saved = read_json(contract_path, "saved mix contract")
    if not isinstance(saved, dict):
        raise MixError("saved mix contract must be an object")
    expected_fingerprint = saved.get("contractFingerprint")
    body = dict(saved)
    body.pop("contractFingerprint", None)
    if not SHA.fullmatch(str(expected_fingerprint or "")) or canonical_sha(body) != expected_fingerprint:
        raise MixError("saved mix contract fingerprint is stale")
    evidence = saved.get("inputEvidence", {})
    current = build_mix_contract(
        episode,
        sfx_input_path=episode_path(episode, evidence.get("sfxInput", {}).get("path", ""), "contract SFX input"),
        edit_plan_path=episode_path(episode, evidence.get("editPlan", {}).get("path", ""), "contract edit plan"),
        picture_path=episode_path(episode, evidence.get("picture", {}).get("path", ""), "contract picture"),
        voice_path=episode_path(episode, evidence.get("voice", {}).get("path", ""), "contract voice"),
        music_path=episode_path(episode, evidence.get("music", {}).get("path", ""), "contract music"),
        height=720,
        probe=probe,
        sfx_revalidator=sfx_revalidator,
        settings=saved.get("settings"),
    )
    if current != saved:
        raise MixError("saved mix contract differs from a fresh preflight")
    return current


def filter_graph(contract: dict[str, Any]) -> str:
    settings = contract["settings"]
    music = settings["musicDuck"]
    sfx = settings["sfxDuck"]
    volume = contract["scoreNotchEnvelope"]["ffmpegVolumeExpression"]
    return (
        "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,asplit=3[voice][mkey][skey];"
        "[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"volume={settings['musicGainDb']}dB,volume='{volume}':eval=frame[music];"
        "[2:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[sfx];"
        f"[music][mkey]sidechaincompress=threshold={music['threshold']}:ratio={music['ratio']}:"
        f"attack={music['attackMs']}:release={music['releaseMs']}[mduck];"
        f"[sfx][skey]sidechaincompress=threshold={sfx['threshold']}:ratio={sfx['ratio']}:"
        f"attack={sfx['attackMs']}:release={sfx['releaseMs']}[sduck];"
        "[voice][mduck][sduck]amix=inputs=3:normalize=0:dropout_transition=0[out]"
    )


def parse_loudnorm(stderr: bytes) -> dict[str, Any]:
    text = stderr.decode("utf-8", errors="replace")
    start, end = text.rfind("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise MixError("ffmpeg loudnorm did not return JSON measurements")
    try:
        result = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise MixError("ffmpeg loudnorm JSON is malformed") from exc
    return result


def loudness(path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    result = run(
        [
            "ffmpeg", "-hide_banner", "-i", str(path), "-vn", "-af",
            f"loudnorm=I={settings['targetIntegratedLufs']}:TP={settings['targetTruePeakDbtp']}:"
            f"LRA={measurement_lra(settings)}:print_format=json",
            "-f", "null", "-",
        ]
    )
    return parse_loudnorm(result.stderr)


def second_pass_record(measured: dict[str, Any], target_lra: float) -> dict[str, Any]:
    """What the second loudnorm pass actually did, read from its own JSON report."""
    kind = measured.get("normalization_type")
    if kind not in ("linear", "dynamic"):
        raise MixError("ffmpeg loudnorm did not report its normalization type")
    return {
        "normalizationType": kind,
        "targetLra": target_lra,
        "inputLra": finite(measured.get("input_lra"), "second-pass input LRA"),
        "outputLra": finite(measured.get("output_lra"), "second-pass output LRA"),
        "inputIntegratedLufs": finite(measured.get("input_i"), "second-pass input loudness"),
        "outputIntegratedLufs": finite(measured.get("output_i"), "second-pass output loudness"),
        "note": ("linear: one gain for the whole film, its dynamics kept" if kind == "linear" else
                 "dynamic: loudnorm could not meet the targets with one gain and rode the level, "
                 "which narrows the film's dynamic range"),
    }


def dynamic_mastering_message(record: dict[str, Any]) -> str:
    return (
        f"loudnorm fell back to dynamic gain (input LRA {record['inputLra']}, output LRA "
        f"{record['outputLra']}, target LRA {record['targetLra']}). Some ways to choose instead: "
        'targetLra "measured" or a higher number, a lower targetIntegratedLufs or targetTruePeakDbtp, '
        "the measured fixed gain with a qualified limiter in master_review_audio.py (fixedGainMaster), "
        "or leaving refuseDynamicMastering off and accepting the dynamic pass. Each changes the film's "
        "dynamics differently."
    )


def decode_samples(path: Path) -> np.ndarray:
    raw = run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
            "-ar", str(SR), "-ac", str(CHANNELS), "-f", "f32le", "pipe:1",
        ]
    ).stdout
    samples = np.frombuffer(raw, dtype="<f4")
    if len(samples) % 2:
        raise MixError(f"decoded audio has an incomplete stereo frame: {path}")
    return samples.reshape((-1, 2)).copy()


def intelligibility(voice: Path, premaster: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    voice_samples = decode_samples(voice)
    mixed = decode_samples(premaster)
    if len(voice_samples) != len(mixed):
        raise MixError("voice and premaster sample counts differ")
    residual = mixed - voice_samples
    frame = 4_800
    rows = []
    for event in events:
        start = round(finite(event["start"], "event start") * SR)
        end = min(len(mixed), round(finite(event["end"], "event end") * SR))
        count = (end - start) // frame
        if count <= 0:
            continue
        v = voice_samples[start : start + count * frame].reshape(count, frame, 2)
        r = residual[start : start + count * frame].reshape(count, frame, 2)
        vrms = np.sqrt(np.mean(v.astype(np.float64) ** 2, axis=(1, 2)))
        rrms = np.sqrt(np.mean(r.astype(np.float64) ** 2, axis=(1, 2)))
        voiced = vrms > 10 ** (-38 / 20)
        if np.any(voiced):
            ratios = 20 * np.log10(
                np.maximum(vrms[voiced], 1e-9) / np.maximum(rrms[voiced], 1e-9)
            )
            rows.append(
                {
                    "event": event["id"],
                    "kind": event["kind"],
                    "voicedWindows": int(np.sum(voiced)),
                    "medianVoiceToBackgroundDb": round(float(np.median(ratios)), 3),
                    "p10VoiceToBackgroundDb": round(float(np.percentile(ratios, 10)), 3),
                }
            )
    summary = {}
    for kind in sorted({event["kind"] for event in events} & {"narration", "dialogue"}):
        values = [row["medianVoiceToBackgroundDb"] for row in rows if row["kind"] == kind]
        if not values:
            raise MixError(f"no voiced {kind} windows exist for intelligibility audit")
        summary[kind] = {
            "events": len(values),
            "minimumDb": min(values),
            "medianDb": round(float(np.median(values)), 3),
        }
    return {
        "method": "100ms voiced-window voice versus premaster-minus-voice diagnostic; does not establish listening quality",
        "voicedThresholdDbfs": -38,
        "summary": summary,
        "events": rows,
    }


SILENT_LUFS = -70.0  # BS.1770's absolute gate: anything quieter counts as silence


def momentary_loudness(path: Path) -> list[tuple[float, float, float]]:
    """BS.1770 momentary (400 ms) loudness every 100 ms, from ffmpeg's ebur128.

    Returns ``(windowStart, windowEnd, lufs)`` in seconds on the file's clock.
    """
    result = run([
        "ffmpeg", "-nostats", "-hide_banner", "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-af", "aresample=48000,asetnsamples=n=4800:p=0,ebur128=metadata=1,"
               "ametadata=mode=print:key=lavfi.r128.M:file=-",
        "-f", "null", "-",
    ])
    rows, at = [], None
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("frame:"):
            at = float(line.rsplit("pts_time:", 1)[1])
        elif line.startswith("lavfi.r128.M=") and at is not None:
            # The value attached to the 100 ms frame starting at ``at`` covers
            # the 400 ms that end when that frame ends.
            rows.append((round(at - 0.3, 6), round(at + 0.1, 6), float(line.split("=", 1)[1])))
    return rows


def event_loudness(windows: list[tuple[float, float, float]], start: float, end: float) -> float | None:
    """Energy mean of the momentary windows wholly inside one event, in LUFS.

    ``None`` when the event is shorter than one window; ``SILENT_LUFS`` or
    below means nothing measurable sounded there.
    """
    inside = [lufs for first, last, lufs in windows if first >= start - 1e-6 and last <= end + 1e-6]
    if not inside:
        return None
    power = float(np.mean([10 ** ((value + 0.691) / 10) for value in inside]))
    return round(-0.691 + 10 * math.log10(max(power, 1e-30)), 3)


def plan_following(events: list[dict[str, Any]], buses: dict[str, Path], breaths: list[dict[str, Any]],
                   wave_targets: list[dict[str, Any]] | None,
                   measure: Callable[[Path], list[tuple[float, float, float]]] = momentary_loudness,
                   ) -> dict[str, Any]:
    """Measure each event on each bus and flag only contradictions of the plan.

    A flag is raised when an authored zero-level breath still sounds on the
    score bus, when the authored wave rises into an event but the score bus
    falls, or when mastering moved one event's level against the rest. Nothing
    here judges whether the mix sounds right.
    """
    measured = {name: measure(path) for name, path in buses.items()}
    rows = []
    for event in events:
        row = {"event": event["id"], "kind": event["kind"]}
        for name, windows in measured.items():
            row[name + "Lufs"] = event_loudness(windows, float(event["start"]), float(event["end"]))
        rows.append(row)
    flags = []
    zero_breath_shots = {shot for breath in breaths if float(breath.get("level", 1)) == 0
                         for shot in range(int(breath["shots"][0]), int(breath["shots"][1]) + 1)}
    for event, row in zip(events, rows):
        music = row.get("musicLufs")
        if event.get("shot") in zero_breath_shots and music is not None and music > SILENT_LUFS:
            flags.append({"event": event["id"], "kind": "zero-breath-sounds",
                          "detail": f"an authored level-0 breath measures {music} LUFS on the score bus"})
    if wave_targets is not None and len(wave_targets) == len(rows):
        for previous, current, before, after in zip(wave_targets, wave_targets[1:], rows, rows[1:]):
            rise = float(current.get("envelopeTarget") or 0) / max(float(previous.get("envelopeTarget") or 0), 1e-9)
            heard = (None if before.get("musicLufs") is None or after.get("musicLufs") is None
                     or min(before["musicLufs"], after["musicLufs"]) <= SILENT_LUFS
                     else after["musicLufs"] - before["musicLufs"])
            if rise >= 1.12 and heard is not None and heard < -1.0:
                flags.append({"event": after["event"], "kind": "authored-rise-falls",
                              "detail": f"the wave raises the score {20 * math.log10(rise):.1f} dB into this event "
                                        f"but the score bus falls {-heard:.1f} LU"})
    if "premaster" in measured and "master" in measured:
        offsets = [row["masterLufs"] - row["premasterLufs"] for row in rows
                   if row.get("masterLufs") is not None and row.get("premasterLufs") is not None
                   and min(row["masterLufs"], row["premasterLufs"]) > SILENT_LUFS]
        if offsets:
            median = float(np.median(offsets))
            for row in rows:
                if (row.get("masterLufs") is not None and row.get("premasterLufs") is not None
                        and min(row["masterLufs"], row["premasterLufs"]) > SILENT_LUFS
                        and abs(row["masterLufs"] - row["premasterLufs"] - median) > 1.0):
                    flags.append({"event": row["event"], "kind": "mastering-moved-event-level",
                                  "detail": f"mastering changed this event by "
                                            f"{row['masterLufs'] - row['premasterLufs']:.1f} LU against the "
                                            f"film's {median:.1f} LU"})
    return {
        "method": ("BS.1770 momentary loudness (ffmpeg ebur128), energy-averaged over the 400 ms windows "
                   "wholly inside each event; the score bus is the music stem before sidechain ducking"),
        "silentBelowLufs": SILENT_LUFS,
        "events": rows,
        "flags": flags,
        "listeningVerdict": None,
    }


def packet_hash(path: Path) -> str:
    result = run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-c", "copy", "-f", "hash", "-hash", "sha256", "-",
        ]
    )
    return result.stdout.decode("utf-8").strip()


def body_mux_command(
    picture: Path,
    premaster: Path,
    output: Path,
    normalizer: str,
    runtime: float,
    log_level: str = "error",
) -> list[str]:
    """Return the exact body mux command; intentionally never uses -shortest.

    ``log_level`` changes only what ffmpeg prints (``info`` lets loudnorm report
    its second pass), never the encoded bytes.
    """
    return [
        "ffmpeg", "-y", *(["-nostats"] if log_level != "error" else []), "-v", log_level, "-i", str(picture),
        "-i", str(premaster), "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-af", normalizer, "-ar", str(SR), "-c:a", "aac",
        "-b:a", "256k", "-t", f"{runtime:.9f}",
        "-movflags", "+faststart", str(output),
    ]


def execute_mix(episode: Path, contract_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    evidence = contract["inputEvidence"]
    paths = {
        key: episode_path(episode, evidence[key]["path"], key)
        for key in ("picture", "voice", "music", "sfxStem", "editPlan")
    }
    outputs = {
        key: work_output(episode, value, f"mix {key}")
        for key, value in contract["outputs"].items()
    }
    outputs["premaster"].parent.mkdir(parents=True, exist_ok=True)
    premaster_partial = outputs["premaster"].with_name(
        f".{outputs['premaster'].name}.{os.getpid()}.partial.wav"
    )
    run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(paths["voice"]),
            "-i", str(paths["music"]), "-i", str(paths["sfxStem"]),
            "-filter_complex", filter_graph(contract), "-map", "[out]", "-ar", str(SR),
            "-t", f"{contract['runtimeSeconds']:.9f}", "-c:a", "pcm_f32le",
            str(premaster_partial),
        ]
    )
    os.replace(premaster_partial, outputs["premaster"])
    plan = read_json(paths["editPlan"], "edit plan")
    settings = contract["settings"]
    first_pass = loudness(outputs["premaster"], settings)
    target_lra = second_pass_lra(settings, first_pass)
    norm = (
        f"loudnorm=I={settings['targetIntegratedLufs']}:TP={settings['targetTruePeakDbtp']}:"
        f"LRA={target_lra}:measured_I={first_pass['input_i']}:"
        f"measured_TP={first_pass['input_tp']}:measured_LRA={first_pass['input_lra']}:"
        f"measured_thresh={first_pass['input_thresh']}:offset={first_pass['target_offset']}:"
        "linear=true:print_format=json"
    )
    body_partial = outputs["bodyMaster"].with_name(
        f".{outputs['bodyMaster'].name}.{os.getpid()}.partial.mp4"
    )
    # Exact input durations were already proven. Explicit -t fixes the intended
    # clock without hiding a short stream; -shortest is intentionally absent.
    muxed = run(
        body_mux_command(
            paths["picture"],
            outputs["premaster"],
            body_partial,
            norm,
            contract["runtimeSeconds"],
            log_level="info",
        )
    )
    second_pass = second_pass_record(parse_loudnorm(muxed.stderr), target_lra)
    if second_pass["normalizationType"] == "dynamic":
        if settings.get("refuseDynamicMastering") is True:
            body_partial.unlink(missing_ok=True)
            raise MixError(dynamic_mastering_message(second_pass))
        print(f"warning: {dynamic_mastering_message(second_pass)}", file=sys.stderr, flush=True)
    run(["ffmpeg", "-v", "error", "-i", str(body_partial), "-f", "null", "-"])
    os.replace(body_partial, outputs["bodyMaster"])
    final_probe = probe_media(outputs["bodyMaster"])
    video = final_probe.get("video", [])
    audio = final_probe.get("audio", [])
    if len(video) != 1 or len(audio) != 1:
        raise MixError("body master must contain exactly one video and one audio stream")
    expected_width = 1280
    if (
        video[0].get("width") != expected_width
        or video[0].get("height") != 720
        or Fraction(video[0].get("frameRate", "0/1")) != Fraction(FPS)
        or video[0].get("frames") != contract["expectedVideoFrames"]
    ):
        raise MixError("body master picture geometry/cadence/frame count changed")
    if audio[0].get("sampleRate") != SR or audio[0].get("channels") != CHANNELS:
        raise MixError("body master audio is not 48 kHz stereo")
    if abs(final_probe["durationSeconds"] - contract["runtimeSeconds"]) > 0.1:
        raise MixError("body master duration differs from the measured clock")
    source_picture_packets = packet_hash(paths["picture"])
    output_picture_packets = packet_hash(outputs["bodyMaster"])
    if source_picture_packets != output_picture_packets:
        raise MixError("picture packets changed during audio mux")
    delivered = loudness(outputs["bodyMaster"], settings)
    if (
        abs(finite(delivered.get("input_i"), "delivered integrated loudness")
            - settings["targetIntegratedLufs"]) > 0.5
        or finite(delivered.get("input_tp"), "delivered true peak") > settings["targetTruePeakDbtp"] + 0.5
    ):
        raise MixError("delivered loudness or true peak is outside the acceptance window")
    wave_targets = None
    if "musicWaveAudit" in evidence:
        wave_path = episode_path(episode, evidence["musicWaveAudit"]["path"], "music wave audit")
        verify_hash(wave_path, evidence["musicWaveAudit"].get("sha256"), "music wave audit")
        wave_audit = read_json(wave_path, "music wave audit")
        wave_targets = wave_audit.get("eventTargets") if isinstance(wave_audit, dict) else None
    bus_audit = {
        "schemaVersion": 1,
        "mixerVersion": MIXER_VERSION,
        "contract": {"path": portable(contract_path, episode), "sha256": sha256_file(contract_path)},
        "inputs": {
            key: {"path": portable(path, episode), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "premaster": {
            "path": portable(outputs["premaster"], episode),
            "sha256": sha256_file(outputs["premaster"]),
            "loudness": first_pass,
        },
        "intelligibility": intelligibility(paths["voice"], outputs["premaster"], plan["events"]),
        "planFollowing": plan_following(
            plan["events"],
            {"voice": paths["voice"], "music": paths["music"], "sfx": paths["sfxStem"],
             "premaster": outputs["premaster"], "master": outputs["bodyMaster"]},
            plan.get("musicBreaths", []), wave_targets),
        "scoreNotchAutomationAppliedOnlyToMusicBus": contract["scoreNotchAutomation"],
        "creativeApprovalGranted": False,
    }
    atomic_json(outputs["busAudit"], bus_audit)
    audit = {
        "schemaVersion": 1,
        "mixerVersion": MIXER_VERSION,
        "status": "technical-body-ready-for-audition",
        "output": {
            "path": portable(outputs["bodyMaster"], episode),
            "sha256": sha256_file(outputs["bodyMaster"]),
            "durationSeconds": final_probe["durationSeconds"],
            "video": video[0],
            "audio": audio[0],
        },
        "contract": {"path": portable(contract_path, episode), "sha256": sha256_file(contract_path)},
        "busAudit": {"path": portable(outputs["busAudit"], episode), "sha256": sha256_file(outputs["busAudit"])},
        "deliveredLoudness": delivered,
        "loudnormSecondPass": second_pass,
        "picturePacketsUnchanged": True,
        "picturePacketHashes": {
            "source": source_picture_packets,
            "output": output_picture_packets,
        },
        "usedShortest": False,
        "publishingAuthorized": False,
        "creativeApprovalGranted": False,
    }
    atomic_json(outputs["masterAudit"], audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--sfx-input", default="work/sfx-resolve-v1/final-mix-input.json")
    parser.add_argument("--edit-plan", default="work/assembly-v1/edit-plan.json")
    parser.add_argument("--picture", default="work/assembly-v1/picture-720p.mp4")
    parser.add_argument("--voice", default="work/assembly-v1/voice-hi-48k24.wav")
    parser.add_argument("--music", default="work/assembly-v1/music-wave-48k24.wav")
    parser.add_argument(
        "--settings", type=Path,
        help="optional JSON with settingsVersion 2 and the values it changes: musicGainDb -6..6, "
             "musicDuck/sfxDuck {threshold .005-.1, ratio 1-12, attackMs 1-100, releaseMs 100-1500}, "
             "targetIntegratedLufs -27..-12, targetTruePeakDbtp -3..-1, targetLra 1..50 or "
             '"measured", refuseDynamicMastering (default false: a dynamic loudnorm pass warns and '
             "is recorded). Without it the reviewed v1 defaults apply.")
    parser.add_argument("--write-contract", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args(argv)
    episode = args.episode.resolve()
    try:
        if args.execute:
            if args.write_contract is not None or args.contract is None:
                raise MixError("--execute requires an existing --contract and cannot write one in the same step")
            contract_path = episode_path(episode, args.contract, "saved mix contract")
            contract = verify_saved_contract(episode, contract_path)
            result = execute_mix(episode, contract_path, contract)
        else:
            if args.contract is not None:
                raise MixError("--contract is accepted only with --execute")
            settings = (read_json(episode_path(episode, args.settings, "mixer settings"), "mixer settings")
                        if args.settings is not None else None)
            contract = build_mix_contract(
                episode,
                sfx_input_path=episode_path(episode, args.sfx_input, "SFX input"),
                edit_plan_path=episode_path(episode, args.edit_plan, "edit plan"),
                picture_path=episode_path(episode, args.picture, "picture"),
                voice_path=episode_path(episode, args.voice, "voice"),
                music_path=episode_path(episode, args.music, "music"),
                settings=settings,
            )
            if args.write_contract is not None:
                target = work_output(episode, args.write_contract, "mix contract output")
                atomic_json(target, contract)
                result = {
                    **contract,
                    "writtenContract": portable(target, episode),
                    "writtenContractSha256": sha256_file(target),
                }
            else:
                result = contract
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except MixError as exc:
        print(f"Absolute SFX long-form mix blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
