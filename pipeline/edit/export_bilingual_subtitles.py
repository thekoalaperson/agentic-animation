#!/usr/bin/env python3
"""Export hash-bound Hindi and English SRTs into an episode work directory.

This is the final-master counterpart to ``srt_from_lines.py --assembly``.  It
does not infer a clock, translate text, inspect draft plans, or write delivery
assets.  It accepts only the final schema-v2 edit plan, both timeline-generated
subtitle timing documents, a passed packet-clock audit for the exact master,
and a passed voice-sync audit binding that master to the edit plan.

Example usage (after the final 720p master and its audits exist)::

    python3 pipeline/edit/export_bilingual_subtitles.py \
      /path/to/private-project \
      --master <final-720p-review.mp4> \
      --clock-audit <final-clock-audit.json> \
      --voice-sync-audit <final-voice-band-sync-audit.json> \
      --output-dir work/subtitles-v1

Outputs are constrained beneath ``<episode>/work``.  A path in
``deliverables`` (including one reached through a symlink) is always refused.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import textwrap
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import load_relocation_map, relocated_path_matches


SHA256_RE = re.compile(r"[0-9a-f]{64}")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
LANGUAGE_WIDTH = {"hi": 34, "en": 42}
FLOAT_TOLERANCE = 1e-5


class SubtitleExportError(RuntimeError):
    """An input is incomplete, stale, unsafe, or inconsistent."""


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str
    scene_id: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SubtitleExportError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubtitleExportError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SubtitleExportError(f"{label} must be a JSON object: {path}")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise SubtitleExportError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SubtitleExportError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise SubtitleExportError(f"{label} must be a finite number")
    return result


def same_number(left: Any, right: Any, *, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return abs(finite_number(left, "timeline value") - finite_number(right, "timeline value")) <= tolerance


def resolve_episode_path(episode: Path, value: Path | None, default: str) -> Path:
    path = value if value is not None else Path(default)
    return (path if path.is_absolute() else episode / path).resolve()


def safe_output_directory(episode: Path, value: Path) -> Path:
    raw = value if value.is_absolute() else episode / value
    output = raw.resolve()
    work = (episode / "work").resolve()
    deliverables = (episode / "deliverables").resolve()
    if output == deliverables or deliverables in output.parents:
        raise SubtitleExportError("Subtitle exporter refuses every deliverables output path")
    if output != work and work not in output.parents:
        raise SubtitleExportError(f"Subtitle output must remain beneath episode work/: {output}")
    return output


def validate_edit_plan(plan: dict[str, Any], expected_scenes: int) -> tuple[list[dict[str, Any]], str, float, float]:
    if plan.get("schemaVersion") != 2 or plan.get("language") != "hi":
        raise SubtitleExportError("Final edit plan must be schema 2 on the Hindi master clock")
    fingerprint = plan.get("timelineFingerprint")
    if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
        raise SubtitleExportError("Edit plan has no valid final timeline fingerprint")
    events = plan.get("events")
    if not isinstance(events, list) or len(events) != expected_scenes:
        raise SubtitleExportError(
            f"Final edit plan must contain exactly {expected_scenes} scenes; "
            f"found {len(events) if isinstance(events, list) else 'invalid'}"
        )
    fps = finite_number(plan.get("deliveryFps"), "edit-plan deliveryFps")
    if fps <= 0:
        raise SubtitleExportError("Edit-plan deliveryFps must be positive")
    runtime = finite_number(plan.get("runtimeSeconds"), "edit-plan runtimeSeconds")
    if runtime <= 0:
        raise SubtitleExportError("Edit-plan runtimeSeconds must be positive")

    ids: set[str] = set()
    previous_end = 0.0
    frame_tolerance = 0.51 / fps
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise SubtitleExportError(f"Edit-plan scene {index + 1} is invalid")
        scene_id = event.get("id")
        if not isinstance(scene_id, str) or not scene_id or scene_id in ids:
            raise SubtitleExportError(f"Edit-plan scene ID is missing or duplicated: {scene_id!r}")
        ids.add(scene_id)
        start = finite_number(event.get("start"), f"{scene_id}.start")
        duration = finite_number(event.get("duration"), f"{scene_id}.duration")
        end = finite_number(event.get("end"), f"{scene_id}.end")
        if start < 0 or duration <= 0 or not same_number(start + duration, end):
            raise SubtitleExportError(f"Invalid edit-plan window for {scene_id}")
        if index and start < previous_end - FLOAT_TOLERANCE:
            raise SubtitleExportError(f"Overlapping edit-plan scenes: {events[index - 1]['id']} and {scene_id}")
        if abs(start - previous_end) > frame_tolerance:
            raise SubtitleExportError(f"Missing timeline coverage before {scene_id}")
        previous_end = end
    if abs(previous_end - runtime) > frame_tolerance:
        raise SubtitleExportError("Edit-plan scenes do not cover the declared runtime")
    return events, fingerprint, runtime, fps


def _validate_timing_document(
    doc: dict[str, Any],
    language: str,
    events: list[dict[str, Any]],
    fingerprint: str,
    runtime: float,
) -> list[dict[str, Any]]:
    if (
        doc.get("schemaVersion") != 1
        or doc.get("kind") != "hindi-master-subtitle-timing-input"
        or doc.get("language") != language
        or doc.get("masterLanguage") != "hi"
    ):
        raise SubtitleExportError(f"Invalid final {language} subtitle timing contract")
    if doc.get("timelineFingerprint") != fingerprint:
        raise SubtitleExportError(f"Stale {language} timing timeline fingerprint")
    if not same_number(doc.get("duration"), runtime):
        raise SubtitleExportError(f"{language} timing duration differs from the final edit plan")
    shots = doc.get("shots")
    if not isinstance(shots, list) or len(shots) != len(events):
        raise SubtitleExportError(
            f"{language} timing must cover all {len(events)} scenes exactly once"
        )

    previous_window_end = 0.0
    for index, (shot, event) in enumerate(zip(shots, events)):
        if not isinstance(shot, dict) or shot.get("id") != event["id"]:
            got = shot.get("id") if isinstance(shot, dict) else None
            raise SubtitleExportError(
                f"{language} timing scene order/coverage differs at {index + 1}: {got!r}"
            )
        scene_id = event["id"]
        expected = {
            "start": event.get("start"),
            "dur": event.get("duration"),
            "audioAt": event.get("subtitleAudioAt"),
            "audioDuration": event.get("subtitleAudioDuration"),
        }
        for key, value in expected.items():
            if not same_number(shot.get(key), value):
                raise SubtitleExportError(f"{language} timing changed {scene_id}.{key}")
        if shot.get("dialogue") is not (event.get("kind") == "dialogue"):
            raise SubtitleExportError(f"{language} timing changed {scene_id}.dialogue")
        expected_text = event.get("subtitleTextHi" if language == "hi" else "subtitleTextEn")
        text = shot.get("text")
        if not isinstance(text, str) or not text.strip() or text != expected_text:
            raise SubtitleExportError(f"{language} timing text differs from final plan at {scene_id}")
        if event.get('subtitleCuePairs'):
            if (shot.get('cueParts') != [p[language] for p in event['subtitleCuePairs']]
                    or shot.get('cueWeights') != event.get('subtitleCueWeights')):
                raise SubtitleExportError(f'Authored subtitle clauses differ from final plan at {scene_id}')
        elif 'cueParts' in shot or 'cueWeights' in shot:
            raise SubtitleExportError(f'Unbound authored subtitle clauses at {scene_id}')
        if language == "hi" and not DEVANAGARI_RE.search(text):
            raise SubtitleExportError(f"Hindi subtitle has no Devanagari at {scene_id}")
        if language == "en" and DEVANAGARI_RE.search(text):
            raise SubtitleExportError(f"English subtitle contains Devanagari at {scene_id}")

        start = finite_number(shot["start"], f"{scene_id}.start")
        duration = finite_number(shot["dur"], f"{scene_id}.dur")
        audio_at = finite_number(shot["audioAt"], f"{scene_id}.audioAt")
        audio_duration = finite_number(shot["audioDuration"], f"{scene_id}.audioDuration")
        window_start = start + audio_at
        window_end = min(start + duration, window_start + audio_duration)
        if audio_at < 0 or audio_duration <= 0 or not (start <= window_start < window_end <= start + duration + FLOAT_TOLERANCE):
            raise SubtitleExportError(f"Invalid measured subtitle window at {scene_id}")
        if window_start < previous_window_end - FLOAT_TOLERANCE:
            raise SubtitleExportError(f"Overlapping subtitle windows at {scene_id}")
        if window_end > runtime + FLOAT_TOLERANCE:
            raise SubtitleExportError(f"Subtitle window exceeds final timeline at {scene_id}")
        previous_window_end = window_end
    return shots


def validate_bilingual_geometry(hi: list[dict[str, Any]], en: list[dict[str, Any]]) -> None:
    geometry = ("id", "start", "dur", "audioAt", "audioDuration", "dialogue")
    for left, right in zip(hi, en):
        for field in geometry:
            if field in {"start", "dur", "audioAt", "audioDuration"}:
                equal = same_number(left.get(field), right.get(field))
            else:
                equal = left.get(field) == right.get(field)
            if not equal:
                raise SubtitleExportError(
                    f"Hindi/English timing geometry differs at {left.get('id')}.{field}"
                )


def _evidence_master_path(evidence: dict[str, Any], key: str) -> Path:
    value = evidence.get(key)
    if not isinstance(value, str) or not value:
        raise SubtitleExportError(f"Master evidence lacks {key}")
    return Path(value).resolve()


def _names_master(evidence: dict[str, Any], master: Path, episode: Path | None,
                  relocation: dict[str, Any] | None) -> bool:
    """The audit names this master, here or in the project before a move.

    A moved master matches through the episode-relative path new audits record
    beside the absolute one, or through an explicit relocation map. The master
    hash comparison that follows is unchanged in every case.
    """
    recorded = _evidence_master_path(evidence, "master")
    if recorded == master or episode is None:
        return recorded == master
    return relocated_path_matches(evidence["master"], master, episode, relocation,
                                  evidence.get("masterEpisodeRelative"))


def validate_master_evidence(
    master: Path,
    clock: dict[str, Any],
    sync: dict[str, Any],
    plan_path: Path,
    events: list[dict[str, Any]],
    runtime: float,
    fps: float,
    max_master_tail_seconds: float,
    episode: Path | None = None,
    relocation: dict[str, Any] | None = None,
) -> tuple[str, float]:
    if not master.is_file() or master.stat().st_size <= 0:
        raise SubtitleExportError(f"Missing or empty final master: {master}")
    master_sha = sha256_file(master)
    if not SHA256_RE.fullmatch(master_sha):
        raise SubtitleExportError("Could not hash final master")

    if clock.get("passed") is not True or clock.get("decodeTimestampsStrictlyIncreasing") is not True:
        raise SubtitleExportError("Final master packet-clock audit has not passed")
    if not _names_master(clock, master, episode, relocation):
        raise SubtitleExportError("Packet-clock audit names a different final master")
    if clock.get("sha256") != master_sha:
        raise SubtitleExportError("Final master hash differs from packet-clock evidence")
    clock_fps = finite_number(clock.get("fps"), "packet-clock fps")
    frames = finite_number(clock.get("frames"), "packet-clock frames")
    if clock_fps <= 0 or frames <= 0 or not same_number(clock_fps, fps):
        raise SubtitleExportError("Packet-clock cadence differs from the final edit plan")
    duration = frames / clock_fps
    declared_duration = clock.get("durationSeconds", clock.get("duration"))
    if declared_duration is not None and abs(finite_number(declared_duration, "packet-clock duration") - duration) > 1.01 / fps:
        raise SubtitleExportError("Packet-clock duration evidence is internally inconsistent")
    if duration < runtime - 1.01 / fps:
        raise SubtitleExportError("Final master ends before the subtitle timeline")
    if duration - runtime > max_master_tail_seconds + 1.01 / fps:
        raise SubtitleExportError("Final master tail exceeds the approved subtitle export bound")

    if sync.get("passed") is not True or sync.get("timingPassed") is not True or sync.get("voicePresencePassed") is not True:
        raise SubtitleExportError("Final master voice-sync evidence has not passed")
    if not _names_master(sync, master, episode, relocation):
        raise SubtitleExportError("Voice-sync audit names a different final master")
    if sync.get("masterSha256") != master_sha:
        raise SubtitleExportError("Final master hash differs from voice-sync evidence")
    if sync.get("planSha256") != sha256_file(plan_path):
        raise SubtitleExportError("Voice-sync evidence is not bound to the final edit plan")
    if sync.get("events") != len(events):
        raise SubtitleExportError("Voice-sync evidence does not cover all subtitle scenes")
    rows = sync.get("results")
    if not isinstance(rows, list) or len(rows) != len(events):
        raise SubtitleExportError("Voice-sync evidence lacks per-scene results")
    for event, row in zip(events, rows):
        if not isinstance(row, dict) or row.get("event") != event["id"] or row.get("pass") is not True:
            raise SubtitleExportError(f"Voice-sync evidence is missing/pending for {event['id']}")
    return master_sha, duration


def _wrapped_lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _partition(text: str, count: int, width: int) -> list[str] | None:
    words = text.split()
    if not words or count <= 0 or count > len(words):
        return None
    target = len(" ".join(words)) / count
    dangling = {
        "the", "a", "an", "to", "of", "and", "or", "by", "with", "his", "her",
        "के", "की", "का", "और", "से", "में", "को", "तो",
    }
    # (parts used, words consumed) -> (cost, chunks)
    states: dict[tuple[int, int], tuple[float, list[str]]] = {(0, 0): (0.0, [])}
    for parts in range(1, count + 1):
        for end in range(parts, len(words) + 1):
            candidates: list[tuple[float, list[str]]] = []
            for start in range(parts - 1, end):
                previous = states.get((parts - 1, start))
                if previous is None:
                    continue
                chunk = " ".join(words[start:end])
                lines = _wrapped_lines(chunk, width)
                if not lines or len(lines) > 2 or any(len(line) > width for line in lines):
                    continue
                cost = previous[0] + (len(chunk) - target) ** 2
                if words[end - 1].lower().strip(".,;:!?\u0964\u0965\"'”’") in dangling:
                    cost += 100.0
                if re.search(r"[.!?\u0964\u0965:;][\"'”’)]?$", chunk):
                    cost -= 8.0
                candidates.append((cost, previous[1] + [chunk]))
            if candidates:
                states[(parts, end)] = min(candidates, key=lambda row: row[0])
    result = states.get((count, len(words)))
    if result is None:
        return None
    chunks = result[1]
    if " ".join(chunks) != " ".join(words):
        raise SubtitleExportError("Subtitle partition changed source words")
    return chunks


def minimum_partition_count(text: str, width: int) -> int:
    for count in range(1, len(text.split()) + 1):
        if _partition(text, count, width) is not None:
            return count
    raise SubtitleExportError("Subtitle contains a word wider than the two-line reading contract")


def build_synchronized_cues(
    hi: list[dict[str, Any]], en: list[dict[str, Any]]
) -> tuple[list[Cue], list[Cue], list[dict[str, Any]]]:
    result = {"hi": [], "en": []}
    coverage: list[dict[str, Any]] = []
    for hi_shot, en_shot in zip(hi, en):
        scene_id = hi_shot["id"]
        texts = {"hi": hi_shot["text"].strip(), "en": en_shot["text"].strip()}
        authored = 'cueParts' in hi_shot or 'cueParts' in en_shot
        if authored:
            parts={'hi':hi_shot.get('cueParts'), 'en':en_shot.get('cueParts')}
            if any(not isinstance(p,list) or not p for p in parts.values()):
                raise SubtitleExportError(f'Authored bilingual clauses must be paired at {scene_id}')
            count=len(parts['hi'])
            if len(parts['en']) != count:
                raise SubtitleExportError(f'Authored bilingual clause counts differ at {scene_id}')
            for language in ('hi','en'):
                if (any(not isinstance(p,str) or not p.strip() for p in parts[language])
                        or ' '.join(' '.join(p.split()) for p in parts[language]) != ' '.join(texts[language].split())):
                    raise SubtitleExportError(f'Authored clauses changed subtitle words at {scene_id}')
            weights=hi_shot.get('cueWeights')
            if (not isinstance(weights,list) or len(weights)!=count or weights!=en_shot.get('cueWeights')
                    or any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0 for w in weights)):
                raise SubtitleExportError(f'Invalid paired clause weights at {scene_id}')
        else:
            minimums = {
                language: minimum_partition_count(texts[language], LANGUAGE_WIDTH[language])
                for language in ("hi", "en")
            }
            count = max(minimums.values())
            parts = {
                language: _partition(texts[language], count, LANGUAGE_WIDTH[language])
                for language in ("hi", "en")
            }
            weights=[1]*count
        if parts["hi"] is None or parts["en"] is None:
            raise SubtitleExportError(
                f"Cannot make a shared bilingual cue grid without empty fragments at {scene_id}"
            )
        start = finite_number(hi_shot["start"], f"{scene_id}.start") + finite_number(
            hi_shot["audioAt"], f"{scene_id}.audioAt"
        )
        end = min(
            finite_number(hi_shot["start"], f"{scene_id}.start")
            + finite_number(hi_shot["dur"], f"{scene_id}.dur"),
            start + finite_number(hi_shot["audioDuration"], f"{scene_id}.audioDuration"),
        )
        cue_first = len(result["hi"]) + 1
        cumulative=0.0
        total_weight=sum(weights)
        for index in range(count):
            cue_start = start + (end - start) * cumulative / total_weight
            cumulative += weights[index]
            cue_end = end if index == count - 1 else start + (end - start) * cumulative / total_weight
            for language in ("hi", "en"):
                assert parts[language] is not None
                wrapped = "\n".join(_wrapped_lines(parts[language][index], LANGUAGE_WIDTH[language]))
                result[language].append(Cue(cue_start, cue_end, wrapped, scene_id))
        coverage.append(
            {
                "id": scene_id,
                "start": round(start, 6),
                "end": round(end, 6),
                "firstCue": cue_first,
                "cueCount": count,
                "partitionPolicy": 'authored-paired-clauses' if authored else 'balanced-word-partition',
                "timingPolicy": 'proportional-within-complete-utterance; not forced alignment',
            }
        )
    validate_cues(result["hi"], "hi")
    validate_cues(result["en"], "en")
    if [(row.start, row.end) for row in result["hi"]] != [
        (row.start, row.end) for row in result["en"]
    ]:
        raise SubtitleExportError("Bilingual cue grids differ")
    return result["hi"], result["en"], coverage


def validate_cues(cues: Iterable[Cue], language: str) -> None:
    previous_end = 0.0
    previous_end_ms = 0
    seen = False
    for cue in cues:
        seen = True
        if cue.start < previous_end - FLOAT_TOLERANCE or cue.end <= cue.start or not cue.text.strip():
            raise SubtitleExportError(f"Invalid or overlapping {language} cue at {cue.scene_id}")
        start_ms = round(cue.start * 1000)
        end_ms = round(cue.end * 1000)
        if start_ms < previous_end_ms or end_ms <= start_ms:
            raise SubtitleExportError(
                f"{language} cue becomes overlapping or empty after SRT millisecond rounding "
                f"at {cue.scene_id}"
            )
        lines = cue.text.splitlines()
        if len(lines) > 2 or any(len(line) > LANGUAGE_WIDTH[language] for line in lines):
            raise SubtitleExportError(f"{language} cue exceeds reading width at {cue.scene_id}")
        if language == "en" and DEVANAGARI_RE.search(cue.text):
            raise SubtitleExportError(f"English cue contains Devanagari at {cue.scene_id}")
        previous_end = cue.end
        previous_end_ms = end_ms
    if not seen:
        raise SubtitleExportError(f"No {language} subtitle cues were generated")


def timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def render_srt(cues: list[Cue]) -> str:
    return "\n".join(
        f"{index}\n{timestamp(cue.start)} --> {timestamp(cue.end)}\n{cue.text}\n"
        for index, cue in enumerate(cues, 1)
    )


def atomic_text(path: Path, text: str) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def export_subtitles(
    episode: Path,
    *,
    master: Path,
    clock_audit_path: Path,
    voice_sync_audit_path: Path,
    output_dir: Path,
    edit_plan_path: Path | None = None,
    hi_timing_path: Path | None = None,
    en_timing_path: Path | None = None,
    expected_scenes: int | None = None,
    max_master_tail_seconds: float = 30.0,
    basename: str | None = None,
    relocation_map: Path | None = None,
) -> dict[str, Any]:
    episode = episode.resolve()
    if not episode.is_dir():
        raise SubtitleExportError(f"Episode directory does not exist: {episode}")
    if expected_scenes is not None and expected_scenes <= 0:
        raise SubtitleExportError("expected-scenes must be positive")
    if not math.isfinite(max_master_tail_seconds) or max_master_tail_seconds < 0:
        raise SubtitleExportError("max-master-tail-seconds must be finite and nonnegative")
    output = safe_output_directory(episode, output_dir)
    plan_path = resolve_episode_path(episode, edit_plan_path, "work/assembly-v1/edit-plan.json")
    hi_path = resolve_episode_path(episode, hi_timing_path, "work/assembly-v1/subtitle-hi-timing.json")
    en_path = resolve_episode_path(episode, en_timing_path, "work/assembly-v1/subtitle-en-timing.json")
    master = master.resolve()
    clock_audit_path = clock_audit_path.resolve()
    voice_sync_audit_path = voice_sync_audit_path.resolve()

    plan = load_json(plan_path, "final edit plan")
    hi_doc = load_json(hi_path, "final Hindi timing")
    en_doc = load_json(en_path, "final English timing")
    clock = load_json(clock_audit_path, "final packet-clock audit")
    sync = load_json(voice_sync_audit_path, "final voice-sync audit")
    if expected_scenes is None:
        expected_scenes = len(plan.get("events", []))
    if expected_scenes <= 0:
        raise SubtitleExportError("The frozen edit plan must contain at least one scene")
    events, fingerprint, runtime, fps = validate_edit_plan(plan, expected_scenes)
    hi_shots = _validate_timing_document(hi_doc, "hi", events, fingerprint, runtime)
    en_shots = _validate_timing_document(en_doc, "en", events, fingerprint, runtime)
    validate_bilingual_geometry(hi_shots, en_shots)
    master_sha, master_duration = validate_master_evidence(
        master,
        clock,
        sync,
        plan_path,
        events,
        runtime,
        fps,
        max_master_tail_seconds,
        episode=episode,
        relocation=load_relocation_map(relocation_map),
    )
    hi_cues, en_cues, coverage = build_synchronized_cues(hi_shots, en_shots)
    if hi_cues[-1].end > master_duration + 1.01 / fps:
        raise SubtitleExportError("Final subtitle cue extends beyond the final master")

    if basename is None:
        episode_config = load_json(episode / "episode.json", "episode config")
        basename = episode_config.get("slug")
    if not isinstance(basename, str) or not basename or basename in {".", ".."} or Path(basename).name != basename:
        raise SubtitleExportError("Subtitle basename must be one safe filename component")

    output.mkdir(parents=True, exist_ok=True)
    hi_output = output / f"{basename}-hi.srt"
    en_output = output / f"{basename}-en.srt"
    hi_text = render_srt(hi_cues)
    en_text = render_srt(en_cues)
    atomic_text(hi_output, hi_text)
    atomic_text(en_output, en_text)
    audit_path = output / "subtitle-export-audit.json"
    audit = {
        "schemaVersion": 1,
        "kind": "bilingual-work-subtitle-export",
        "episode": plan.get("episode", episode.name),
        "masterLanguage": "hi",
        "timelineFingerprint": fingerprint,
        "sceneCount": len(events),
        "cueCount": len(hi_cues),
        "timelineDurationSeconds": round(runtime, 6),
        "finalMaster": {
            "path": str(master),
            "sha256": master_sha,
            "durationSeconds": round(master_duration, 6),
            "tailAfterTimelineSeconds": round(master_duration - runtime, 6),
        },
        "inputs": {
            "editPlan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "subtitleHiTiming": {"path": str(hi_path), "sha256": sha256_file(hi_path)},
            "subtitleEnTiming": {"path": str(en_path), "sha256": sha256_file(en_path)},
            "packetClockAudit": {
                "path": str(clock_audit_path),
                "sha256": sha256_file(clock_audit_path),
            },
            "voiceSyncAudit": {
                "path": str(voice_sync_audit_path),
                "sha256": sha256_file(voice_sync_audit_path),
            },
        },
        "outputs": {
            "hi": {"path": str(hi_output), "sha256": sha256_file(hi_output)},
            "en": {"path": str(en_output), "sha256": sha256_file(en_output)},
        },
        "validation": {
            "allScenesExactlyOnce": True,
            "noTimelineGapsOrOverlaps": True,
            "sharedBilingualCueGrid": True,
            "englishDevanagariLeakage": False,
            "finalMasterHashesMatchBothAudits": True,
            "voiceSyncBoundToEditPlan": True,
            "outputRestrictedToEpisodeWork": True,
            "maximumLinesPerCue": 2,
            "maximumCharactersPerLine": LANGUAGE_WIDTH,
        },
        "coverage": coverage,
        "humanReview": (
            "pending: watch the exact final master with each selectable track and approve "
            "translation nuance, reading rhythm, line breaks, names, and title-card boundary"
        ),
        "deliveryApproval": False,
        "publicationAuthorization": False,
    }
    atomic_json(audit_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--clock-audit", type=Path, required=True)
    parser.add_argument("--voice-sync-audit", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/subtitles-v1"),
        help="Output directory; must resolve beneath <episode>/work",
    )
    parser.add_argument("--edit-plan", type=Path)
    parser.add_argument("--hi-timing", type=Path)
    parser.add_argument("--en-timing", type=Path)
    parser.add_argument("--expected-scenes", type=int, help="Optional explicit scene-count cross-check")
    parser.add_argument("--max-master-tail-seconds", type=float, default=30.0)
    parser.add_argument("--basename")
    parser.add_argument("--relocation-map", type=Path,
                        help="explicit map for a moved project: originalEpisodeRoot and/or paths")
    args = parser.parse_args()
    audit = export_subtitles(
        args.episode,
        master=args.master,
        clock_audit_path=args.clock_audit,
        voice_sync_audit_path=args.voice_sync_audit,
        output_dir=args.output_dir,
        edit_plan_path=args.edit_plan,
        hi_timing_path=args.hi_timing,
        en_timing_path=args.en_timing,
        expected_scenes=args.expected_scenes,
        max_master_tail_seconds=args.max_master_tail_seconds,
        basename=args.basename,
        relocation_map=args.relocation_map,
    )
    print(
        json.dumps(
            {
                "status": "exported-to-work",
                "sceneCount": audit["sceneCount"],
                "cueCount": audit["cueCount"],
                "outputs": audit["outputs"],
                "humanReview": audit["humanReview"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
