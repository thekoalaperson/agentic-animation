#!/usr/bin/env python3
"""Verify a designed-SFX v2 stem and optionally retain source-bound review notes.

Renderer, source, timeline and output hashes are technical checks. Review is
optional: missing notes never prevent making or auditioning a mix, and supplied
hash bindings are checked without treating a renderer flag as creative approval.
``--write`` records the technical handoff under work/; it grants no publication
permission. Legacy episode-local renderer receipts are not supported.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SAMPLE_RATE = 48_000
CHANNELS = 2
ROUTE = "self-hosted-text-to-audio/StableAudio3-Small-SFX+deterministic-edits"  # the Small-SFX-only receipt


class ContractError(RuntimeError):
    """The SFX source, render or context-review evidence is incomplete or stale."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ContractError(f"{label} must be a finite number")
    return result


def contained(path: Path, episode: Path, label: str, *, must_exist: bool = True) -> Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(episode.resolve())
    except ValueError as exc:
        raise ContractError(f"{label} escapes project root") from exc
    if "deliverables" in relative.parts:
        raise ContractError(f"{label} must not use deliverables/")
    if must_exist and not resolved.is_file():
        raise ContractError(f"{label} is missing: {resolved}")
    return resolved


def evidence_path(raw: Any, episode: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"{label} has no path")
    return contained(episode / raw, episode, label)


def portable(path: Path, episode: Path) -> str:
    return path.resolve().relative_to(episode.resolve()).as_posix()


def verify_file_hash(path: Path, declared: Any, label: str) -> str:
    if not isinstance(declared, str) or not SHA256.fullmatch(declared):
        raise ContractError(f"{label} has no valid SHA-256")
    actual = sha256_file(path)
    if actual != declared:
        raise ContractError(f"{label} changed: {path}")
    return actual


def binding(path: Path, episode: Path) -> dict[str, str]:
    return {"path": portable(path, episode), "sha256": sha256_file(path)}


def load_renderer():
    path = Path(__file__).with_name("render_designed_sfx.py")
    spec = importlib.util.spec_from_file_location("designed_sfx_admission_renderer", path)
    if spec is None or spec.loader is None:
        raise ContractError("Retained designed-SFX renderer is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def validate_notches(rows: Any, cue_ids: list[str], runtime: float) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ContractError("scoreNotchAutomation must be an array")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("cueId") not in cue_ids or row["cueId"] in seen:
            raise ContractError("Score notch must name one unique rendered cue")
        cue = row["cueId"]
        seen.add(cue)
        start = finite(row.get("startSeconds"), f"{cue} notch start")
        end = finite(row.get("endSeconds"), f"{cue} notch end")
        gain = finite(row.get("scoreGainDb"), f"{cue} notch scoreGainDb")
        attack = finite(row.get("attackSeconds"), f"{cue} notch attack")
        release = finite(row.get("releaseSeconds"), f"{cue} notch release")
        if not 0 <= start < end <= runtime or not -18 <= gain < 0:
            raise ContractError("Score notch has unsafe time or gain")
        if min(attack, release) <= 0 or max(attack, release) > end - start + 1e-6:
            raise ContractError("Score notch has unsafe attack/release")
    return rows


def build_contract(episode: Path, absolute_design_path: Path, render_audit_path: Path,
                   edit_plan_path: Path, picture_segments_path: Path,
                   context_review_path: Path | None = None) -> dict[str, Any]:
    episode = episode.resolve()
    paths = {"absoluteDesign": contained(absolute_design_path, episode, "SFX design"),
             "renderAudit": contained(render_audit_path, episode, "render audit"),
             "editPlan": contained(edit_plan_path, episode, "edit plan"),
             "pictureSegments": contained(picture_segments_path, episode, "picture segments")}
    if context_review_path is not None:
        paths["contextReview"] = contained(context_review_path, episode, "context review")
    design = read_object(paths["absoluteDesign"], "SFX design")
    audit = read_object(paths["renderAudit"], "render audit")
    review = read_object(paths["contextReview"], "context review") if context_review_path is not None else None
    if design.get("version") != 2 or design.get("workingAudioOnly") is not False:
        raise ContractError("Final admission requires designed-SFX v2; working previews and legacy receipts are refused")
    if design.get("generationApiAllowed") is not False or not design.get("authority"):
        raise ContractError("SFX design must declare its authority and allow no generation API")
    if design.get("hardSfxSilenceWindows"):
        raise ContractError("This renderer does not certify legacy hard-silence windows; use a supported authored design")
    for key, design_key in (("editPlan", "clock"), ("pictureSegments", "picture")):
        row = design.get(design_key, {})
        if evidence_path(row.get("path"), episode, design_key) != paths[key]:
            raise ContractError(f"SFX design binds a different {key}")
    resolved = copy.deepcopy(design)
    renderer, renderer_path = load_renderer()
    try:
        batches, plan, picture, wave, render_contract = renderer.load_contract(episode, resolved)
    except (RuntimeError, AssertionError, KeyError, TypeError, ValueError, OSError) as exc:
        raise ContractError(f"Designed-SFX sources or timeline no longer validate: {exc}") from exc
    if plan.get("schemaVersion") != 2 or plan.get("renderable") is not True:
        raise ContractError("Final admission requires a renderable schemaVersion 2 edit plan")
    episode_id, language = plan.get("episode"), plan.get("language")
    if not isinstance(episode_id, str) or not episode_id or not isinstance(language, str) or not language:
        raise ContractError("Edit plan must declare its project identity and language")
    timeline = plan.get("timelineFingerprint")
    if not isinstance(timeline, str) or not SHA256.fullmatch(timeline):
        raise ContractError("Edit plan must bind a measured timeline fingerprint")
    runtime = finite(plan.get("runtimeSeconds"), "runtime")
    if runtime <= 0:
        raise ContractError("Measured runtime must be positive")
    # The receipt names the route its batches declared (ROUTE for Small-SFX alone).
    if (audit.get("route") != renderer.receipt_route(render_contract.get("sourceRoutes", {}))
            or audit.get("workingAudioOnly") is not False):
        raise ContractError("Render receipt is not an independent final-capable SFX render")
    verify_file_hash(renderer_path, audit.get("rendererSha256"), "retained renderer")
    verify_file_hash(paths["absoluteDesign"], audit.get("designSha256"), "rendered design")
    for key in ("clockSha256", "dramaticWaveSha256", "sourceBatches", "picturePlanSha256"):
        if audit.get(key) != render_contract[key]:
            raise ContractError(f"Render receipt has stale {key}")
    if audit.get("authority") != design["authority"]:
        raise ContractError("Render receipt changed its design authority record")
    paths["dramaticWave"] = evidence_path(design["wave"]["path"], episode, "dramatic wave")
    paths["sfxStem"] = evidence_path(audit.get("stem"), episode, "rendered SFX stem")
    stem_hash = verify_file_hash(paths["sfxStem"], audit.get("sha256"), "rendered SFX stem")
    frames = round(runtime * SAMPLE_RATE)
    if (audit.get("codec") != "pcm_f32le" or audit.get("sampleRate") != SAMPLE_RATE
            or audit.get("channels") != CHANNELS or isinstance(audit.get("frames"), bool)
            or not isinstance(audit.get("frames"), int)
            or audit.get("frames") != frames
            or abs(finite(audit.get("seconds"), "SFX seconds") - frames / SAMPLE_RATE) > 1 / SAMPLE_RATE):
        raise ContractError("Rendered stem does not declare the exact 48 kHz stereo float-PCM clock")
    peak = finite(audit.get("peakLinear"), "SFX peakLinear")
    if not 0 <= peak < .95:
        raise ContractError("Rendered SFX stem violates the renderer's independent-stem peak guard")
    sources = {}
    for batch_id, (directory, jobs, hashes) in batches.items():
        for job_id, job in jobs.items():
            source = contained(directory / job["output"], episode, "SFX source")
            if not source.is_relative_to(directory.resolve()):
                raise ContractError("SFX source escapes its verified batch")
            sources[(batch_id, job_id)] = (source, verify_file_hash(source, hashes[job_id], "SFX source"))
    cue_ids = [row.get("id") for row in resolved["cues"]]
    rows = audit.get("cues")
    if (len(cue_ids) != len(set(cue_ids)) or not isinstance(rows, list)
            or not all(isinstance(row, dict) for row in rows)
            or [row.get("id") for row in rows] != cue_ids or audit.get("placements") != len(cue_ids)):
        raise ContractError("Render receipt must cover each authored cue exactly once and in order")
    for cue, row in zip(resolved["cues"], rows):
        if any(row.get(key) != value for key, value in cue.items()):
            raise ContractError(f"Rendered cue differs from its authored design: {cue['id']}")
        identity = (cue.get("sourceBatch"), cue.get("job"))
        if identity not in sources:
            raise ContractError("Cue names an unverified source batch/job")
        source, digest = sources[identity]
        if evidence_path(row.get("source"), episode, "cue source") != source or row.get("sourceSha256") != digest:
            raise ContractError("Rendered cue binds a different source")
        edited = evidence_path(row.get("edited"), episode, "edited SFX cue")
        verify_file_hash(edited, row.get("sha256"), "edited SFX cue")
        duration = finite(row.get("duration"), "edited cue duration")
        if not 0 < duration <= finite(cue["maxSeconds"], "cue maxSeconds") + 1 / SAMPLE_RATE:
            raise ContractError("Rendered cue duration exceeds its authored bound")
    notches = validate_notches(design.get("scoreNotchAutomation", []), cue_ids, runtime)
    evidence = {key: binding(path, episode) for key, path in paths.items()}
    review_info = None
    if review is not None:
        render_bindings = (("designSha256", paths["absoluteDesign"]),
                           ("renderAuditSha256", paths["renderAudit"]), ("stemSha256", paths["sfxStem"]))
        for key, path in render_bindings:
            if key in review:
                verify_file_hash(path, review[key], f"context-review {key}")
        review_info = {**binding(paths["contextReview"], episode),
                       "recordedStatus": review.get("status"),
                       "bindsCurrentRender": all(key in review for key, _ in render_bindings)}
        supplied_inputs = review.get("inputEvidence", {})
        if not isinstance(supplied_inputs, dict):
            raise ContractError("Supplied review inputEvidence must be a hash-binding object")
        for key, row in supplied_inputs.items():
            if not isinstance(row, dict):
                raise ContractError(f"Supplied review {key} must have a path and SHA-256")
            path = evidence_path(row.get("path"), episode, f"reviewed {key}")
            verify_file_hash(path, row.get("sha256"), f"reviewed {key}")
            evidence["reviewed" + key[:1].upper() + key[1:]] = binding(path, episode)
        if review.get("authority") is not None:
            row = review["authority"]
            if not isinstance(row, dict):
                raise ContractError("Supplied review authority must have a path and SHA-256")
            path = evidence_path(row.get("path"), episode, "review authority")
            verify_file_hash(path, row.get("sha256"), "review authority")
            evidence["reviewAuthority"] = binding(path, episode)
    return {"schemaVersion": 2, "kind": "stable-audio-absolute-sfx-final-mix-input",
            "status": "technically-validated-sfx-input-review-not-implied", "episode": episode_id,
            "language": language, "timelineFingerprint": timeline, "runtimeSeconds": runtime,
            "sampleRate": SAMPLE_RATE, "channels": CHANNELS, "inputEvidence": evidence,
            "sourceCount": len(sources), "cueCount": len(cue_ids), "contextReview": review_info,
            "sfxStem": {**binding(paths["sfxStem"], episode), "codec": "pcm_f32le",
                        "sampleRate": SAMPLE_RATE, "channels": CHANNELS, "frames": frames,
                        "durationSeconds": frames / SAMPLE_RATE, "peakLinear": peak,
                        "role": "independent-removable-sfx-only"},
            "scoreNotchAutomation": notches,
            "consumerRequirements": {"thisContractIsNecessaryButNotDeliveryApproval": True,
                "independentlyProbeStemAndAllOtherMedia": True,
                "applyScoreNotchAutomationOnlyToTheMusicBus": True, "preservePicturePacketsDuringAudioMux": True,
                "muxMustNotUseShortest": True, "exactRuntimeFramesAt24Fps": round(runtime * 24)}}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--design", "--absolute-design", dest="design", required=True)
    parser.add_argument("--render-audit", required=True)
    parser.add_argument("--edit-plan", required=True)
    parser.add_argument("--picture-segments", required=True)
    parser.add_argument("--context-review", help="Optional JSON notes; supplied source hashes are verified")
    parser.add_argument("--write", type=Path, help="Optional project-relative output under work/")
    args = parser.parse_args(argv)
    episode = args.episode.resolve()
    try:
        contract = build_contract(episode, *(episode / getattr(args, key) for key in
            ("design", "render_audit", "edit_plan", "picture_segments")),
            context_review_path=episode / args.context_review if args.context_review else None)
        if args.write is not None:
            output = contained(episode / args.write, episode, "contract output", must_exist=False)
            if not output.is_relative_to(episode / "work"):
                raise ContractError("Contract output must remain under work/")
            atomic_json(output, contract)
        print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except ContractError as exc:
        print(f"SFX mix-input refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
