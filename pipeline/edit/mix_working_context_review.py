#!/usr/bin/env python3
"""Render a non-promotable working watch from its validated SFX inventory.

This is deliberately separate from the final-SFX admission path. It reuses the
shared absolute mixer DSP and technical checks, without declaring provisional
anchors, automated source checks or studio score choices to be final approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import mix_absolute_sfx_longform as mix


def working_settings(profile: str) -> dict:
    settings = mix.default_settings()
    if profile == "music-presence-v2":
        # First measured watch had median voice/background margins of 33.7dB
        # (narration) and44dB (dialogue): safe speech, but overly recessed score.
        # Preserve the authored wave, notches and zero windows; soften only the
        # music sidechain. This remains a listening hypothesis, not approval.
        settings["musicGainDb"] = 4.0
        settings["musicDuck"] = {"threshold": 0.056, "ratio": 3.0, "attackMs": 35, "releaseMs": 700}
    elif profile != "conservative-v1":
        raise mix.MixError("Unknown working mix profile")
    return settings


def speech_protection_notches(plan: dict, audit: dict) -> list[dict]:
    """One bounded local screen-driven pass, not automatic listening approval."""
    events = {row["id"]: row for row in plan["events"]}
    result = []
    for row in audit["intelligibility"]["events"]:
        if row["kind"] != "narration":
            continue
        p10 = mix.finite(row["p10VoiceToBackgroundDb"], "voice/background p10")
        reduction = min(8.0, max(0.0, 3.0 - p10))
        if reduction == 0:
            continue
        event = events[row["event"]]
        # Extend ramps outside the scene so its edge syllables have full cover.
        start = max(0.0, event["start"] - .15)
        end = min(plan["runtimeSeconds"], event["end"] + .22)
        result.append({"cueId": "working-voice-protection-" + row["event"],
                       "sceneId": row["event"], "startSeconds": round(start, 6),
                       "endSeconds": round(end, 6), "scoreGainDb": -round(reduction, 3),
                       "attackSeconds": .15, "releaseSeconds": .22,
                       "screenedP10VoiceToBackgroundDb": p10,
                       "reason": "Bounded local score dip after measured v2 weak-window competition; SFX/voice unchanged; listening pending"})
    return result


def source_paths(episode: Path, inventory: dict) -> dict[str, Path]:
    if inventory.get("kind") != "designed-sfx-context-review-inventory":
        raise mix.MixError("A validated designed-SFX context inventory is required")
    if not inventory.get("episode") or inventory.get("language") != "hi":
        raise mix.MixError("An explicit project and Hindi context contract are required")
    paths = {}
    for row in inventory.get("sourceBindings", []):
        path = mix.episode_path(episode, row["path"], row["role"])
        mix.verify_hash(path, row["sha256"], row["role"])
        paths[row["role"]] = path
    for row in inventory.get("sources", []):
        path = mix.episode_path(episode, row["path"], "raw SFX source")
        mix.verify_hash(path, row["sha256"], "raw SFX source")
    for role in ("picture", "voiceStem", "selectedScoreStem", "contextPreviewSfxStem",
                 "editPlan", "sfxDesign", "sfxRenderAudit"):
        if role not in paths:
            raise mix.MixError(f"Context inventory lacks {role}")
    design = mix.read_json(paths["sfxDesign"], "absolute design")
    audit = mix.read_json(paths["sfxRenderAudit"], "SFX preview audit")
    import copy
    from render_designed_sfx import load_contract
    _, plan, _, _, _ = load_contract(episode, copy.deepcopy(design))
    if (design.get("version") != 2 or design.get("finalApproved") is not False
            or audit.get("finalApproved") is not False or audit.get("workingAudioOnly") is not False
            or plan.get("episode") != inventory["episode"]):
        raise mix.MixError("This preview requires an explicitly non-approved, current designed-SFX receipt")
    mix.verify_hash(paths["sfxDesign"], audit["designSha256"], "SFX design")
    mix.verify_hash(paths["contextPreviewSfxStem"], audit["sha256"], "SFX stem")
    mix.verify_hash(paths["editPlan"], audit["clockSha256"], "SFX clock")
    return paths


def preview_output(episode: Path, directory: str) -> Path:
    path = mix.work_output(episode, directory, "working review directory")
    relative = path.relative_to(episode.resolve())
    if len(relative.parts) < 2 or relative.parts[1] in ("assembly-v1", "final-delivery-v1"):
        raise mix.MixError("Working reviews must not write canonical assembly/final locations")
    return path


def check_vacuums(paths: dict, premaster: Path, design: dict) -> list[dict]:
    """Read small exact sample windows; no whole-film arrays needed here."""
    rows = []
    for window in design.get("hardSfxSilenceWindows", []):
        start, end = round(window["start"] * mix.SR), round(window["end"] * mix.SR)
        samples = {}
        for key, path in {"voice": paths["voiceStem"], "music": paths["selectedScoreStem"],
                          "sfx": paths["contextPreviewSfxStem"], "mix": premaster}.items():
            decoded = mix.run([
                "ffmpeg", "-v", "error", "-ss", f"{start / mix.SR:.9f}", "-i", str(path),
                "-map", "0:a:0", "-t", f"{(end - start) / mix.SR:.9f}",
                "-ar", str(mix.SR), "-ac", "2", "-f", "f32le", "pipe:1",
            ]).stdout
            samples[key] = np.frombuffer(decoded, dtype="<f4").reshape((-1, 2))
            if len(samples[key]) != end - start:
                raise mix.MixError(f"Incomplete exact-sample silence window: {key}")
        music_peak = float(np.max(np.abs(samples["music"])))
        sfx_peak = float(np.max(np.abs(samples["sfx"])))
        residual = float(np.max(np.abs(samples["mix"] - samples["voice"])))
        if music_peak != 0 or sfx_peak != 0 or residual > 1e-6:
            raise mix.MixError(f"Hard silence leaked at {window['sceneId']}")
        rows.append({"sceneId": window["sceneId"], "startSample": start, "endSample": end,
                     "musicPeak": music_peak, "sfxPeak": sfx_peak, "mixMinusVoicePeak": residual})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--inventory", default="work/review/SFX_CONTEXT_INVENTORY.json")
    parser.add_argument("--output-dir", default="work/render-review-v1")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Recheck an already-rendered exact contract; never re-mix media")
    parser.add_argument("--profile", choices=("conservative-v1", "music-presence-v2"),
                        default="conservative-v1")
    parser.add_argument("--speech-protection-audit", help="Exact v2 bus audit for one local protective pass")
    args = parser.parse_args()
    episode = args.episode.resolve()
    inventory_path = mix.episode_path(episode, args.inventory, "working context inventory")
    inventory_sha = mix.sha256_file(inventory_path)
    inventory = mix.read_json(inventory_path, "working context inventory")
    paths = source_paths(episode, inventory)
    output_dir = preview_output(episode, args.output_dir)
    if args.profile != "conservative-v1" and args.output_dir == "work/render-review-v1":
        raise mix.MixError("A revised profile needs a separate versioned output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = mix.read_json(paths["editPlan"], "edit plan")
    runtime = float(plan["runtimeSeconds"])
    frames = round(runtime * mix.FPS)
    if (plan.get("finalApproved") is not False
            or plan["timelineFingerprint"] != inventory["timelineFingerprint"]
            or abs(runtime - inventory["runtimeSeconds"]) > 1e-6):
        raise mix.MixError("Working plan scope/clock differs from the context inventory")
    mix.validate_events(plan, runtime)
    mix.validate_score_placements(episode, plan, runtime)
    conform = mix.read_json(paths["musicConform"], "music conform")
    if any(row.get("repriseJoins") != [] for row in conform):
        raise mix.MixError("This first watch has no authorized automatic music reprises")
    design = mix.read_json(paths["sfxDesign"], "absolute design")
    notches = mix.validate_notches(design.get("scoreNotchAutomation", []), runtime)
    protection_binding = None
    if args.speech_protection_audit:
        if args.profile != "music-presence-v2":
            raise mix.MixError("Local protection is qualified only for the music-presence-v2 profile")
        protection_path = mix.episode_path(episode, args.speech_protection_audit, "v2 bus audit")
        protection = mix.read_json(protection_path, "v2 bus audit")
        for key, role in (("voice", "voiceStem"), ("music", "selectedScoreStem"), ("editPlan", "editPlan")):
            mix.verify_hash(paths[role], protection["inputs"][key]["sha256"], "v2 " + key)
        previous_contract_path = mix.episode_path(episode, protection["contract"]["path"], "v2 contract")
        mix.verify_hash(previous_contract_path, protection["contract"]["sha256"], "v2 contract")
        previous_contract = mix.read_json(previous_contract_path, "v2 contract")
        if (previous_contract.get("profile") != "music-presence-v2"
                or previous_contract.get("speechProtectionEvidence") is not None
                or previous_contract.get("settings") != working_settings(args.profile)):
            raise mix.MixError("Protection must derive once from the original v2 presence mix")
        protection_binding = {"path": mix.portable(protection_path, episode),
                              "sha256": mix.sha256_file(protection_path)}
        notches += mix.validate_notches(speech_protection_notches(plan, protection), runtime)
    roles = {"picture": "picture", "voice": "voiceStem", "music": "selectedScoreStem",
             "sfxStem": "contextPreviewSfxStem", "editPlan": "editPlan"}
    contract = {
        "schemaVersion": 1, "kind": "non-promotable-working-context-mix", "episode": plan["episode"],
        "finalApproved": False, "publishingAuthorized": False,
        "reviewScope": "Studio first full watch; SFX context/anchor and final film listening remain pending",
        "contextReviewInventory": {"path": mix.portable(inventory_path, episode), "sha256": inventory_sha},
        "runtimeSeconds": runtime, "expectedVideoFrames": frames,
        "inputEvidence": {key: {"path": mix.portable(paths[role], episode),
                                "sha256": mix.sha256_file(paths[role])} for key, role in roles.items()},
        "settings": working_settings(args.profile), "scoreNotchAutomation": notches,
        "scoreNotchEnvelope": {"ffmpegVolumeExpression": mix.notch_volume_expression(notches)},
        "outputs": {key: mix.portable(output_dir / filename, episode) for key, filename in {
            "premaster": "working-premaster-48kf32.wav", "bodyMaster": "HI_720P_WORKING_REVIEW.mp4",
            "busAudit": "working-bus-audit.json", "masterAudit": "working-render-audit.json"}.items()},
    }
    if args.profile != "conservative-v1":
        contract["profile"] = args.profile
        contract["profileReason"] = "Restore score presence after first full-watch measurements; retain wave, silences and SFX unchanged"
    if protection_binding:
        contract["speechProtectionEvidence"] = protection_binding
    contract_path = output_dir / "working-mix-contract.json"
    if args.verify_existing:
        if mix.read_json(contract_path, "saved working contract") != contract:
            raise mix.MixError("Existing working mix contract differs from current inputs")
        result = mix.read_json(episode / contract["outputs"]["masterAudit"], "existing mix audit")
        mix.verify_hash(contract_path, result["contract"]["sha256"], "saved mix contract")
        mix.verify_hash(episode / result["output"]["path"], result["output"]["sha256"], "existing watch")
        mix.verify_hash(episode / result["busAudit"]["path"], result["busAudit"]["sha256"], "bus audit")
        bus = mix.read_json(episode / result["busAudit"]["path"], "bus audit")
        mix.verify_hash(episode / bus["premaster"]["path"], bus["premaster"]["sha256"], "premaster")
    else:
        mix.atomic_json(contract_path, contract)
        print("Mixing full Hindi voice + score + absolute SFX preview; final gates unchanged.", flush=True)
        result = mix.execute_mix(episode, contract_path, contract)
    # Recheck every bound input after the long render, not only before it.
    mix.verify_hash(inventory_path, inventory_sha, "context inventory")
    source_paths(episode, inventory)
    if protection_binding:
        mix.verify_hash(episode / protection_binding["path"], protection_binding["sha256"], "v2 protection evidence")
    vacuums = check_vacuums(paths, episode / contract["outputs"]["premaster"], design)
    result.update(kind="non-promotable-working-review", status="working-watch-ready-not-final",
                  finalApproved=False, promotable=False, hardVacuumVerification=vacuums,
                  pendingSfxAnchorCount=len(inventory["provisionalCueIds"]),
                  scoreTakeSequence=[row["take"] for row in conform],
                  contextInventory={"path": mix.portable(inventory_path, episode), "sha256": inventory_sha})
    mix.atomic_json(episode / contract["outputs"]["masterAudit"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
