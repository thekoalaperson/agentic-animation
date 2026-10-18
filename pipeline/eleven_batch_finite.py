#!/usr/bin/env python3
"""Finite, resumable ElevenLabs narration batch runner.

The command is a dry run unless ``--submit`` is present.  It intentionally
does not query subscriptions, enable overages, top up an account, choose a
fallback format, or retry a POST.  An interrupted POST is cost-ambiguous and
therefore remains blocked for manual reconciliation.

Manifest schema, version 1 (item paths are relative to the manifest directory
and must remain below ``episodeRoot``):

    {
      "version": 1,
      "language": "hi",
      "voiceId": "<the voice cast for this film>",
      "modelId": "eleven_multilingual_v2",
      "outputFormat": "mp3_44100_192",
      "voiceSettings": {
        "stability": <0..1>,
        "similarity_boost": <0..1>,
        "style": <0..1>,
        "use_speaker_boost": <true|false>,
        "speed": <0.7..1.2>
      },
      "items": [
        {"id": "narration-part-01", "textFile": "part-01-hi.txt",
         "output": "audio/part-01-hi.mp3"}
      ]
    }

The tool supplies no voice settings: every value is the operator's frozen
choice for this film, inside the ranges shown.  An item may contain ``text``
instead of ``textFile``, but never both.  In version 1, voice, model, output
format and voice settings are root-level so a batch cannot silently drift
between items.

Version 2 keeps that property while letting a film cast several voices in one
manifest and one approval.  An optional root ``cast`` maps a speaker name to
``{"voiceId", "modelId", "voiceSettings"}``; an item names its ``speaker``
from the cast, and an item without one uses the root voice (``voiceId``,
``modelId`` and ``voiceSettings`` together), which then must exist.  Every
voice is frozen in the manifest, so nothing drifts between items.  Version 2
also accepts a declared ISO 639-1 ``language`` (sent as ``language_code`` only
to models that accept it), per-item ``voiceSettings`` and ``seed``, and root
``pronunciationDictionaryLocators``.  Models come from ``MODELS``, a dated
allowlist with no fallback.  Run ``--print-schema`` for the whole contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

if __package__:
    from ._common import (atomic_json as _atomic_json,
                          atomic_new_bytes as _atomic_new_bytes,
                          canonical_sha256 as _canonical_sha, env_key, project_root,
                          sha256_file as _sha256_file)
    from .paid_approval import PaidApprovalError, validate_paid_approval
else:
    from _common import (atomic_json as _atomic_json,
                         atomic_new_bytes as _atomic_new_bytes,
                         canonical_sha256 as _canonical_sha, env_key, project_root,
                         sha256_file as _sha256_file)
    from paid_approval import PaidApprovalError, validate_paid_approval


API_ROOT = "https://api.elevenlabs.io/v1/text-to-speech"
ALLOWED_MODEL = "eleven_multilingual_v2"
ALLOWED_FORMAT = "mp3_44100_192"
STATE_VERSION = 1
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
VOICE_SETTING_RANGES = {"stability": (0, 1), "similarity_boost": (0, 1), "style": (0, 1),
                        "speed": (0.7, 1.2), "use_speaker_boost": "boolean"}
ITEM_FIELDS = {"id", "text", "textFile", "textSha256", "output"}
# Version 2 only. Version 1 manifests keep exactly ITEM_FIELDS.
V2_ITEM_FIELDS = {"speaker", "voiceSettings", "seed"}
V2_ROOT_FIELDS = {"cast", "pronunciationDictionaryLocators"}
LANGUAGE_CODE = re.compile(r"[a-z]{2,3}\Z")
SEED_RANGE = (0, 4294967295)
MAX_DICTIONARY_LOCATORS = 3
MODEL_EVIDENCE = ("https://elevenlabs.io/docs/overview/models and "
                  "https://elevenlabs.io/docs/api-reference/text-to-speech/convert, read 2026-10-18")
PRICING_EVIDENCE = ("https://elevenlabs.io/pricing, read 2026-10-18: one credit per character "
                    "for multilingual v2; 0.5 to 1 credit per character for Flash through the "
                    "API; no credit rate stated for v3 or v4. https://elevenlabs.io/pricing/api "
                    "gives dollar prices only and no credit rate.")
# The models a version 2 manifest may freeze. There is no fallback: a model
# not listed here is refused. languageCode says whether the provider documents
# language_code for the model (it states multilingual_v2 does not take it).
# creditsPerCharacter is the highest credit rate the provider publishes for the
# model, or None where it publishes none; the approval and the preflight count
# characters, so they bound credits only where this is at most 1.
# exercised says whether this repository has a recorded paid run on it; a new
# entry needs its own page reading on the day it is added.
MODELS = {
    ALLOWED_MODEL: {"verifiedAt": "2026-10-18", "languageCode": False,
                    "maxCharactersPerRequest": 10000, "creditsPerCharacter": 1,
                    "exercised": True},
    "eleven_v4": {"verifiedAt": "2026-10-18", "languageCode": True,
                  "maxCharactersPerRequest": 10000, "creditsPerCharacter": None,
                  "exercised": False},
    "eleven_v3": {"verifiedAt": "2026-10-18", "languageCode": True,
                  "maxCharactersPerRequest": 5000, "creditsPerCharacter": None,
                  "exercised": False},
    # The pricing page gives 0.5 to 1 credit per character through the API.
    "eleven_flash_v2_5": {"verifiedAt": "2026-10-18", "languageCode": True,
                          "maxCharactersPerRequest": 40000, "creditsPerCharacter": 1,
                          "exercised": False},
}


def model_cautions(plan: dict[str, Any]) -> dict[str, list[str]]:
    """The models a plan uses that have no published credit rate, or no recorded run.

    The approval binds characters. For a model whose credit rate is unpublished
    or above one per character, that ceiling bounds spend only by assumption,
    so the dry run names it for the person who signs the approval record.
    """
    used = sorted({voice["modelId"] for voice in plan["voices"]})
    return {
        "unpublishedPricingModels": [
            m for m in used if MODELS[m]["creditsPerCharacter"] is None
            or MODELS[m]["creditsPerCharacter"] > 1],
        "unexercisedModels": [m for m in used if not MODELS[m]["exercised"]],
    }


def schema() -> dict[str, Any]:
    """The narration manifest contract, read from the validator's own values."""
    return {
        "manifest": "any work/ path; item paths are relative to the manifest",
        "requiredRootFields": ["version", "financialApproval", "language", "voiceId", "modelId",
                               "outputFormat", "voiceSettings", "aggregateCharacters", "items"],
        "optionalRootFields": ["episode", "episodeRoot", "provenancePathBase", "projectRoot",
                               "sourcePlan", "sourcePlanSha256", "productionOverrides",
                               "productionOverridesSha256", "authority", "authoritySha256"],
        "describes": "version 1; version 2 differs where the version2 block says",
        "fixedValues": {"version": 1, "language": "hi", "modelId": ALLOWED_MODEL,
                        "outputFormat": ALLOWED_FORMAT},
        "voiceSettingRanges": VOICE_SETTING_RANGES,
        "voiceSettingsNote": "every key is required and frozen; no value is supplied by the tool",
        "itemFields": sorted(ITEM_FIELDS),
        "itemRule": ("exactly one of text or textFile; textSha256, when present, binds the "
                     "textFile bytes; output is a relative .mp3 below episodeRoot"),
        "aggregateCharacters": "sum of stripped item text lengths; --max-characters must equal it",
        "financialApproval": "boolean; reported, never sufficient: submission needs an approval record",
        "approvalLane": APPROVAL_LANE,
        "approvalScope": ["items", "characters"],
        "version2": {
            "version": 2,
            "rootVoice": ("voiceId, modelId and voiceSettings together; optional when every "
                          "item names a cast speaker"),
            "optionalRootFields": sorted(V2_ROOT_FIELDS),
            "cast": "speaker name -> {voiceId, modelId, voiceSettings}; every key required",
            "language": ("required ISO 639-1 code (or ISO 639-3 where the provider uses it); "
                         "sent as language_code only to models whose entry says languageCode"),
            "itemFields": sorted(ITEM_FIELDS | V2_ITEM_FIELDS),
            "itemVoiceSettings": "optional complete voiceSettings that replaces the speaker's for one item",
            "seed": list(SEED_RANGE),
            "pronunciationDictionaryLocators": (
                f"up to {MAX_DICTIONARY_LOCATORS} of "
                "{pronunciationDictionaryId, versionId}; sent with every item"),
            "models": MODELS,
            "modelEvidence": MODEL_EVIDENCE,
            "pricingEvidence": PRICING_EVIDENCE,
            "pricingNote": ("the approval and the credit preflight count characters; for a "
                            "model whose creditsPerCharacter is null or above 1 that bounds "
                            "credits only by assumption, and the dry run names such models"),
            "deferred": ("previous_text/next_text: the provider page does not say whether "
                         "context text is billed, so the character ceiling could not bind it"),
        },
    }


class PlanError(ValueError):
    """The manifest or durable state is unsafe to execute."""


class UncertainPost(RuntimeError):
    """A POST may have reached ElevenLabs and must not be retried."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _now() -> float:
    return time.time()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _verified_binding(manifest_path: Path, doc: dict[str, Any],
                      path_key: str, sha_key: str) -> dict[str, str] | None:
    raw_path = doc.get(path_key)
    raw_sha = doc.get(sha_key)
    if raw_path is None and raw_sha is None:
        return None
    if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_sha, str):
        raise PlanError(f"{path_key} and {sha_key} must be supplied together")
    rel = Path(raw_path)
    if rel.is_absolute():
        raise PlanError(f"{path_key} must be relative")
    base_mode = doc.get("provenancePathBase")
    if base_mode == "manifest":
        base = manifest_path.parent
    elif base_mode in {"episode", "project"}:
        episode_value = doc.get("projectRoot", doc.get("episodeRoot"))
        if (not isinstance(episode_value, str) or not episode_value
                or Path(episode_value).is_absolute()):
            raise PlanError("episode provenance requires a relative episodeRoot")
        base = (manifest_path.parent / episode_value).resolve()
        if not base.is_dir():
            raise PlanError("episode provenance root is missing")
    else:
        raise PlanError(
            "provenancePathBase must be 'project', 'manifest' or 'episode' when provenance is declared"
        )
    path = (base / rel).resolve()
    if not _is_below(path, base):
        raise PlanError(f"{path_key} escapes its declared provenance base")
    if not path.is_file():
        raise PlanError(f"{path_key} is missing: {path}")
    actual = _sha256_file(path)
    if actual != raw_sha.lower():
        raise PlanError(f"{path_key} changed after the narration manifest was frozen")
    return {"kind": path_key, "path": str(path), "sha256": actual}


def _verified_authority(manifest_path: Path, doc: dict[str, Any]) -> dict[str, str] | None:
    """Check the production authority at load time, so the dry run finds a change.

    Paths resolve from the manifest's directory, exactly as the GPU wrapper
    always resolved them. A manifest without an authority binding is unchanged.
    """
    raw_path, raw_sha = doc.get("authority"), doc.get("authoritySha256")
    if raw_path is None and raw_sha is None:
        return None
    if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_sha, str):
        raise PlanError("authority and authoritySha256 must be supplied together")
    path = (manifest_path.parent / raw_path).resolve()
    if not path.is_file() or _sha256_file(path) != raw_sha:
        raise PlanError(f"production authority changed after the narration manifest was frozen: {path}")
    return {"path": str(path), "sha256": raw_sha}


def _validate_voice_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError("voiceSettings must be an object")
    expected = {"stability", "similarity_boost", "style",
                "use_speaker_boost", "speed"}
    if set(value) != expected:
        raise PlanError(f"voiceSettings keys must be exactly {sorted(expected)}")
    for key in ("stability", "similarity_boost", "style"):
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)):
            raise PlanError(f"voiceSettings.{key} must be numeric")
        if not 0 <= float(value[key]) <= 1:
            raise PlanError(f"voiceSettings.{key} must be between 0 and 1")
    if not isinstance(value["use_speaker_boost"], bool):
        raise PlanError("voiceSettings.use_speaker_boost must be boolean")
    if isinstance(value["speed"], bool) or not isinstance(value["speed"], (int, float)):
        raise PlanError("voiceSettings.speed must be numeric")
    if not 0.7 <= float(value["speed"]) <= 1.2:
        raise PlanError("voiceSettings.speed must be between 0.7 and 1.2")
    return dict(value)


def _hindi_text(text: Any, item_id: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise PlanError(f"{item_id}: narration text must be non-empty")
    text = text.strip()
    return text


def _cast_voice(value: Any, label: str) -> dict[str, Any]:
    """One frozen version 2 voice: an allowlisted model, a voice ID and every setting."""
    if not isinstance(value, dict) or set(value) != {"voiceId", "modelId", "voiceSettings"}:
        raise PlanError(f"{label} must be exactly voiceId, modelId and voiceSettings")
    if not isinstance(value["voiceId"], str) or not SAFE_ID.fullmatch(value["voiceId"]):
        raise PlanError(f"{label}: voiceId is missing or malformed")
    if value["modelId"] not in MODELS:
        raise PlanError(f"{label}: modelId {value['modelId']!r} is not in the dated allowlist "
                        f"{sorted(MODELS)}; there is no fallback")
    return {"voiceId": value["voiceId"], "modelId": value["modelId"],
            "voiceSettings": _validate_voice_settings(value["voiceSettings"])}


def _version2_voices(doc: dict[str, Any]) -> tuple[str, dict[str | None, dict[str, Any]], list]:
    """Language, the voices by speaker (None is the root voice) and dictionary locators."""
    language = doc.get("language")
    if not isinstance(language, str) or not LANGUAGE_CODE.fullmatch(language):
        raise PlanError("version 2 needs a declared lowercase ISO 639 language code")
    voices: dict[str | None, dict[str, Any]] = {}
    root = {key: doc[key] for key in ("voiceId", "modelId", "voiceSettings") if key in doc}
    if root:
        voices[None] = _cast_voice(root, "root voice")
    cast = doc.get("cast", {})
    if not isinstance(cast, dict):
        raise PlanError("cast must map speaker names to voices")
    for speaker, voice in cast.items():
        if not isinstance(speaker, str) or not speaker.strip() or speaker != speaker.strip():
            raise PlanError(f"cast speaker name is empty or padded: {speaker!r}")
        voices[speaker] = _cast_voice(voice, f"cast[{speaker}]")
    if not voices:
        raise PlanError("version 2 needs a root voice or a cast")
    locators = doc.get("pronunciationDictionaryLocators", [])
    if not isinstance(locators, list) or len(locators) > MAX_DICTIONARY_LOCATORS:
        raise PlanError(f"pronunciationDictionaryLocators must list at most "
                        f"{MAX_DICTIONARY_LOCATORS} locators")
    for locator in locators:
        if (not isinstance(locator, dict)
                or set(locator) != {"pronunciationDictionaryId", "versionId"}
                or not all(isinstance(v, str) and SAFE_ID.fullmatch(v) for v in locator.values())):
            raise PlanError("each pronunciation dictionary locator is exactly "
                            "pronunciationDictionaryId and versionId, both frozen")
    return language, voices, [dict(locator) for locator in locators]


def load_plan(manifest_path: Path, max_characters: int) -> dict[str, Any]:
    if max_characters <= 0:
        raise PlanError("--max-characters must be a positive integer")
    manifest_path = manifest_path.resolve()
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read manifest: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("version") not in (1, 2):
        raise PlanError("manifest version must be 1 or 2")
    version = doc["version"]
    if not isinstance(doc.get("financialApproval"), bool):
        raise PlanError("financialApproval must be an explicit boolean")
    if version == 1:
        if set(doc) & V2_ROOT_FIELDS:
            # An older runner would ignore these and speak every line in one voice.
            raise PlanError(f"{sorted(set(doc) & V2_ROOT_FIELDS)} need a version 2 manifest")
        if doc.get("language") != "hi":
            raise PlanError("finite narration batches are Hindi-only (language must be 'hi')")
        voice_id = doc.get("voiceId")
        if not isinstance(voice_id, str) or not SAFE_ID.fullmatch(voice_id):
            raise PlanError("voiceId is missing or malformed")
        if doc.get("modelId") != ALLOWED_MODEL:
            raise PlanError(f"modelId must be {ALLOWED_MODEL}")
        if doc.get("outputFormat") != ALLOWED_FORMAT:
            raise PlanError(f"outputFormat must be {ALLOWED_FORMAT}; no quality fallback")
        settings = _validate_voice_settings(doc.get("voiceSettings"))
        language, locators = "hi", []
        voices: dict[str | None, dict[str, Any]] = {
            None: {"voiceId": voice_id, "modelId": ALLOWED_MODEL, "voiceSettings": settings}}
    else:
        known = set(schema()["requiredRootFields"]) | set(schema()["optionalRootFields"])
        unknown = set(doc) - known - V2_ROOT_FIELDS
        if unknown:
            raise PlanError(f"version 2 does not know root fields {sorted(unknown)}")
        if doc.get("outputFormat") != ALLOWED_FORMAT:
            raise PlanError(f"outputFormat must be {ALLOWED_FORMAT}; no quality fallback")
        language, voices, locators = _version2_voices(doc)
    items = doc.get("items")
    if not isinstance(items, list) or not items:
        raise PlanError("items must be a non-empty array")

    root = manifest_path.parent
    episode_root_raw = doc.get("episodeRoot", ".")
    if not isinstance(episode_root_raw, str) or not episode_root_raw:
        raise PlanError("episodeRoot must be a non-empty relative path")
    if Path(episode_root_raw).is_absolute():
        raise PlanError("episodeRoot must be relative to the manifest")
    episode_root = (root / episode_root_raw).resolve()
    if not episode_root.is_dir():
        raise PlanError(f"episodeRoot is not a directory: {episode_root}")
    bindings = [
        binding for binding in (
            _verified_binding(manifest_path, doc, "sourcePlan", "sourcePlanSha256"),
            _verified_binding(
                manifest_path, doc, "productionOverrides", "productionOverridesSha256"
            ),
        ) if binding is not None
    ]
    authority = _verified_authority(manifest_path, doc)
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    outputs: set[str] = set()
    total = 0
    allowed_item_keys = ITEM_FIELDS if version == 1 else ITEM_FIELDS | V2_ITEM_FIELDS
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            raise PlanError(f"item {pos}: must be an object")
        extra = set(item) - allowed_item_keys
        if extra:
            raise PlanError(f"item {pos}: unsupported per-item fields {sorted(extra)}")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not SAFE_ID.fullmatch(item_id):
            raise PlanError(f"item {pos}: id is missing or malformed")
        if item_id in ids:
            raise PlanError(f"duplicate item id: {item_id}")
        ids.add(item_id)
        has_text = "text" in item
        has_file = "textFile" in item
        if has_text == has_file:
            raise PlanError(f"{item_id}: specify exactly one of text or textFile")
        if has_file:
            if not isinstance(item["textFile"], str) or not item["textFile"]:
                raise PlanError(f"{item_id}: textFile must be a non-empty relative path")
            if Path(item["textFile"]).is_absolute():
                raise PlanError(f"{item_id}: textFile must be relative")
            text_path = (root / item["textFile"]).resolve()
            if not _is_below(text_path, episode_root):
                raise PlanError(f"{item_id}: textFile escapes episodeRoot")
            try:
                raw_text = text_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise PlanError(f"{item_id}: cannot read textFile: {exc}") from exc
            if "textSha256" in item and _sha256_file(text_path) != item["textSha256"]:
                raise PlanError(f"{item_id}: textFile changed after the manifest was frozen")
        else:
            if "textSha256" in item:
                raise PlanError(f"{item_id}: textSha256 binds a textFile, not inline text")
            text_path = None
            raw_text = item["text"]
        text = _hindi_text(raw_text, item_id)
        output_value = item.get("output")
        if not isinstance(output_value, str) or not output_value:
            raise PlanError(f"{item_id}: output must be a non-empty relative path")
        output_rel = Path(output_value)
        if output_rel.is_absolute() or output_rel.suffix.lower() != ".mp3":
            raise PlanError(f"{item_id}: output must be a relative .mp3 path")
        output = (root / output_rel).resolve()
        if not _is_below(output, episode_root):
            raise PlanError(f"{item_id}: output escapes episodeRoot")
        # resolve() keeps case, but macOS volumes usually fold it: two such
        # names are one file, and the second paid response would collide.
        output_key = str(output).casefold()
        if output_key in outputs:
            raise PlanError(f"duplicate output path: {output_value}")
        outputs.add(output_key)
        chars = len(text)
        total += chars
        row = {"id": item_id, "text": text, "textFile":
               str(text_path) if text_path else None,
               "output": str(output), "characters": chars}
        if version == 2:
            row.update(_version2_item(item, item_id, voices, chars))
        normalized.append(row)
    if total > max_characters:
        raise PlanError(f"aggregate characters {total} exceed explicit cap {max_characters}")
    declared_total = doc.get("aggregateCharacters")
    if (not isinstance(declared_total, int) or isinstance(declared_total, bool)
            or declared_total <= 0):
        raise PlanError("aggregateCharacters must be a positive integer")
    if declared_total != total:
        raise PlanError(
            f"declared aggregateCharacters {declared_total} does not equal computed {total}"
        )
    if max_characters != declared_total:
        raise PlanError(
            f"--max-characters must equal frozen aggregateCharacters {declared_total}"
        )

    if version == 1:
        # Unchanged since version 1: recorded state binds this exact dict's hash.
        config = {"language": "hi", "voiceId": voice_id,
                  "modelId": ALLOWED_MODEL, "outputFormat": ALLOWED_FORMAT,
                  "voiceSettings": settings}
    else:
        config = {"version": 2, "language": language, "outputFormat": ALLOWED_FORMAT,
                  "rootVoice": voices.get(None),
                  "cast": {k: v for k, v in voices.items() if k is not None},
                  "pronunciationDictionaryLocators": locators}
    used = {item.get("speaker") for item in normalized}
    return {"manifest": str(manifest_path), "manifestSha256": _sha256_file(manifest_path),
            "version": version,
            "voices": [{"speaker": speaker, "voiceId": voices[speaker]["voiceId"],
                        "modelId": voices[speaker]["modelId"]}
                       for speaker in sorted(used, key=lambda s: (s is not None, s or ""))],
            "config": config, "configSha256": _canonical_sha(config),
            "items": normalized, "aggregateCharacters": total,
            "maxCharacters": max_characters,
            "episodeRoot": str(episode_root),
            "sourceBindings": bindings,
            "sourceBindingsSha256": _canonical_sha(bindings),
            "financialApproval": doc.get("financialApproval") is True,
            "authority": authority,
            "episode": doc.get("episode"),
            "approvalAdmissionSha256": None, "approvalAuthorizationId": None}


def _version2_item(item: dict[str, Any], item_id: str,
                   voices: dict[str | None, dict[str, Any]], chars: int) -> dict[str, Any]:
    speaker = item.get("speaker")
    if speaker not in voices:
        raise PlanError(f"{item_id}: speaker {speaker!r} is not in the cast"
                        if speaker is not None else
                        f"{item_id}: names no speaker and the manifest has no root voice")
    voice = dict(voices[speaker])
    if "voiceSettings" in item:
        voice["voiceSettings"] = _validate_voice_settings(item["voiceSettings"])
    if chars > MODELS[voice["modelId"]]["maxCharactersPerRequest"]:
        raise PlanError(f"{item_id}: {chars} characters exceed {voice['modelId']}'s "
                        "documented per-request limit")
    result: dict[str, Any] = {"speaker": speaker, "voice": voice}
    if "seed" in item:
        seed = item["seed"]
        if (isinstance(seed, bool) or not isinstance(seed, int)
                or not SEED_RANGE[0] <= seed <= SEED_RANGE[1]):
            raise PlanError(f"{item_id}: seed must be an integer in {list(SEED_RANGE)}")
        result["seed"] = seed
    return result


def request_url(plan: dict[str, Any], item: dict[str, Any]) -> str:
    config = plan["config"]
    voice_id = item["voice"]["voiceId"] if plan.get("version") == 2 else config["voiceId"]
    return (f"{API_ROOT}/{urllib.parse.quote(voice_id, safe='')}"
            f"?output_format={urllib.parse.quote(config['outputFormat'], safe='')}")


def request_body(plan: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """The exact POST body. Version 1 keeps its original three keys."""
    config = plan["config"]
    if plan.get("version") != 2:
        return {"text": item["text"], "model_id": config["modelId"],
                "voice_settings": config["voiceSettings"]}
    voice = item["voice"]
    body: dict[str, Any] = {"text": item["text"], "model_id": voice["modelId"],
                            "voice_settings": voice["voiceSettings"]}
    if MODELS[voice["modelId"]]["languageCode"]:
        body["language_code"] = config["language"]
    if "seed" in item:
        body["seed"] = item["seed"]
    if config["pronunciationDictionaryLocators"]:
        body["pronunciation_dictionary_locators"] = [
            {"pronunciation_dictionary_id": row["pronunciationDictionaryId"],
             "version_id": row["versionId"]}
            for row in config["pronunciationDictionaryLocators"]]
    return body


def _initial_state(plan: dict[str, Any]) -> dict[str, Any]:
    return {"version": STATE_VERSION, "manifest": plan["manifest"],
            "manifestSha256": plan["manifestSha256"],
            "configSha256": plan["configSha256"],
            "aggregateCharacters": plan["aggregateCharacters"],
            "sourceBindingsSha256": plan["sourceBindingsSha256"],
            "approvalAdmissionSha256": (
                plan["approvalAdmission"]["sha256"]
                if plan.get("approvalAdmission") else None
            ),
            "approvalAuthorizationId": (
                plan["approvalAdmission"]["authorizationId"]
                if plan.get("approvalAdmission") else None
            ),
            "maxCharacters": plan["maxCharacters"], "createdAt": _now(),
            "updatedAt": _now(),
            "items": {item["id"]: {"status": "PLANNED",
                                     "characters": item["characters"],
                                     "output": item["output"]}
                      for item in plan["items"]}}


def _load_or_create_state(plan: dict[str, Any], state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(f"cannot read durable state: {exc}") from exc
        if state.get("version") != STATE_VERSION:
            raise PlanError("durable state version mismatch")
        for field in ("manifestSha256", "configSha256", "aggregateCharacters",
                      "maxCharacters", "sourceBindingsSha256",
                      "approvalAdmissionSha256", "approvalAuthorizationId"):
            if state.get(field) != plan[field]:
                raise PlanError(f"durable state does not match current plan: {field}")
        if set(state.get("items", {})) != {x["id"] for x in plan["items"]}:
            raise PlanError("durable state item set does not match manifest")
        return state
    state = _initial_state(plan)
    _atomic_json(state_path, state)
    return state


def validate_mp3(path: Path, minimum_seconds: float = 0.05) -> dict[str, Any]:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format",
             "-of", "json", str(path)], check=True, capture_output=True, text=True)
        info = json.loads(probe.stdout)
        audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        if not audio:
            raise PlanError(f"response contains no audio stream: {path}")
        duration = float((info.get("format") or {}).get("duration") or
                         audio[0].get("duration") or 0)
        if duration < minimum_seconds:
            raise PlanError(f"response audio is empty or truncated: {path}")
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-map", "0:a:0", "-f", "null", "-"], check=True,
                       capture_output=True)
    except (OSError, subprocess.CalledProcessError, ValueError,
            json.JSONDecodeError) as exc:
        raise PlanError(f"MP3 decode validation failed for {path}: {exc}") from exc
    return {"durationSeconds": duration, "audioStreams": len(audio),
            "codec": audio[0].get("codec_name")}


def _post_once(url: str, body: dict[str, Any], api_key: str) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"xi-api-key": api_key, "Content-Type": "application/json",
                 "Accept": "audio/mpeg"}, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
        return raw, {"status": response.status,
                     "requestId": response.headers.get("request-id"),
                     "contentType": response.headers.get("content-type")}


def _persist(state_path: Path, state: dict[str, Any]) -> None:
    state["updatedAt"] = _now()
    _atomic_json(state_path, state)


def _verify_resumable(plan: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for item in plan["items"]:
        record = state["items"][item["id"]]
        status = record.get("status")
        output = Path(item["output"])
        if status == "COMPLETED":
            expected = record.get("sha256")
            if not output.is_file() or not expected or _sha256_file(output) != expected:
                raise PlanError(f"{item['id']}: completed output missing or hash changed")
            continue
        if status != "PLANNED":
            raise UncertainPost(
                f"{item['id']}: state is {status}; reconcile it manually, never auto-retry")
        if output.exists():
            raise PlanError(f"{item['id']}: untracked output already exists: {output}")
        pending.append(item)
    return pending


APPROVAL_LANE = "elevenlabs-narration"


def approval_scope(plan: dict[str, Any]) -> dict[str, Any]:
    """The finite execution limits an approval record must bind exactly."""
    return {"items": len(plan["items"]), "characters": plan["aggregateCharacters"]}


def execute(manifest_path: Path, max_characters: int, submit: bool,
            state_path: Path | None = None,
            post_fn: Callable[[str, dict[str, Any], str],
                              tuple[bytes, dict[str, Any]]] | None = None,
            api_key: str | None = None,
            approval_admission: Path | None = None) -> dict[str, Any]:
    plan = load_plan(manifest_path, max_characters)
    try:
        approval = validate_paid_approval(
            approval_admission,
            episode=plan["episode"],
            lane=APPROVAL_LANE,
            manifest_path=Path(plan["manifest"]),
            manifest_sha256=plan["manifestSha256"],
            finite_execution=approval_scope(plan),
        )
    except PaidApprovalError as exc:
        raise PlanError(str(exc)) from exc
    plan["approvalAdmission"] = approval
    plan["approvalAdmissionSha256"] = approval["sha256"] if approval else None
    plan["approvalAuthorizationId"] = approval["authorizationId"] if approval else None
    # The manifest's financialApproval is reported, never trusted on its own.
    plan["financialAuthorized"] = approval is not None
    default_state = Path(plan["manifest"]).with_suffix(".eleven-state.json")
    state_path = (state_path or default_state).resolve()
    if state_path != default_state.resolve():
        raise PlanError(
            f"custom state paths are forbidden for paid batches; use canonical {default_state}"
        )
    if not submit:
        report = {"mode": "DRY_RUN", "networkCalls": 0,
                  "financialApproval": plan["financialApproval"],
                  "financialAuthorized": plan["financialAuthorized"],
                  "approvalAdmissionSha256": (
                      approval["sha256"] if approval else None
                  ),
                  "manifest": plan["manifest"],
                  "aggregateCharacters": plan["aggregateCharacters"],
                  "maxCharacters": plan["maxCharacters"],
                  "items": [{"id": x["id"], "characters": x["characters"],
                             "output": x["output"],
                             **({"speaker": x["speaker"], "voiceId": x["voice"]["voiceId"],
                                 "modelId": x["voice"]["modelId"],
                                 "creditsPerCharacter":
                                     MODELS[x["voice"]["modelId"]]["creditsPerCharacter"],
                                 "exercised": MODELS[x["voice"]["modelId"]]["exercised"]}
                                if plan["version"] == 2 else {})}
                            for x in plan["items"]]}
        if plan["version"] == 2:
            report.update(model_cautions(plan))
        return report

    if not plan["financialAuthorized"]:
        raise PlanError(
            "submission requires a current approval record (--approval-admission); "
            "the manifest's financialApproval never authorizes spending"
        )

    # The lock is owned by the immutable manifest, never by a caller-selected
    # state path.  That makes a second runner for the same paid batch collide.
    lock_path = Path(plan["manifest"]).with_suffix(".eleven-execution.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PlanError(f"batch lock exists; verify no runner is active: {lock_path}") from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        os.fsync(lock_fd)
        os.close(lock_fd)
        state = _load_or_create_state(plan, state_path)
        pending = _verify_resumable(plan, state)
        if not pending:
            return {"mode": "SUBMIT", "networkCalls": 0,
                    "status": "ALREADY_COMPLETED", "state": str(state_path)}
        fn = post_fn or _post_once
        if api_key is None:
            try:
                api_key = env_key("ELEVENLABS_API_KEY")
            except (SystemExit, OSError, ValueError) as exc:
                raise PlanError(str(exc)) from exc
        if not api_key:
            raise PlanError("ELEVENLABS_API_KEY is missing from the explicit credential sources")
        pending_root = state_path.parent / f".{state_path.stem}-responses"
        calls = 0
        for item in pending:
            record = state["items"][item["id"]]
            url = request_url(plan, item)
            body = request_body(plan, item)
            record.update(status="POSTING_UNCERTAIN", postAttempt=1,
                          requestSha256=_canonical_sha(body), postStartedAt=_now())
            _persist(state_path, state)  # durable before the network boundary
            calls += 1
            try:
                raw, response_meta = fn(url, body, api_key)
            except urllib.error.HTTPError as exc:
                detail = exc.read(1024).decode("utf-8", "replace")
                if 400 <= exc.code < 500:
                    record.update(status="HTTP_4XX_REFUSED", httpStatus=exc.code,
                                  responseExcerpt=detail, postFinishedAt=_now())
                    _persist(state_path, state)
                    raise PlanError(f"{item['id']}: HTTP {exc.code}; automatic retry forbidden") from exc
                record.update(status="UNCERTAIN_HTTP_ERROR", httpStatus=exc.code,
                              responseExcerpt=detail, postFinishedAt=_now())
                _persist(state_path, state)
                raise UncertainPost(f"{item['id']}: HTTP {exc.code}; reconcile before retry") from exc
            except Exception as exc:
                record.update(status="UNCERTAIN_NETWORK_ERROR",
                              error=f"{type(exc).__name__}: {exc}", postFinishedAt=_now())
                _persist(state_path, state)
                raise UncertainPost(f"{item['id']}: POST outcome uncertain; reconcile before retry") from exc

            response_sha = _sha256_bytes(raw)
            staged = pending_root / f"{item['id']}-{response_sha[:16]}.mp3"
            _atomic_bytes(staged, raw)
            record.update(status="RESPONSE_CAPTURED", postFinishedAt=_now(),
                          responseSha256=response_sha, stagedResponse=str(staged),
                          response=response_meta)
            _persist(state_path, state)  # durable immediately after response capture
            try:
                # A tiny but decodable MP3 is not a narration response.  This
                # conservative Hindi-character floor catches severe truncation
                # while leaving wide room for speaking-rate variation.
                media = validate_mp3(staged, max(0.5, item["characters"] / 40.0))
            except Exception as exc:
                record.update(status="RESPONSE_INVALID", validationError=str(exc))
                _persist(state_path, state)
                raise
            record.update(status="RESPONSE_VALIDATED", media=media)
            _persist(state_path, state)
            output = Path(item["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                record.update(status="OUTPUT_CONFLICT")
                _persist(state_path, state)
                raise PlanError(f"refusing to overwrite output created during run: {output}")
            # The final temp file must live beside the destination so publication
            # remains atomic even if the work tree crosses filesystem boundaries.
            # Keep the captured response as reconciliation evidence.
            try:
                _atomic_new_bytes(output, raw)
            except FileExistsError as exc:
                record.update(status="OUTPUT_CONFLICT")
                _persist(state_path, state)
                raise PlanError(f"refusing to overwrite output created during run: {output}") from exc
            final_sha = _sha256_file(output)
            if final_sha != response_sha:
                record.update(status="OUTPUT_HASH_MISMATCH", outputSha256=final_sha)
                _persist(state_path, state)
                raise PlanError(f"{item['id']}: atomic output hash mismatch")
            record.update(status="COMPLETED", outputSha256=final_sha,
                          sha256=final_sha, completedAt=_now())
            _persist(state_path, state)
        return {"mode": "SUBMIT", "networkCalls": calls, "status": "COMPLETED",
                "state": str(state_path), "aggregateCharacters": plan["aggregateCharacters"],
                "approvalAdmissionSha256": approval["sha256"] if approval else None}
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def freeze(draft_path: Path, manifest_path: Path, project: Path, *,
           voice_settings: dict[str, Any] | None, voice_id: str | None = None,
           authority_path: Path | None = None, cast: dict[str, Any] | None = None,
           language: str | None = None) -> dict[str, Any]:
    """Write a narration manifest from a draft plan, then dry-run it.

    Accepts the narration-plan.json that prepare_episode_production.py writes
    (project-relative outputs, inline text) and items that name a textFile,
    which are bound by textSha256. Voice settings are the operator's own
    frozen choice; the tool supplies none. financialApproval is always false.
    """
    project = project.resolve()
    manifest_path = manifest_path.resolve()
    if manifest_path.exists():
        raise PlanError(f"refusing to overwrite a frozen manifest: {manifest_path}")
    if not _is_below(manifest_path, project) or not _is_below(draft_path.resolve(), project):
        raise PlanError("draft and manifest must both sit inside the project")
    try:
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read narration draft: {exc}") from exc
    voice = voice_id or draft.get("voiceId") or draft.get("voice")
    model = draft.get("modelId", draft.get("model", ALLOWED_MODEL))
    if cast is None and language in (None, "hi"):
        if voice_settings is None:
            raise PlanError("a single-voice freeze needs --voice-settings")
        settings = _validate_voice_settings(voice_settings)
        if not isinstance(voice, str) or not SAFE_ID.fullmatch(voice):
            raise PlanError("no cast voice: pass --voice-id or name voiceId in the draft")
        if model != ALLOWED_MODEL:
            raise PlanError(f"a version 1 manifest accepts only {ALLOWED_MODEL}; the draft asks "
                            f"for {model!r}; pass --cast to freeze version 2")
    items = draft.get("items")
    if not isinstance(items, list) or not items:
        raise PlanError("draft items must be a non-empty list")
    here = manifest_path.parent
    frozen, problems, total = [], [], 0
    version = 1 if cast is None and language in (None, "hi") else 2
    for position, item in enumerate(items):
        item_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(item_id, str) or not SAFE_ID.fullmatch(item_id):
            problems.append(f"item {position}: id is missing or malformed")
            continue
        output = item.get("output")
        if not isinstance(output, str) or not output.endswith(".mp3"):
            problems.append(f"{item_id}: output must be a project-relative .mp3 path")
            continue
        row: dict[str, Any] = {"id": item_id,
                               "output": os.path.relpath(project / output, here)}
        if isinstance(item.get("textFile"), str):
            text_path = (project / item["textFile"]).resolve()
            if not text_path.is_file():
                problems.append(f"{item_id}: textFile is missing: {text_path}")
                continue
            row.update(textFile=os.path.relpath(text_path, here),
                       textSha256=_sha256_file(text_path))
            text = text_path.read_text(encoding="utf-8")
        else:
            text = item.get("text")
            row["text"] = text
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{item_id}: narration text is empty")
            continue
        total += len(text.strip())
        if version == 2:
            if isinstance(item.get("speaker"), str) and item["speaker"].strip():
                row["speaker"] = item["speaker"].strip()
            row.update({key: item[key] for key in ("seed", "voiceSettings") if key in item})
        frozen.append(row)
    if problems:
        raise PlanError("cannot freeze yet: " + "; ".join(problems))
    relative_root = os.path.relpath(project, here)
    if version == 1:
        doc: dict[str, Any] = {
            "version": 1, "episode": draft.get("episode"), "financialApproval": False,
            "language": "hi", "voiceId": voice, "modelId": ALLOWED_MODEL,
            "outputFormat": ALLOWED_FORMAT, "voiceSettings": settings,
            "episodeRoot": relative_root, "projectRoot": relative_root,
            "provenancePathBase": "project",
            "sourcePlan": draft_path.resolve().relative_to(project).as_posix(),
            "sourcePlanSha256": _sha256_file(draft_path.resolve()),
            "aggregateCharacters": total, "items": frozen,
        }
    else:
        doc = {
            "version": 2, "episode": draft.get("episode"), "financialApproval": False,
            "language": language or draft.get("language") or "hi",
            "outputFormat": ALLOWED_FORMAT, "cast": cast or {},
            "episodeRoot": relative_root, "projectRoot": relative_root,
            "provenancePathBase": "project",
            "sourcePlan": draft_path.resolve().relative_to(project).as_posix(),
            "sourcePlanSha256": _sha256_file(draft_path.resolve()),
            "aggregateCharacters": total, "items": frozen,
        }
        # The root voice speaks every item that names no speaker; it is the
        # operator's choice, frozen only when they give its settings.
        if voice_settings is not None:
            if not isinstance(voice, str) or not SAFE_ID.fullmatch(voice):
                raise PlanError("--voice-settings names a root voice: pass --voice-id too")
            doc.update(voiceId=voice, modelId=model, voiceSettings=voice_settings)
    if authority_path is not None:
        authority_path = authority_path.resolve()
        if not authority_path.is_file():
            raise PlanError(f"authority file is missing: {authority_path}")
        doc.update(authority=os.path.relpath(authority_path, here),
                   authoritySha256=_sha256_file(authority_path))
    raw = (json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        _atomic_new_bytes(manifest_path, raw)
    except FileExistsError as exc:
        raise PlanError(f"refusing to overwrite a frozen manifest: {manifest_path}") from exc
    try:
        return execute(manifest_path, total, False)
    except PlanError:
        manifest_path.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Item fields: " + ", ".join(sorted(ITEM_FIELDS)) + ". Voice settings: "
                + ", ".join(sorted(VOICE_SETTING_RANGES)) + ". Run --print-schema for the "
                "whole contract; the approval lane is " + APPROVAL_LANE + "."))
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--max-characters", type=int,
                        help="explicit hard cap for all manifest text; must equal aggregateCharacters")
    parser.add_argument("--print-schema", action="store_true",
                        help="print the narration manifest contract and exit")
    parser.add_argument("--freeze", type=Path, metavar="DRAFT",
                        help="write MANIFEST from a narration plan, then dry-run it")
    freezing = parser.add_argument_group("freeze inputs")
    freezing.add_argument("--project", type=Path, help="project root (or AGENTIC_PROJECT_ROOT)")
    freezing.add_argument("--voice-settings", type=Path,
                          help="JSON with every voiceSettings key, chosen for this film")
    freezing.add_argument("--voice-id", help="the cast voice; otherwise the draft's voiceId")
    freezing.add_argument("--cast", type=Path,
                          help="JSON mapping each speaker to {voiceId, modelId, voiceSettings}; "
                               "writes a version 2 manifest")
    freezing.add_argument("--language", help="ISO 639 code a version 2 manifest declares")
    freezing.add_argument("--authority", type=Path, help="production authority file to bind")
    parser.add_argument("--submit", action="store_true",
                        help="perform at most one POST for each pending item")
    parser.add_argument("--approval-admission", type=Path,
                        help="hash-bound external paid authorization record")
    args = parser.parse_args(argv)
    if args.print_schema:
        print(json.dumps(schema(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.freeze is not None:
        if args.manifest is None or (args.voice_settings is None and args.cast is None):
            parser.error("--freeze DRAFT needs the MANIFEST path to write and "
                         "--voice-settings, --cast or both")
        try:
            settings = (json.loads(args.voice_settings.read_text(encoding="utf-8"))
                        if args.voice_settings else None)
            cast = json.loads(args.cast.read_text(encoding="utf-8")) if args.cast else None
            result = freeze(args.freeze, args.manifest, project_root(args.project),
                            voice_settings=settings, voice_id=args.voice_id,
                            authority_path=args.authority, cast=cast,
                            language=args.language)
        except (PlanError, ValueError, OSError) as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"frozen": str(args.manifest.resolve()), **result},
                         ensure_ascii=False, indent=2))
        return 0
    if args.manifest is None or args.max_characters is None:
        parser.error("manifest and --max-characters are required")
    try:
        result = execute(
            args.manifest,
            args.max_characters,
            args.submit,
            approval_admission=args.approval_admission,
        )
    except (PlanError, UncertainPost) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
