#!/usr/bin/env python3
"""Dry-run-first, finite Kie/Suno score execution and recovery.

This lane never queries or mutates a wallet and has no top-up operation.  A
generation POST is possible only with --submit, an authorized immutable
manifest, and caller-supplied request/credit ceilings.  Each movement receives
at most one POST.  Ambiguous POSTs and provider failures stop the run and are
never regenerated automatically.  --poll is a separate, idempotent recovery
operation for already-persisted task IDs.

--submit is re-entrant. On a fresh batch it posts one cue; once that cue is
DOWNLOADED, the next --submit posts every cue still PLANNED. While the first
cue is in flight it posts nothing. A wrong price or payload therefore costs one
request. run_score_gpu.py does this sequence in one invocation.

A version 2 cue may extend an earlier cue's take (derivedFrom). It waits as
WAITING_PARENT, is never posted while its parent is in flight, and is posted
once by a later --submit after the parent is DOWNLOADED. Recorded version 1
manifests keep their nine-key payload rule and their recorded routes.
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
                          atomic_new_bytes as _atomic_bytes_new, env_key,
                          sha256_file as _sha_file, usd_per_credit)
    from .paid_approval import PaidApprovalError, validate_paid_approval
else:
    from _common import (atomic_json as _atomic_json,
                         atomic_new_bytes as _atomic_bytes_new, env_key,
                         sha256_file as _sha_file, usd_per_credit)
    from paid_approval import PaidApprovalError, validate_paid_approval


API_ROOT = "https://api.kie.ai/api/v1/"
STATE_VERSION = 1
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
SUCCESS = {"SUCCESS", "CALLBACK_EXCEPTION"}
# https://docs.kie.ai/suno-api/get-music-details — partial success still needs
# polling until both included tracks are ready.
IN_PROGRESS = {"PENDING", "WAITING", "QUEUING", "GENERATING", "CREATE_TASK",
               "TEXT_SUCCESS", "FIRST_SUCCESS"}


PAYLOAD_FIELDS = {"model", "customMode", "instrumental", "prompt", "title",
                  "duration", "callBackUrl", "negativeTags", "style"}
CUE_FIELDS = {"id", "shots", "emotionalArc", "payload"}

# Version 2 manifests. Version 1 keeps its nine-key rule above, unchanged, so
# recorded manifests validate and resume exactly as they were frozen.
V6_FAMILY = frozenset({"V6", "V6_MINI", "V6_WILD"})
MANIFEST_VERSIONS = {1, 2}
DERIVED_CUE_FIELD = "derivedFrom"
DERIVED_OPERATIONS = {"extend"}
# Per-route code allowlists, read from Kie's generate-music and extend-music
# pages on 2026-10-18. Frozen keys are camelCase; _create maps them to the
# provider's snake_case input names. A key on no list is refused.
V6_OPTIONAL_FIELDS = frozenset({"prompt", "callBackUrl", "negativeTags", "styleWeight",
                                "weirdnessConstraint", "audioWeight", "variety",
                                "vocalGender", "personaId", "personaModel"})
ROUTE_FIELDS = {
    "generate-v6": {"required": frozenset({"model", "customMode", "instrumental", "title", "style"}),
                    "optional": V6_OPTIONAL_FIELDS | {"duration"}},
    # instrumental is required here although the provider defaults it to false:
    # an extend of an instrumental take must not become a sung request by omission.
    "extend-v6": {"required": frozenset({"model", "instrumental"}),
                  "optional": V6_OPTIONAL_FIELDS | {"style", "title"}},
}
V6_INPUT_NAMES = {"customMode": "custom_mode", "negativeTags": "negative_tags",
                  "styleWeight": "style_weight", "weirdnessConstraint": "weirdness_constraint",
                  "audioWeight": "audio_weight", "vocalGender": "vocal_gender",
                  "personaId": "persona_id", "personaModel": "persona_model",
                  "audioId": "audio_id", "continueAt": "continue_at"}
UNIT_FIELDS = ("styleWeight", "weirdnessConstraint", "audioWeight")


class PlanError(ValueError):
    pass


def schema() -> dict[str, Any]:
    """The score execution manifest contract, read from the validator's own sets."""
    return {
        "manifest": "execution manifest; sourceDraft and outputDirectory sit below it",
        "requiredRootFields": ["version", "financialApproval", "currentPriceVerified",
                               "sourceDraft", "sourceDraftSha256", "cues", "creditsPerRequest",
                               "maxRequests", "maxCredits", "outputDirectory",
                               "expectedCandidatesPerRequest"],
        "optionalRootFields": ["episode", "pricing", "provenancePathBase", "projectRoot",
                               "productionOverrides", "productionOverridesSha256",
                               "authority", "authoritySha256", "creativeBindings",
                               "creditsPerExtendRequest"],
        "versions": {
            "1": "recorded manifests: exactly the nine payloadFields, model V5_5 or V6, "
                 "instrumental with an empty prompt, callBackUrl and non-empty negativeTags",
            "2": "what --freeze writes: exactly the keys frozen in the draft, each from its "
                 "route's allowlist below; cues may carry derivedFrom",
        },
        "cueFields": sorted(CUE_FIELDS),
        "optionalCueFieldsV2": [DERIVED_CUE_FIELD],
        "payloadFields": sorted(PAYLOAD_FIELDS),
        "payloadRules": {"model": ["V5_5", "V6"], "customMode": True, "instrumental": True,
                         "prompt": "", "duration": "integer 10-360", "style": "1-1000 characters",
                         "title": "1-80 characters", "negativeTags": "non-empty"},
        "payloadRoutesV2": {
            "legacy": "model V5_5 keeps the version 1 nine-key rule and posts to generate "
                      "with its required callBackUrl",
            **{route: {"models": sorted(V6_FAMILY), "required": sorted(fields["required"]),
                       "optional": sorted(fields["optional"])}
               for route, fields in ROUTE_FIELDS.items()},
        },
        "payloadRangesV2": {
            "customMode": True, "instrumental": "boolean, always frozen explicitly",
            "prompt": "the lyrics: empty or absent when instrumental; 1-5000 characters "
                      "and required on generate when instrumental is false",
            "title": "1-80 characters (extend 1-100)", "style": "1-1000 characters",
            "duration": "integer 10-360 (generate only)", "negativeTags": "non-empty if present",
            "styleWeight": "0-1, two decimals", "weirdnessConstraint": "0-1, two decimals",
            "audioWeight": "0-1, two decimals; not with instrumental",
            "variety": "integer 0-4", "vocalGender": "m or f; not with instrumental",
            "personaId": "passed through as given; this runner creates no persona",
            "personaModel": "style_persona or voice_persona",
            "callBackUrl": "optional https URL; sent when present. Omitting it on "
                           "jobs/createTask is unverified until one bounded live call",
        },
        "derivedFrom": {
            "shape": "{cueId, take, operation: extend, continueAt}",
            "rules": "the parent is an earlier cue of this manifest with the same V6-family "
                     "model; submit waits (WAITING_PARENT) until the parent is DOWNLOADED, "
                     "then posts with the parent take's real provider track id, refusing a "
                     "synthesized one; continueAt must fall inside the parent take",
            "pricing": "creditsPerExtendRequest is required when any cue extends",
            "sourceSeconds": "an extend requests no duration and adds 0 sourceSeconds",
            "retention": "the provider keeps generated files 14 days, which limits extending "
                         "a take across episodes; parents in other manifests are not supported",
        },
        "sungWords": "a sung cue's words are text in the film that the screenplay may not "
                     "contain: record whether they come from the source or were added, and "
                     "whether they are subtitled",
        "sourceDraft": "{commonPayload, cues}; frozen cues equal each draft cue with commonPayload merged in",
        "ceilings": "cues x their kind's price (creditsPerRequest, creditsPerExtendRequest) must "
                    "fit maxCredits; --max-requests/--max-credits may not exceed the manifest",
        "financialApproval": "boolean; reported, never sufficient: submission needs an approval record",
        "approvalLane": "suno-score",
        "approvalScope": ["requests", "credits", "sourceSeconds"],
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _short_text(value: Any, limit: int) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit and bool(value.strip())


def _validate_v1_payload(cue_id: str, payload: Any) -> None:
    """The version 1 rule, kept exactly as recorded manifests were validated."""
    required = PAYLOAD_FIELDS
    if not isinstance(payload, dict) or set(payload) != required:
        raise PlanError(f"{cue_id}: payload fields must be exactly {sorted(required)}")
    if payload["model"] not in {"V5_5", "V6"} or payload["customMode"] is not True:
        raise PlanError(f"{cue_id}: only V5_5 or V6 custom mode is allowed")
    if payload["instrumental"] is not True or payload["prompt"] != "":
        raise PlanError(f"{cue_id}: score must be instrumental with empty lyrics prompt")
    if not isinstance(payload["duration"], int) or not 10 <= payload["duration"] <= 360:
        raise PlanError(f"{cue_id}: duration must be an integer from 10 to 360")
    if not isinstance(payload["style"], str) or not 0 < len(payload["style"]) <= 1000:
        raise PlanError(f"{cue_id}: style must contain at most 1000 characters")
    if not isinstance(payload["title"], str) or not 0 < len(payload["title"]) <= 80:
        raise PlanError(f"{cue_id}: title must contain at most 80 characters")
    if not isinstance(payload["negativeTags"], str) or not payload["negativeTags"].strip():
        raise PlanError(f"{cue_id}: negativeTags must be non-empty")


def _validate_v6_payload(cue_id: str, payload: Any, route: str) -> None:
    """Exactly the keys frozen in the draft, each from the route's allowlist and range."""
    fields = ROUTE_FIELDS[route]
    if not isinstance(payload, dict):
        raise PlanError(f"{cue_id}: payload must be an object")
    keys = set(payload)
    missing = fields["required"] - keys
    unknown = keys - fields["required"] - fields["optional"]
    if missing or unknown:
        raise PlanError(f"{cue_id}: {route} payload is missing {sorted(missing)} or carries "
                        f"keys outside its allowlist {sorted(unknown)}")
    if payload["model"] not in V6_FAMILY:
        raise PlanError(f"{cue_id}: {route} needs a model from {sorted(V6_FAMILY)}")
    if "customMode" in payload and payload["customMode"] is not True:
        raise PlanError(f"{cue_id}: customMode must be true")
    instrumental = payload["instrumental"]
    if not isinstance(instrumental, bool):
        raise PlanError(f"{cue_id}: instrumental must be a boolean")
    lyrics = payload.get("prompt")
    if instrumental:
        if lyrics not in (None, "") or (route == "extend-v6" and "prompt" in payload):
            raise PlanError(f"{cue_id}: an instrumental cue carries no lyrics prompt")
        for key in ("vocalGender", "audioWeight"):
            if key in payload:
                raise PlanError(f"{cue_id}: {key} does not apply to an instrumental cue")
    elif lyrics is not None or route == "generate-v6":
        # A sung generate needs lyrics; a sung extend may continue from the audio alone.
        if not _short_text(lyrics, 5000):
            raise PlanError(f"{cue_id}: a sung cue freezes its authored lyrics (1-5000 "
                            "characters) in prompt")
    if "title" in payload and not _short_text(payload["title"], 80 if route == "generate-v6" else 100):
        raise PlanError(f"{cue_id}: title is empty or too long")
    if "style" in payload and not _short_text(payload["style"], 1000):
        raise PlanError(f"{cue_id}: style must contain 1-1000 characters")
    if "duration" in payload and (not isinstance(payload["duration"], int)
                                  or isinstance(payload["duration"], bool)
                                  or not 10 <= payload["duration"] <= 360):
        raise PlanError(f"{cue_id}: duration must be an integer from 10 to 360")
    if "negativeTags" in payload and not _short_text(payload["negativeTags"], 1000):
        raise PlanError(f"{cue_id}: negativeTags, when present, must be non-empty")
    for key in UNIT_FIELDS:
        if key in payload:
            value = payload[key]
            if not _is_number(value) or not 0 <= value <= 1 or round(value, 2) != value:
                raise PlanError(f"{cue_id}: {key} must be 0-1 with at most two decimals")
    if "variety" in payload and (not isinstance(payload["variety"], int)
                                 or isinstance(payload["variety"], bool)
                                 or not 0 <= payload["variety"] <= 4):
        raise PlanError(f"{cue_id}: variety must be an integer from 0 to 4")
    if "vocalGender" in payload and payload["vocalGender"] not in {"m", "f"}:
        raise PlanError(f"{cue_id}: vocalGender must be m or f")
    if "personaId" in payload and not _short_text(payload["personaId"], 200):
        raise PlanError(f"{cue_id}: personaId must be a non-empty id")
    if "personaModel" in payload and payload["personaModel"] not in {"style_persona", "voice_persona"}:
        raise PlanError(f"{cue_id}: personaModel must be style_persona or voice_persona")
    callback = payload.get("callBackUrl")
    if "callBackUrl" in payload and (not isinstance(callback, str)
                                     or not callback.startswith("https://")):
        raise PlanError(f"{cue_id}: callBackUrl, when present, must be an https URL")


def _validate_derived(cue: dict[str, Any], earlier: dict[str, dict[str, Any]],
                      expected_candidates: Any) -> None:
    cue_id, spec = cue["id"], cue[DERIVED_CUE_FIELD]
    if not isinstance(spec, dict) or set(spec) != {"cueId", "take", "operation", "continueAt"}:
        raise PlanError(f"{cue_id}: derivedFrom must be exactly {{cueId, take, operation, continueAt}}")
    if spec["operation"] not in DERIVED_OPERATIONS:
        raise PlanError(f"{cue_id}: derivedFrom operation must be one of {sorted(DERIVED_OPERATIONS)}")
    parent = earlier.get(spec["cueId"])
    if parent is None:
        raise PlanError(f"{cue_id}: derivedFrom must name an earlier cue of this manifest")
    take = spec["take"]
    if (not isinstance(take, int) or isinstance(take, bool) or take < 1
            or (isinstance(expected_candidates, int) and take > expected_candidates)):
        raise PlanError(f"{cue_id}: derivedFrom take must be one of the parent's takes")
    if not _is_number(spec["continueAt"]) or not spec["continueAt"] > 0:
        raise PlanError(f"{cue_id}: continueAt must be a positive number of seconds")
    model = cue["payload"].get("model")
    if model not in V6_FAMILY or parent["payload"].get("model") != model:
        raise PlanError(f"{cue_id}: an extend uses the parent's own V6-family model")


def _validate_v2_cues(cues: list[Any], expected_candidates: Any) -> None:
    earlier: dict[str, dict[str, Any]] = {}
    for cue in cues:
        if (not isinstance(cue, dict) or not CUE_FIELDS <= set(cue)
                or set(cue) - CUE_FIELDS - {DERIVED_CUE_FIELD}):
            raise PlanError("each cue must freeze id, shots, emotionalArc and payload, "
                            "and may add derivedFrom")
        cue_id = cue.get("id")
        if not isinstance(cue_id, str) or not SAFE_ID.fullmatch(cue_id) or cue_id in earlier:
            raise PlanError(f"duplicate or malformed cue id: {cue_id}")
        payload = cue.get("payload")
        if DERIVED_CUE_FIELD in cue:
            _validate_v6_payload(cue_id, payload, "extend-v6")
            _validate_derived(cue, earlier, expected_candidates)
        elif isinstance(payload, dict) and payload.get("model") == "V5_5":
            _validate_v1_payload(cue_id, payload)
        else:
            _validate_v6_payload(cue_id, payload, "generate-v6")
        earlier[cue_id] = cue


def cue_credits(doc: dict[str, Any], cue: dict[str, Any]) -> int:
    """The frozen price of one cue's single request, by kind."""
    if DERIVED_CUE_FIELD in cue:
        return doc["creditsPerExtendRequest"]
    return doc["creditsPerRequest"]


class UncertainRequest(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _now() -> float:
    return time.time()


def _persist(path: Path, state: dict[str, Any]) -> None:
    state["updatedAt"] = _now()
    _atomic_json(path, state)


def _relative_child(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PlanError(f"{label} must be a non-empty relative path")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise PlanError(f"{label} must stay below the execution manifest")
    return (root / rel).resolve()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _verified_binding(manifest_path: Path, doc: dict[str, Any],
                      path_key: str, sha_key: str) -> dict[str, str] | None:
    raw_path, raw_sha = doc.get(path_key), doc.get(sha_key)
    if raw_path is None and raw_sha is None:
        return None
    if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_sha, str):
        raise PlanError(f"{path_key} and {sha_key} must be supplied together")
    rel = Path(raw_path)
    if rel.is_absolute():
        raise PlanError(f"{path_key} must be relative")
    mode = doc.get("provenancePathBase")
    if mode == "project":
        value = doc.get("projectRoot")
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise PlanError("project provenance requires a relative projectRoot")
        base = (manifest_path.parent / value).resolve()
        if not base.is_dir():
            raise PlanError("project provenance root is missing")
    elif mode == "manifest":
        base = manifest_path.parent
    else:
        raise PlanError(
            "provenancePathBase must be 'project' or 'manifest' when provenance is declared"
        )
    path = (base / rel).resolve()
    if not _is_below(path, base) or not path.is_file():
        raise PlanError(f"{path_key} is missing or escapes its declared base: {path}")
    actual = _sha_file(path)
    if actual != raw_sha.lower():
        raise PlanError(f"{path_key} changed after the score manifest was frozen")
    return {"kind": path_key, "path": str(path), "sha256": actual}


def _verified_authority(manifest_path: Path, doc: dict[str, Any]) -> dict[str, Any] | None:
    """Check authority and creative bindings at load time, so the dry run finds a change.

    Paths resolve from the manifest's directory, exactly as run_score_gpu.py
    always resolved them. A manifest without these fields is unchanged.
    """
    raw_path, raw_sha = doc.get("authority"), doc.get("authoritySha256")
    result: dict[str, Any] = {}
    if raw_path is not None or raw_sha is not None:
        if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_sha, str):
            raise PlanError("authority and authoritySha256 must be supplied together")
        path = (manifest_path.parent / raw_path).resolve()
        if not path.is_file() or _sha_file(path) != raw_sha:
            raise PlanError(f"authority changed after the score manifest was frozen: {path}")
        result["authority"] = {"path": str(path), "sha256": raw_sha}
    bindings = doc.get("creativeBindings")
    if bindings is not None:
        if not isinstance(bindings, list):
            raise PlanError("creativeBindings must be a list of {path, sha256}")
        for position, binding in enumerate(bindings):
            if (not isinstance(binding, dict) or not isinstance(binding.get("path"), str)
                    or not isinstance(binding.get("sha256"), str)):
                raise PlanError(f"creativeBindings[{position}] must carry path and sha256")
            path = (manifest_path.parent / binding["path"]).resolve()
            if not path.is_file() or _sha_file(path) != binding["sha256"]:
                raise PlanError(f"creative/timing input changed after freezing: {binding['path']}")
        result["creativeBindings"] = len(bindings)
    return result or None


def load_plan(manifest_path: Path, max_requests: int,
              max_credits: int) -> dict[str, Any]:
    if max_requests <= 0 or max_credits <= 0:
        raise PlanError("--max-requests and --max-credits must be positive")
    manifest_path = manifest_path.resolve()
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read execution manifest: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("version") not in MANIFEST_VERSIONS:
        raise PlanError("execution manifest version must be 1 or 2")
    version = doc["version"]
    if not isinstance(doc.get("financialApproval"), bool):
        raise PlanError("financialApproval must be an explicit boolean")
    if not isinstance(doc.get("currentPriceVerified"), bool):
        raise PlanError("currentPriceVerified must be an explicit boolean")
    root = manifest_path.parent
    source_path = _relative_child(root, doc.get("sourceDraft"), "sourceDraft")
    if not source_path.is_file():
        raise PlanError(f"source draft is missing: {source_path}")
    source_sha = _sha_file(source_path)
    if source_sha != doc.get("sourceDraftSha256"):
        raise PlanError("source draft changed after execution manifest was frozen")
    production_binding = _verified_binding(
        manifest_path, doc, "productionOverrides", "productionOverridesSha256"
    )
    authority = _verified_authority(manifest_path, doc)
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"source draft is invalid JSON: {exc}") from exc
    cues = doc.get("cues")
    if not isinstance(cues, list) or not cues:
        raise PlanError("cues must be a non-empty array")
    source_common = source.get("commonPayload")
    source_cues = source.get("cues")
    if not isinstance(source_common, dict) or not isinstance(source_cues, list):
        raise PlanError("source draft lacks commonPayload or cues")
    materialized_source = [
        {**cue, "payload": {**source_common, **cue.get("payload", {})}}
        for cue in source_cues
    ]
    if cues != materialized_source:
        raise PlanError("frozen cues are not byte-semantically equal to source draft cues")
    if version == 1:
        ids: set[str] = set()
        for cue in cues:
            if not isinstance(cue, dict) or set(cue) != CUE_FIELDS:
                raise PlanError("each cue must freeze id, shots, emotionalArc and payload")
            cue_id = cue.get("id")
            if not isinstance(cue_id, str) or not SAFE_ID.fullmatch(cue_id) or cue_id in ids:
                raise PlanError(f"duplicate or malformed cue id: {cue_id}")
            ids.add(cue_id)
            _validate_v1_payload(cue_id, cue.get("payload"))
    else:
        _validate_v2_cues(cues, doc.get("expectedCandidatesPerRequest"))
    credits_each = doc.get("creditsPerRequest")
    manifest_requests = doc.get("maxRequests")
    manifest_credits = doc.get("maxCredits")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
               for x in (credits_each, manifest_requests, manifest_credits)):
        raise PlanError("creditsPerRequest/maxRequests/maxCredits must be positive integers")
    extends = [cue for cue in cues if DERIVED_CUE_FIELD in cue]
    extend_price = doc.get("creditsPerExtendRequest")
    if version == 1 and "creditsPerExtendRequest" in doc:
        raise PlanError("creditsPerExtendRequest needs a version 2 manifest")
    if "creditsPerExtendRequest" in doc or extends:
        if not isinstance(extend_price, int) or isinstance(extend_price, bool) or extend_price <= 0:
            raise PlanError("an extend cue needs creditsPerExtendRequest, a positive integer "
                            "you verified for the extend route")
    request_count = len(cues)
    expected_credits = sum(cue_credits(doc, cue) for cue in cues)
    if request_count > manifest_requests or expected_credits > manifest_credits:
        raise PlanError("frozen cue count exceeds manifest ceiling")
    if request_count > max_requests or expected_credits > max_credits:
        raise PlanError(
            f"plan needs {request_count} requests/{expected_credits} credits, exceeding "
            f"explicit CLI caps {max_requests}/{max_credits}")
    if max_requests > manifest_requests or max_credits > manifest_credits:
        raise PlanError("CLI caps may not exceed the immutable manifest ceilings")
    output_root = _relative_child(root, doc.get("outputDirectory"), "outputDirectory")
    expected_candidates = doc.get("expectedCandidatesPerRequest")
    if (not isinstance(expected_candidates, int) or isinstance(expected_candidates, bool)
            or expected_candidates <= 0 or expected_candidates > 8):
        raise PlanError("expectedCandidatesPerRequest must be an integer from 1 to 8")
    fingerprint = _sha_file(manifest_path)
    return {"path": str(manifest_path), "sha256": fingerprint, "doc": doc,
            "sourceSha256": source_sha, "cues": cues, "outputRoot": str(output_root),
            "productionBinding": production_binding, "authority": authority,
            "expectedCandidatesPerRequest": expected_candidates,
            "requestCount": request_count, "expectedCredits": expected_credits,
            "maxRequests": max_requests, "maxCredits": max_credits}


def _state_for(plan: dict[str, Any]) -> dict[str, Any]:
    return {"version": STATE_VERSION, "executionManifest": plan["path"],
            "executionManifestSha256": plan["sha256"],
            "approvalAdmissionSha256": plan.get("approvalAdmissionSha256"),
            "approvalAuthorizationId": plan.get("approvalAuthorizationId"),
            "sourceDraftSha256": plan["sourceSha256"],
            "productionOverridesSha256": (
                plan["productionBinding"]["sha256"] if plan["productionBinding"] else None
            ),
            "maxRequests": plan["maxRequests"], "maxCredits": plan["maxCredits"],
            "expectedCredits": plan["expectedCredits"], "createdAt": _now(),
            "updatedAt": _now(), "walletQueried": False, "topUpAttempted": False,
            "items": {cue["id"]: {"status": "PLANNED", "postAttempts": 0,
                                       "requestSha256": _sha_bytes(_canonical(cue["payload"]))}
                      for cue in plan["cues"]}}


def _load_state(plan: dict[str, Any], path: Path, create: bool) -> dict[str, Any]:
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(f"cannot read durable state: {exc}") from exc
    elif create:
        state = _state_for(plan)
        _atomic_json(path, state)
    else:
        raise PlanError(f"durable state does not exist: {path}")
    expected = {x["id"] for x in plan["cues"]}
    if state.get("version") != STATE_VERSION or set(state.get("items", {})) != expected:
        raise PlanError("durable state schema/item set mismatch")
    checks = {"executionManifestSha256": plan["sha256"],
              "approvalAdmissionSha256": plan.get("approvalAdmissionSha256"),
              "approvalAuthorizationId": plan.get("approvalAuthorizationId"),
              "sourceDraftSha256": plan["sourceSha256"],
              "productionOverridesSha256": (
                  plan["productionBinding"]["sha256"] if plan["productionBinding"] else None
              ),
              "maxRequests": plan["maxRequests"], "maxCredits": plan["maxCredits"]}
    for key, value in checks.items():
        if state.get(key) != value:
            raise PlanError(f"durable state mismatch: {key}")
    return state


def _request_json(method: str, url: str, key: str,
                  payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=_canonical(payload) if payload is not None else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "User-Agent": "agentic-animation-finite-score/1"}, method=method)
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read())
    if result.get("code") != 200 or not isinstance(result.get("data"), dict):
        raise RuntimeError("provider returned a non-success envelope")
    return result["data"]


def provider_body(payload: dict[str, Any]) -> dict[str, Any]:
    """The jobs/createTask body for a V6-family request; an audioId makes it an extend.

    callBackUrl is sent only when frozen, so a version 1 V6 body is unchanged.
    """
    body: dict[str, Any] = {"model": "ai-music-api/extend" if "audioId" in payload
                            else "ai-music-api/generate"}
    if "callBackUrl" in payload:
        body["callBackUrl"] = payload["callBackUrl"]
    body["input"] = {V6_INPUT_NAMES.get(k, k): v for k, v in payload.items()
                     if k != "callBackUrl"}
    return body


def _create(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if payload["model"] in V6_FAMILY:
        return _request_json("POST", API_ROOT + "jobs/createTask", key, provider_body(payload))
    return _request_json("POST", API_ROOT + "generate", key, payload)


def _record(task_id: str, key: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"taskId": task_id})
    return _request_json("GET", API_ROOT + "generate/record-info?" + query, key)


def _record_v6(task_id: str, key: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"taskId": task_id})
    raw = _request_json("GET", API_ROOT + "jobs/recordInfo?" + query, key)
    state = str(raw.get("state", "")).lower()
    result = {"status": {"success": "SUCCESS", "fail": "FAILED"}.get(state, "PENDING"),
              "providerRecord": raw}
    if state == "success":
        payload = raw.get("resultJson") or "{}"
        data = json.loads(payload) if isinstance(payload, str) else payload
        tracks = data.get("data")
        if isinstance(tracks, list) and tracks:
            if any(not isinstance(t, dict) or not isinstance(t.get("audio_url"), str)
                   or not t["audio_url"] for t in tracks):
                raise PlanError("V6 candidate is missing audio_url; retain task for reconciliation")
            result["response"] = {"sunoData": [{**t, "audioUrl": t["audio_url"]} for t in tracks]}
            return result
        urls = data.get("resultUrls") or []
        if not isinstance(urls, list) or not urls or any(not isinstance(u, str) for u in urls):
            raise PlanError("V6 success lacks valid resultUrls; retain task for reconciliation")
        # No provider track id came back. The placeholder keeps takes distinct,
        # and idSynthesized stops it from ever being sent as a parent audio_id.
        result["response"] = {"sunoData": [{"audioUrl": u, "id": f"{task_id}-{i}",
                                            "idSynthesized": True}
                                           for i, u in enumerate(urls, 1)]}
    return result


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _validate_mp3(path: Path, requested_seconds: int | None) -> dict[str, Any]:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of",
             "json", str(path)], check=True, capture_output=True, text=True)
        info = json.loads(probe.stdout)
        audio = [x for x in info.get("streams", []) if x.get("codec_type") == "audio"]
        duration = float((info.get("format") or {}).get("duration") or 0)
        # An extend requests no duration; the provider chooses its length.
        minimum = max(5.0, (requested_seconds or 0) * 0.25)
        if not audio or duration < minimum:
            raise PlanError("candidate has no non-trivial audio stream")
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                        "-f", "null", "-"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError, ValueError,
            json.JSONDecodeError) as exc:
        raise PlanError(f"candidate MP3 failed decode: {exc}") from exc
    return {"durationSeconds": duration, "audioStreams": len(audio),
            "codec": audio[0].get("codec_name"),
            "requestedDurationSeconds": requested_seconds,
            "minimumAcceptedDurationSeconds": minimum}


def _completed_ok(plan: dict[str, Any], cue_id: str,
                  record: dict[str, Any]) -> None:
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PlanError(f"{cue_id}: downloaded state has no candidates")
    for candidate in candidates:
        path = Path(candidate["file"])
        sidecar = Path(candidate["sidecar"])
        if not path.is_file() or _sha_file(path) != candidate.get("sha256"):
            raise PlanError(f"{cue_id}: candidate missing or hash changed: {path}")
        if not sidecar.is_file() or _sha_file(sidecar) != candidate.get("sidecarSha256"):
            raise PlanError(f"{cue_id}: sidecar missing or hash changed: {sidecar}")
        try:
            source = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(f"{cue_id}: sidecar is invalid: {sidecar}") from exc
        expected = {
            "cueId": cue_id,
            "executionManifestSha256": plan["sha256"],
            "sourceDraftSha256": plan["sourceSha256"],
            "sha256": candidate.get("sha256"),
        }
        for key, value in expected.items():
            if source.get(key) != value:
                raise PlanError(f"{cue_id}: sidecar provenance mismatch: {key}")


def _preflight_output_namespace(plan: dict[str, Any], state: dict[str, Any]) -> None:
    """Reject every stale/colliding target before a paid POST or provider poll."""
    output_root = Path(plan["outputRoot"])
    for cue in plan["cues"]:
        cue_id = cue["id"]
        record = state["items"][cue_id]
        existing = set(output_root.glob(f"{cue_id}-take-*")) if output_root.exists() else set()
        if record.get("status") == "DOWNLOADED":
            _completed_ok(plan, cue_id, record)
            allowed = {
                Path(value)
                for candidate in record.get("candidates", [])
                for value in (candidate.get("file"), candidate.get("sidecar"))
                if value
            }
            unexpected = {path.resolve() for path in existing} - {path.resolve() for path in allowed}
            if unexpected:
                raise PlanError(f"{cue_id}: untracked output namespace entries: {sorted(map(str, unexpected))}")
            continue
        if existing:
            raise PlanError(
                f"{cue_id}: output namespace is not empty before paid work: "
                f"{sorted(map(str, existing))}"
            )


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PlanError(f"execution lock exists; verify no runner is active: {path}") from exc
    os.write(fd, f"pid={os.getpid()}\n".encode())
    os.fsync(fd)
    os.close(fd)
    return fd


def _parent_request(plan: dict[str, Any], state: dict[str, Any],
                    cue: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(request, parent binding) for an extend whose parent has landed; (None, None) while it waits.

    The parent is bound by its sidecar hash and the provider's own track and
    task ids. A synthesized track id is refused before anything is posted.
    """
    spec = cue[DERIVED_CUE_FIELD]
    parent = state["items"][spec["cueId"]]
    if parent.get("status") != "DOWNLOADED":
        return None, None
    _completed_ok(plan, spec["cueId"], parent)
    takes = [row for row in parent.get("candidates", []) if row.get("index") == spec["take"]]
    if len(takes) != 1:
        raise PlanError(f"{cue['id']}: parent {spec['cueId']} landed no take {spec['take']}")
    take = takes[0]
    track = take.get("providerMetadata") or {}
    audio_id = track.get("id")
    if (not isinstance(audio_id, str) or not audio_id or track.get("idSynthesized")
            or audio_id == f"{parent.get('taskId')}-{spec['take']}"):
        raise PlanError(f"{cue['id']}: parent take carries no provider track id, only a "
                        "synthesized one; an extend cannot name it")
    seconds = (take.get("media") or {}).get("durationSeconds")
    if not _is_number(seconds) or not spec["continueAt"] < seconds:
        raise PlanError(f"{cue['id']}: continueAt {spec['continueAt']} falls outside the "
                        f"parent take ({seconds} s)")
    binding = {**spec, "sidecar": take["sidecar"], "sidecarSha256": take["sidecarSha256"],
               "sha256": take["sha256"], "taskId": parent["taskId"], "audioId": audio_id,
               "model": cue["payload"]["model"]}
    request = {**cue["payload"], "audioId": audio_id, "continueAt": spec["continueAt"]}
    return request, binding


def submit(plan: dict[str, Any], state_path: Path, key: str,
           create_fn: Callable[[dict[str, Any], str], dict[str, Any]]) -> dict[str, Any]:
    if plan.get("financialAuthorized") is not True:
        raise PlanError("submission requires a current approval record (--approval-admission); "
                        "the manifest's financialApproval never authorizes spending")
    if plan["doc"].get("currentPriceVerified") is not True:
        raise PlanError("currentPriceVerified is false; credit ceiling is not authorized")
    state = _load_state(plan, state_path, create=True)
    _preflight_output_namespace(plan, state)
    # One cue goes first and must land before the rest are posted, so a wrong
    # price or a refused payload costs one request, not the batch.
    statuses = [state["items"][cue["id"]].get("status") for cue in plan["cues"]]
    first_landed = "DOWNLOADED" in statuses
    allowed = len(statuses) if first_landed else 0 if any(
        status != "PLANNED" for status in statuses) else 1
    # Every extend whose parent has landed is resolved before the first POST,
    # so a parent without a provider track id refuses the run, not half of it.
    derived = {cue["id"]: _parent_request(plan, state, cue) for cue in plan["cues"]
               if DERIVED_CUE_FIELD in cue
               and state["items"][cue["id"]].get("status") in {"PLANNED", "WAITING_PARENT"}}
    calls = 0
    for cue in plan["cues"]:
        record = state["items"][cue["id"]]
        status = record.get("status")
        if status == "DOWNLOADED":
            _completed_ok(plan, cue["id"], record)
            continue
        if status == "SUBMITTED" or status == "PENDING" or status == "RESULT_READY":
            continue
        if status not in {"PLANNED", "WAITING_PARENT"}:
            raise UncertainRequest(f"{cue['id']}: {status}; generation POST will not be retried")
        request, parent = derived.get(cue["id"], (cue["payload"], None))
        if request is None:
            # The parent has not landed: never post, and say so in the state.
            if status != "WAITING_PARENT":
                record["status"] = "WAITING_PARENT"
                _persist(state_path, state)
            continue
        if calls >= allowed:
            continue
        record.update(status="POSTING_UNCERTAIN", postAttempts=1, postStartedAt=_now())
        if parent is not None:
            record.update(parent=parent, postedRequestSha256=_sha_bytes(_canonical(request)))
        _persist(state_path, state)
        calls += 1
        try:
            response = create_fn(request, key)
        except urllib.error.HTTPError as exc:
            status = "HTTP_4XX_REFUSED" if 400 <= exc.code < 500 else "UNCERTAIN_HTTP_ERROR"
            record.update(status=status, httpStatus=exc.code, postFinishedAt=_now())
            _persist(state_path, state)
            error = PlanError if status == "HTTP_4XX_REFUSED" else UncertainRequest
            raise error(f"{cue['id']}: POST failed HTTP {exc.code}; automatic retry forbidden") from exc
        except Exception as exc:
            record.update(status="UNCERTAIN_POST", postFinishedAt=_now(),
                          error=f"{type(exc).__name__}: {exc}")
            _persist(state_path, state)
            raise UncertainRequest(f"{cue['id']}: ambiguous POST; reconcile task logs") from exc
        task_id = response.get("taskId")
        if not isinstance(task_id, str) or not task_id:
            record.update(status="UNCERTAIN_RESPONSE", postFinishedAt=_now())
            _persist(state_path, state)
            raise UncertainRequest(f"{cue['id']}: success response lacked taskId")
        record.update(status="SUBMITTED", taskId=task_id, postFinishedAt=_now(),
                      providerCreateResponse=response)
        _persist(state_path, state)
    planned = sum(state["items"][cue["id"]].get("status") == "PLANNED" for cue in plan["cues"])
    waiting = [cue["id"] for cue in plan["cues"]
               if state["items"][cue["id"]].get("status") == "WAITING_PARENT"]
    result = {"operation": "SUBMIT", "generationPosts": calls, "stillPlanned": planned,
              "state": str(state_path), "expectedCredits": plan["expectedCredits"]}
    if planned:
        result["note"] = "the first cue must land (--poll) before --submit posts the rest"
    if waiting:
        result["waitingParent"] = waiting
        result["stillPlanned"] = planned + len(waiting)
    return result


def _tracks(result: dict[str, Any]) -> list[dict[str, Any]]:
    tracks = (result.get("response") or {}).get("sunoData") or []
    if not isinstance(tracks, list) or not tracks:
        raise PlanError("successful task contains no Suno candidates")
    if any(not isinstance(x, dict) or not isinstance(x.get("audioUrl"), str)
           or not x["audioUrl"] for x in tracks):
        raise PlanError("provider candidate is missing audioUrl")
    return tracks


def poll(plan: dict[str, Any], state_path: Path, key: str,
         record_fn: Callable[[str, str], dict[str, Any]],
         download_fn: Callable[[str], bytes]) -> dict[str, Any]:
    state = _load_state(plan, state_path, create=False)
    _preflight_output_namespace(plan, state)
    output_root = Path(plan["outputRoot"])
    polls = downloads = planned = 0
    for cue in plan["cues"]:
        item = state["items"][cue["id"]]
        status = item.get("status")
        if status == "GENERATION_FAILED" and item.get("lastProviderStatus") in IN_PROGRESS:
            # Repair only the former poller's known classification error. The
            # original task ID and provider response survive; no POST is made.
            item["status"] = status = "PENDING"
            item["statusCorrection"] = "provider intermediate success was misclassified as failure"
            _persist(state_path, state)
        if status == "DOWNLOADED":
            _completed_ok(plan, cue["id"], item)
            continue
        if status in {"PLANNED", "WAITING_PARENT"}:
            # Not posted yet: submit posts it once the first cue (or its parent) has landed.
            planned += 1
            continue
        if status in {"POSTING_UNCERTAIN", "UNCERTAIN_POST",
                      "UNCERTAIN_RESPONSE", "UNCERTAIN_HTTP_ERROR",
                      "HTTP_4XX_REFUSED", "GENERATION_FAILED"}:
            raise UncertainRequest(f"{cue['id']}: cannot poll state {status}")
        if status in {"SUBMITTED", "PENDING", "POLL_FAILED_RETRYABLE"}:
            try:
                result = record_fn(item["taskId"], key)
                polls += 1
            except Exception as exc:
                item.update(status="POLL_FAILED_RETRYABLE",
                            pollError=f"{type(exc).__name__}: {exc}")
                _persist(state_path, state)
                raise PlanError(f"{cue['id']}: poll failed; current run stopped") from exc
            provider_status = str(result.get("status") or "").upper()
            item.update(lastProviderResponse=result, lastProviderStatus=provider_status,
                        lastPolledAt=_now())
            if provider_status in IN_PROGRESS:
                item["status"] = "PENDING"
                _persist(state_path, state)
                continue
            if provider_status not in SUCCESS:
                item["status"] = "GENERATION_FAILED"
                _persist(state_path, state)
                raise PlanError(f"{cue['id']}: provider generation failed: {provider_status}")
            tracks = _tracks(result)
            item.update(status="RESULT_READY", providerCandidates=tracks,
                        candidates=[])
            _persist(state_path, state)
        if item.get("status") != "RESULT_READY":
            raise PlanError(f"{cue['id']}: unsupported poll state {item.get('status')}")
        existing = {x["index"]: x for x in item.get("candidates", [])}
        for index, track in enumerate(item["providerCandidates"], 1):
            if index in existing:
                candidate = existing[index]
                if candidate.get("status") == "COMPLETE":
                    continue
                raise PlanError(f"{cue['id']} take {index}: incomplete capture needs reconciliation")
            try:
                raw = download_fn(track["audioUrl"])
                downloads += 1
            except Exception as exc:
                item["downloadError"] = f"take {index}: {type(exc).__name__}: {exc}"
                _persist(state_path, state)
                raise PlanError(f"{cue['id']} take {index}: download failed; run stopped") from exc
            response_sha = _sha_bytes(raw)
            response_root = state_path.parent / f".{state_path.stem}-responses"
            staged = response_root / f"{cue['id']}-take-{index}-{response_sha[:16]}.mp3"
            if not staged.exists():
                _atomic_bytes_new(staged, raw)
            candidate = {"index": index, "status": "CAPTURED", "responseSha256": response_sha,
                         "staged": str(staged), "providerMetadata": track}
            item.setdefault("candidates", []).append(candidate)
            _persist(state_path, state)
            duration = cue["payload"].get("duration")
            media = _validate_mp3(staged, int(duration) if duration is not None else None)
            candidate.update(status="VALIDATED", media=media)
            _persist(state_path, state)
            output = output_root / f"{cue['id']}-take-{index}.mp3"
            sidecar = output.with_suffix(".source.json")
            if output.exists() or sidecar.exists():
                raise PlanError(f"refusing to overwrite candidate or sidecar: {output}")
            _atomic_bytes_new(output, raw)
            sidecar_doc = {"version": 1, "episode": plan["doc"].get("episode"),
                           "cueId": cue["id"], "candidateIndex": index,
                           "taskId": item["taskId"], "request": cue["payload"],
                           "requestSha256": item["requestSha256"],
                           "executionManifest": plan["path"],
                           "executionManifestSha256": plan["sha256"],
                           "approvalAdmissionSha256": plan.get("approvalAdmissionSha256"),
                           "approvalAuthorizationId": plan.get("approvalAuthorizationId"),
                           "sourceDraftSha256": plan["sourceSha256"],
                           "productionOverridesSha256": (
                               plan["productionBinding"]["sha256"]
                               if plan["productionBinding"] else None
                           ),
                           "providerMetadata": track, "media": media,
                           "file": str(output), "sha256": response_sha,
                           "selectionVerdict": "unreviewed"}
            if "parent" in item:
                sidecar_doc["derivedFrom"] = item["parent"]
            _atomic_bytes_new(
                sidecar,
                (json.dumps(sidecar_doc, ensure_ascii=False, sort_keys=True, indent=2)
                 + "\n").encode("utf-8"),
            )
            candidate.update(status="COMPLETE", file=str(output), sha256=response_sha,
                             sidecar=str(sidecar), sidecarSha256=_sha_file(sidecar))
            _persist(state_path, state)
        item["status"] = "DOWNLOADED"
        item["downloadedAt"] = _now()
        _persist(state_path, state)
    return {"operation": "POLL", "recordGets": polls, "candidateDownloads": downloads,
            "notYetPosted": planned, "state": str(state_path)}


APPROVAL_LANE = "suno-score"


def record_route(plan: dict[str, Any]) -> Callable[[str, str], dict[str, Any]]:
    """The poll endpoint for a batch: every V6-family model polls jobs/recordInfo."""
    models = {cue["payload"]["model"] for cue in plan["cues"]}
    if len(models) != 1:
        raise PlanError("Each score batch must use one model for unambiguous recovery")
    return _record_v6 if models <= V6_FAMILY else _record


def approval_scope(plan: dict[str, Any]) -> dict[str, Any]:
    """The finite execution limits an approval record must bind exactly.

    sourceSeconds sums each cue's requested duration; an extend requests none.
    """
    return {"requests": plan["requestCount"], "credits": plan["expectedCredits"],
            "sourceSeconds": sum(x["payload"].get("duration", 0) for x in plan["cues"])}


def execute(manifest_path: Path, max_requests: int, max_credits: int,
            operation: str = "dry-run", state_path: Path | None = None,
            create_fn: Callable | None = None, record_fn: Callable | None = None,
            download_fn: Callable | None = None, api_key: str | None = None,
            approval_admission: Path | None = None) -> dict[str, Any]:
    plan = load_plan(manifest_path, max_requests, max_credits)
    try:
        approval = validate_paid_approval(
            approval_admission, episode=plan["doc"].get("episode"), lane=APPROVAL_LANE,
            manifest_path=Path(plan["path"]), manifest_sha256=plan["sha256"],
            finite_execution=approval_scope(plan),
        )
    except PaidApprovalError as exc:
        raise PlanError(str(exc)) from exc
    # The manifest's financialApproval is reported, never trusted on its own.
    # Polling needs no new record; it checks the one the batch state recorded.
    plan["financialAuthorized"] = approval is not None
    plan["approvalAdmissionSha256"] = approval["sha256"] if approval else None
    plan["approvalAuthorizationId"] = approval["authorizationId"] if approval else None
    if operation == "dry-run":
        usd = usd_per_credit()
        return {"operation": "DRY_RUN", "networkCalls": 0,
                "estimatedCredits": plan["expectedCredits"],
                "estimatedUsd": round(plan["expectedCredits"] * usd, 4) if usd is not None else None,
                "usdPerCredit": usd,
                "financialApproval": plan["doc"].get("financialApproval") is True,
                "financialAuthorized": plan["financialAuthorized"],
                "approvalAdmissionSha256": plan["approvalAdmissionSha256"],
                "requests": plan["requestCount"], "expectedCredits": plan["expectedCredits"],
                "maxRequests": max_requests, "maxCredits": max_credits,
                "sourceSeconds": approval_scope(plan)["sourceSeconds"]}
    if operation not in {"submit", "poll"}:
        raise PlanError(f"unknown operation: {operation}")
    if operation == "submit" and not plan["financialAuthorized"]:
        # Refuse at the immutable plan gate before looking for a credential,
        # creating state, acquiring a lock, or touching the network.
        raise PlanError("submission requires a current approval record (--approval-admission); "
                        "the manifest's financialApproval never authorizes spending")
    if operation == "submit" and plan["doc"].get("currentPriceVerified") is not True:
        raise PlanError("currentPriceVerified is false; credit ceiling is not authorized")
    default_state = Path(plan["path"]).with_suffix(".state.json").resolve()
    state_path = (state_path or default_state).resolve()
    if state_path != default_state:
        raise PlanError(f"custom state paths are forbidden; use canonical {default_state}")
    lock = Path(plan["path"]).with_suffix(".execution.lock")
    _acquire_lock(lock)
    try:
        try:
            key = api_key if api_key is not None else env_key("KIE_KEY")
        except (SystemExit, OSError, ValueError) as exc:
            raise PlanError(str(exc)) from exc
        if not key:
            raise PlanError("KIE_KEY is missing from the explicit credential sources")
        if operation == "submit":
            return submit(plan, state_path, key, create_fn or _create)
        return poll(plan, state_path, key, record_fn or record_route(plan),
                    download_fn or _download)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


UTC_SECOND = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


def _new_bytes(path: Path, raw: bytes) -> None:
    try:
        _atomic_bytes_new(path, raw)
    except FileExistsError as exc:
        raise PlanError(f"refusing to overwrite {path}") from exc


def freeze(draft_path: Path, manifest_path: Path, *, credits_per_request: int,
           price_verified_at: str, price_evidence: str, authority_path: Path,
           output_directory: str = "sources", expected_candidates: int = 2,
           episode: str | None = None,
           credits_per_extend_request: int | None = None) -> dict[str, Any]:
    """Write an execution manifest from a {commonPayload, cues} draft, then dry-run it.

    The cue list is the draft's cues with commonPayload merged in, exactly what
    load_plan compares. The price is the operator's own live reading; the tool
    never supplies a rate. financialApproval is always false: only a separate
    approval record, drafted after this freeze, authorizes spending. It writes
    a version 2 manifest; a draft with extend cues also needs the extend price.
    """
    manifest_path = manifest_path.resolve()
    if manifest_path.exists():
        raise PlanError(f"refusing to overwrite a frozen manifest: {manifest_path}")
    if (isinstance(credits_per_request, bool) or not isinstance(credits_per_request, int)
            or credits_per_request <= 0):
        raise PlanError("--credits-per-request must be the positive whole rate you verified today")
    if not isinstance(price_verified_at, str) or not UTC_SECOND.fullmatch(price_verified_at):
        raise PlanError("--price-verified-at must be a UTC time such as 2026-01-01T12:00:00Z")
    if not isinstance(price_evidence, str) or not price_evidence.strip():
        raise PlanError("--price-evidence must say where the live price was read")
    root = manifest_path.parent
    draft_path = draft_path.resolve()
    try:
        source = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read score draft: {exc}") from exc
    common, cues = source.get("commonPayload"), source.get("cues")
    if not isinstance(common, dict) or not isinstance(cues, list) or not cues:
        raise PlanError("score draft needs commonPayload and a non-empty cues list")
    root.mkdir(parents=True, exist_ok=True)
    if not _is_below(draft_path, root):
        copy = root / "creative-draft.json"
        if copy.exists() and copy.read_bytes() != draft_path.read_bytes():
            raise PlanError(f"{copy} already holds a different draft")
        if not copy.exists():
            _new_bytes(copy, draft_path.read_bytes())
        draft_path = copy
    authority_path = authority_path.resolve()
    if not authority_path.is_file():
        raise PlanError(f"authority file is missing: {authority_path}")
    materialized = [{**cue, "payload": {**common, **cue.get("payload", {})}} for cue in cues]
    extends = sum(isinstance(cue, dict) and DERIVED_CUE_FIELD in cue for cue in materialized)
    if extends and (isinstance(credits_per_extend_request, bool)
                    or not isinstance(credits_per_extend_request, int)
                    or credits_per_extend_request <= 0):
        raise PlanError("--credits-per-extend-request must be the positive whole extend rate "
                        "you verified today")
    pricing = {"creditsPerRequest": credits_per_request, "verifiedAtUtc": price_verified_at,
               "evidence": price_evidence.strip()}
    if extends:
        pricing["creditsPerExtendRequest"] = credits_per_extend_request
    total = (len(materialized) - extends) * credits_per_request + extends * (
        credits_per_extend_request or 0)
    doc = {
        "version": 2, "episode": episode or source.get("episode"),
        "financialApproval": False, "currentPriceVerified": True,
        "pricing": pricing,
        "sourceDraft": draft_path.relative_to(root).as_posix(),
        "sourceDraftSha256": _sha_file(draft_path),
        "authority": os.path.relpath(authority_path, root),
        "authoritySha256": _sha_file(authority_path),
        "creditsPerRequest": credits_per_request, "maxRequests": len(materialized),
        "maxCredits": total,
        **({"creditsPerExtendRequest": credits_per_extend_request} if extends else {}),
        "outputDirectory": output_directory,
        "expectedCandidatesPerRequest": expected_candidates,
        "cues": materialized,
    }
    _new_bytes(manifest_path, (json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2)
                               + "\n").encode("utf-8"))
    try:
        return execute(manifest_path, doc["maxRequests"], doc["maxCredits"])
    except PlanError:
        manifest_path.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Cue fields: " + ", ".join(sorted(CUE_FIELDS)) + " (version 2 adds "
                + DERIVED_CUE_FIELD + "). Version 1 payload fields: "
                + ", ".join(sorted(PAYLOAD_FIELDS)) + ". Version 2 generate fields: "
                + ", ".join(sorted(ROUTE_FIELDS["generate-v6"]["required"]
                                   | ROUTE_FIELDS["generate-v6"]["optional"]))
                + ". Run --print-schema for the whole contract and ranges; the approval "
                "lane is suno-score."))
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-credits", type=int)
    parser.add_argument("--print-schema", action="store_true",
                        help="print the score manifest contract and exit")
    parser.add_argument("--freeze", type=Path, metavar="DRAFT",
                        help="write MANIFEST from a {commonPayload, cues} draft, then dry-run it")
    freezing = parser.add_argument_group("freeze inputs (the price is your own live reading)")
    freezing.add_argument("--credits-per-request", type=int)
    freezing.add_argument("--credits-per-extend-request", type=int,
                          help="the extend route's own rate; needed when a cue extends another")
    freezing.add_argument("--price-verified-at", help="UTC time you read the provider's price")
    freezing.add_argument("--price-evidence", help="where you read it: page URL and model")
    freezing.add_argument("--authority", type=Path, help="the production authority file to bind")
    freezing.add_argument("--output-directory", default="sources",
                          help="candidate folder below the manifest (default sources)")
    freezing.add_argument("--expected-candidates", type=int, default=2,
                          help="takes the provider returns per request (default 2)")
    freezing.add_argument("--episode", help="project code when the draft names none")
    parser.add_argument("--approval-admission", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--poll", action="store_true")
    args = parser.parse_args(argv)
    if args.print_schema:
        print(json.dumps(schema(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.freeze is not None:
        if args.manifest is None or args.authority is None:
            parser.error("--freeze DRAFT needs the MANIFEST path to write and --authority")
        try:
            result = freeze(args.freeze, args.manifest,
                            credits_per_request=args.credits_per_request,
                            price_verified_at=args.price_verified_at,
                            price_evidence=args.price_evidence, authority_path=args.authority,
                            output_directory=args.output_directory,
                            expected_candidates=args.expected_candidates, episode=args.episode,
                            credits_per_extend_request=args.credits_per_extend_request)
        except PlanError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"frozen": str(args.manifest.resolve()), **result},
                         ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.manifest is None or args.max_requests is None or args.max_credits is None:
        parser.error("manifest, --max-requests and --max-credits are required")
    operation = "submit" if args.submit else "poll" if args.poll else "dry-run"
    try:
        result = execute(args.manifest, args.max_requests, args.max_credits,
                         operation, approval_admission=args.approval_admission)
    except (PlanError, UncertainRequest) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
