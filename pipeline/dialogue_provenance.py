"""Read-only validation of the actual Grok request, not just sidecar labels."""
from __future__ import annotations

import hashlib
import json


def canonical_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def validate_dialogue_request(record: dict, expected: dict, prompt_text: str) -> None:
    request = record.get("request")
    if not isinstance(request, dict) or canonical_sha(request) != record.get("requestSha256"):
        raise ValueError("dialogue nested request is missing or its fingerprint is invalid")
    if record.get("requestFingerprint") != record["requestSha256"]:
        raise ValueError("dialogue request digest aliases disagree")
    for key in ("model", "sourceSha256", "promptSha256", "executionManifestSha256",
                "executionRequestSha256"):
        if not expected.get(key) or request.get(key) != expected[key]:
            raise ValueError("dialogue nested request mismatch: " + key)
    provider_input = request.get("input")
    if not isinstance(provider_input, dict):
        raise ValueError("dialogue provider input is missing")
    for key, value in {"prompt": prompt_text.strip(), "duration": expected["durationSeconds"],
                       "resolution": expected["resolution"],
                       "aspect_ratio": expected["aspectRatio"]}.items():
        if provider_input.get(key) != value:
            raise ValueError("dialogue provider input mismatch: " + key)
    text_sha = hashlib.sha256(prompt_text.strip().encode()).hexdigest()
    if request.get("promptTextSha256") != text_sha:
        raise ValueError("dialogue stripped prompt text hash mismatch")
    local = record.get("localRequest")
    if not isinstance(local, dict) or canonical_sha(local) != record.get("localRequestSha256"):
        raise ValueError("dialogue local request is missing or its fingerprint is invalid")
    if request.get("localRequestSha256") != record["localRequestSha256"]:
        raise ValueError("dialogue nested/local request bindings disagree")
    for key in ("jobId", "model", "sourceSha256", "promptSha256", "executionManifestSha256",
                "executionRequestSha256", "resolution", "aspectRatio", "durationSeconds", "audio"):
        if local.get(key) != expected[key]:
            raise ValueError("dialogue local request mismatch: " + key)
    hashes = local.get("inputHashes", {})
    # Exactly the frozen reference hashes, in order; none for a single-still job.
    references = list(expected.get("referenceSha256s", []))
    if hashes != {"primary": expected["sourceSha256"], "last": None, "references": references,
                  "videos": []}:
        raise ValueError("dialogue local input hashes differ from the frozen still and references")
    if local.get("referenceOnly") is not False or local.get("multiClip") is not False:
        raise ValueError("dialogue local mode differs from the frozen single-still route")
