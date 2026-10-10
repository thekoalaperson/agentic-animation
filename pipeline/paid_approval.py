#!/usr/bin/env python3
"""Validate a separate, hash-bound paid-generation authorization record.

Paid creative manifests stay byte-immutable.  A user authorization is admitted
beside them and names each exact manifest plus its finite execution boundary.
This prevents an approval-bit edit from invalidating downstream provenance while
still requiring an explicit, durable record before any paid POST.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
ROOT_FIELDS = {
    "schemaVersion", "kind", "episode", "status", "authorizationId",
    "authorizedBy", "authorizedAtUtc", "authorizationText", "budgetPolicy",
    "topUpAuthorized", "automaticRetryAuthorized", "manifests", "projectRoot",
}
MANIFEST_FIELDS = {"lane", "manifestPath", "manifestSha256", "finiteExecution"}


class PaidApprovalError(ValueError):
    """The supplied authorization is missing, stale, or out of scope."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaidApprovalError(f"cannot read paid authorization: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaidApprovalError("paid authorization must be a JSON object")
    return value


def validate_paid_approval(
    admission_path: Path | None,
    *,
    episode: str,
    lane: str,
    manifest_path: Path,
    manifest_sha256: str,
    finite_execution: dict[str, Any],
) -> dict[str, Any] | None:
    """Return normalized authorization evidence, or ``None`` when not supplied.

    Version 2 binds a relocatable private project and explicit finite limits.
    Profiles do not authorize spending. Historical grants and their original
    runners remain immutable; this function never upgrades a legacy grant.
    """
    if admission_path is None:
        return None
    admission_path = admission_path.resolve()
    manifest_path = manifest_path.resolve()
    doc = _read_object(admission_path)
    if set(doc) != ROOT_FIELDS:
        raise PaidApprovalError("paid authorization root schema mismatch")
    expected = {
        "schemaVersion": 2,
        "kind": "paid-generation-authorization",
        "episode": episode,
        "status": "APPROVED",
        "budgetPolicy": "explicit-finite-limits",
        "topUpAuthorized": False,
        "automaticRetryAuthorized": False,
    }
    for key, value in expected.items():
        if doc.get(key) != value:
            raise PaidApprovalError(f"paid authorization {key} must be {value!r}")
    if not isinstance(doc.get("authorizedBy"), str) or not doc["authorizedBy"].strip():
        raise PaidApprovalError("paid authorization must name its authorizer")
    if not isinstance(doc.get("authorizationId"), str) or not doc["authorizationId"]:
        raise PaidApprovalError("paid authorization authorizationId is missing")
    if not isinstance(doc.get("authorizedAtUtc"), str) or not UTC.fullmatch(doc["authorizedAtUtc"]):
        raise PaidApprovalError("paid authorization authorizedAtUtc must be UTC to the second")
    words = doc.get("authorizationText")
    if (not isinstance(words, list) or not words
            or any(not isinstance(item, str) or not item.strip() for item in words)):
        raise PaidApprovalError("paid authorization must preserve non-empty user wording")
    root_value = doc.get("projectRoot")
    if not isinstance(root_value, str) or not root_value or Path(root_value).is_absolute():
        raise PaidApprovalError("projectRoot must be relative to the authorization file")
    root = (admission_path.parent / root_value).resolve()
    if not root.is_dir():
        raise PaidApprovalError("authorized projectRoot is missing")
    try:
        expected_relative = manifest_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PaidApprovalError("paid manifest is outside the authorized project") from exc
    if not isinstance(manifest_sha256, str) or not SHA256.fullmatch(manifest_sha256):
        raise PaidApprovalError("current paid manifest SHA-256 is malformed")
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_sha256:
        raise PaidApprovalError("current paid manifest bytes do not match the supplied hash")

    rows = doc.get("manifests")
    if not isinstance(rows, list) or not rows:
        raise PaidApprovalError("paid authorization manifests must be a non-empty array")
    matches = []
    seen_lanes: set[str] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != MANIFEST_FIELDS:
            raise PaidApprovalError(f"paid authorization manifest row {position} schema mismatch")
        row_lane = row.get("lane")
        if not isinstance(row_lane, str) or not row_lane or row_lane in seen_lanes:
            raise PaidApprovalError("paid authorization manifest lanes must be unique strings")
        seen_lanes.add(row_lane)
        if (not isinstance(row.get("manifestSha256"), str)
                or not SHA256.fullmatch(row["manifestSha256"])):
            raise PaidApprovalError(f"paid authorization {row_lane} manifest hash is malformed")
        scope = row.get("finiteExecution")
        if (not isinstance(scope, dict) or not scope
                or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                       or not math.isfinite(v) or v <= 0 for v in scope.values())):
            raise PaidApprovalError("finite execution limits must be explicit positive numbers")
        if not {"items", "jobs", "requests"}.intersection(scope):
            raise PaidApprovalError("finite execution must limit the number of requests or items")
        if not {"characters", "credits", "generatedSeconds"}.intersection(scope):
            raise PaidApprovalError("finite execution must limit provider usage")
        row_path = row.get("manifestPath")
        if (not isinstance(row_path, str) or not row_path
                or Path(row_path).is_absolute()
                or not (root / row_path).resolve().is_relative_to(root)):
            raise PaidApprovalError("authorized manifest path escapes the project")
        if row_lane == lane:
            matches.append(row)
    if len(matches) != 1:
        raise PaidApprovalError(f"paid authorization must contain exactly one {lane!r} row")
    row = matches[0]
    if row.get("manifestPath") != expected_relative:
        raise PaidApprovalError(f"paid authorization {lane} manifest path is stale")
    if row.get("manifestSha256") != manifest_sha256:
        raise PaidApprovalError(f"paid authorization {lane} manifest hash is stale")
    if row.get("finiteExecution") != finite_execution:
        raise PaidApprovalError(f"paid authorization {lane} finite execution scope differs")

    return {
        "path": str(admission_path),
        "sha256": sha256_file(admission_path),
        "authorizationId": doc["authorizationId"],
        "authorizedAtUtc": doc["authorizedAtUtc"],
        "budgetPolicy": doc["budgetPolicy"],
        "lane": lane,
        "manifestSha256": manifest_sha256,
        "finiteExecution": finite_execution,
    }
