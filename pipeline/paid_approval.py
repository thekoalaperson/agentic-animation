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
# Version 3 is version 2 plus a retry grant: automaticRetryAuthorized true and a
# retryBudget a person set. Everything else is checked by the version 2 rules.
RETRY_BUDGET_FIELDS = {"jobs", "credits"}


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
    allow_retry_budget: bool = False,
) -> dict[str, Any] | None:
    """Return normalized authorization evidence, or ``None`` when not supplied.

    Version 2 binds a relocatable private project and explicit finite limits.
    Profiles do not authorize spending. Historical grants and their original
    runners remain immutable; this function never upgrades a legacy grant.

    Version 3 adds a person's retry grant and is accepted only by a runner that
    passes ``allow_retry_budget``; its other fields pass the version 2 checks.
    """
    if admission_path is None:
        return None
    admission_path = admission_path.resolve()
    doc = _read_object(admission_path)
    if doc.get("schemaVersion") == 3:
        if not allow_retry_budget:
            raise PaidApprovalError(f"the {lane} lane does not retry; use a version 2 record")
        budget = doc.get("retryBudget")
        if doc.get("automaticRetryAuthorized") is not True:
            raise PaidApprovalError("a version 3 record exists to grant retries: "
                                    "automaticRetryAuthorized must be true")
        if (not isinstance(budget, dict) or set(budget) != RETRY_BUDGET_FIELDS
                or isinstance(budget["jobs"], bool) or not isinstance(budget["jobs"], int)
                or budget["jobs"] <= 0 or isinstance(budget["credits"], bool)
                or not isinstance(budget["credits"], (int, float))
                or not math.isfinite(budget["credits"]) or budget["credits"] <= 0):
            raise PaidApprovalError("retryBudget must be {jobs, credits}, explicit positive limits")
        view = {key: value for key, value in doc.items() if key != "retryBudget"}
        view.update(schemaVersion=2, automaticRetryAuthorized=False)
        evidence = _validate_document(view, admission_path, episode=episode, lane=lane,
                                      manifest_path=manifest_path, manifest_sha256=manifest_sha256,
                                      finite_execution=finite_execution)
        return {**evidence, "schemaVersion": 3, "automaticRetryAuthorized": True,
                "retryBudget": dict(budget)}
    return _validate_document(doc, admission_path, episode=episode, lane=lane,
                              manifest_path=manifest_path, manifest_sha256=manifest_sha256,
                              finite_execution=finite_execution)


def _validate_document(doc: dict[str, Any], admission_path: Path, *, episode: str, lane: str,
                       manifest_path: Path, manifest_sha256: str,
                       finite_execution: dict[str, Any]) -> dict[str, Any]:
    """The version 2 checks, applied to the record as read."""
    manifest_path = manifest_path.resolve()
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


# ---------------------------------------------------------------------------
# Command line: describe the lanes, draft an unsigned record, check a record.

LANES = {
    "grok-dialogue": {
        "runner": "kie_dialogue_finite.py",
        "finiteExecution": {
            "jobs": "number of frozen jobs",
            "generatedSeconds": "sum of frozen durationSeconds",
            "credits": "the manifest's maximumCredits",
        },
    },
    "elevenlabs-narration": {
        "runner": "eleven_batch_finite.py",
        "finiteExecution": {
            "items": "number of frozen items",
            "characters": "the manifest's aggregateCharacters",
        },
    },
    "suno-score": {
        "runner": "kie_music_finite.py",
        "finiteExecution": {
            "requests": "number of frozen cues",
            "credits": "cues x creditsPerRequest",
            "sourceSeconds": "sum of each cue's requested duration",
        },
    },
    "media-observation": {
        "runner": "observe_media_finite.py",
        "finiteExecution": {
            "jobs": "number of review jobs",
            "credits": "the plan's maxCredits",
        },
    },
}

GRANT_FIELDS = ("authorizedBy", "authorizedAtUtc", "authorizationText", "authorizationId")
CLI_EPILOG = """\
A draft leaves authorizedBy, authorizedAtUtc, authorizationText and
authorizationId empty and its status DRAFT, which every runner refuses. Those
fields record a person's actual grant: fill them from that person's own words
at the time they give it, then set status to APPROVED. An agent never fills
them from its own judgement, and a profile or a manifest's financialApproval
is never a grant. Draft the record after the final freeze: any later edit to
the manifest changes the hash the record binds."""


def _lane_plan(lane: str, manifest: Path, project: Path | None) -> tuple[str, dict[str, Any], Path]:
    """Return (episode, finiteExecution, projectRoot) exactly as the runner computes them."""
    import importlib
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    manifest = manifest.resolve()
    doc = _read_object(manifest)
    common = importlib.import_module("_common")
    if lane == "grok-dialogue":
        runner = importlib.import_module("kie_dialogue_finite")
        root = common.project_root(project)
        try:
            runner.configure(manifest, root, root / str(doc.get("productionOverrides")),
                             None, root / str(doc.get("authority")))
            plan = runner.load_plan(manifest, doc.get("maximumGeneratedSeconds"))
        except runner.PlanError as exc:
            raise PaidApprovalError(f"manifest is not runnable: {exc}") from exc
        return runner.EPISODE_CODE, runner.approval_scope(plan), root
    if lane == "elevenlabs-narration":
        runner = importlib.import_module("eleven_batch_finite")
        try:
            plan = runner.load_plan(manifest, doc.get("aggregateCharacters") or 0)
        except runner.PlanError as exc:
            raise PaidApprovalError(f"manifest is not runnable: {exc}") from exc
        return doc.get("episode"), runner.approval_scope(plan), common.project_root(project)
    if lane == "suno-score":
        runner = importlib.import_module("kie_music_finite")
        try:
            plan = runner.load_plan(manifest, doc.get("maxRequests") or 0, doc.get("maxCredits") or 0)
        except runner.PlanError as exc:
            raise PaidApprovalError(f"manifest is not runnable: {exc}") from exc
        return doc.get("episode"), runner.approval_scope(plan), common.project_root(project)
    if lane == "media-observation":
        runner = importlib.import_module("observe_media_finite")
        try:
            scope = runner.approval_scope(doc)
        except ValueError as exc:
            raise PaidApprovalError(str(exc)) from exc
        return doc.get("episode"), scope, common.project_root(project)
    raise PaidApprovalError(f"unknown lane {lane!r}; known lanes: {', '.join(LANES)}")


def draft_record(lane: str, manifest: Path, record: Path, project: Path | None = None) -> dict[str, Any]:
    """Write an unsigned version-2 record that the validator refuses until a person grants it."""
    import os

    episode, scope, root = _lane_plan(lane, manifest, project)
    manifest = manifest.resolve()
    record = record.resolve()
    try:
        relative = manifest.relative_to(root).as_posix()
    except ValueError as exc:
        raise PaidApprovalError("manifest is outside the project root") from exc
    if not record.is_relative_to(root):
        raise PaidApprovalError("keep the approval record inside the project it authorizes")
    document = {
        "schemaVersion": 2, "kind": "paid-generation-authorization",
        "episode": episode, "status": "DRAFT",
        "authorizationId": "", "authorizedBy": "", "authorizedAtUtc": "",
        "authorizationText": [],
        "budgetPolicy": "explicit-finite-limits",
        "topUpAuthorized": False, "automaticRetryAuthorized": False,
        "projectRoot": os.path.relpath(root, record.parent),
        "manifests": [{"lane": lane, "manifestPath": relative,
                       "manifestSha256": sha256_file(manifest), "finiteExecution": scope}],
    }
    raw = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    record.parent.mkdir(parents=True, exist_ok=True)
    try:
        with record.open("xb") as target:
            target.write(raw)
    except FileExistsError as exc:
        raise PaidApprovalError(f"refusing to overwrite an existing record: {record}") from exc
    return document


def check_record(record: Path, manifest: Path, lane: str, project: Path | None = None) -> dict[str, Any]:
    """Validate a record against the manifest as its runner would; name what is stale."""
    episode, scope, _ = _lane_plan(lane, manifest, project)
    try:
        return validate_paid_approval(record, episode=episode, lane=lane,
                                      manifest_path=manifest, manifest_sha256=sha256_file(manifest),
                                      finite_execution=scope)
    except PaidApprovalError as exc:
        detail = [str(exc)]
        try:
            rows = [row for row in _read_object(record.resolve()).get("manifests", [])
                    if isinstance(row, dict) and row.get("lane") == lane]
        except PaidApprovalError:
            rows = []
        if len(rows) == 1 and isinstance(rows[0].get("finiteExecution"), dict):
            recorded = rows[0]["finiteExecution"]
            for key in sorted(set(recorded) | set(scope)):
                if recorded.get(key) != scope.get(key):
                    detail.append(f"finiteExecution.{key}: record {recorded.get(key)!r}, "
                                  f"manifest {scope.get(key)!r}")
        raise PaidApprovalError("; ".join(detail)) from exc


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=__doc__, epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="print every lane and its exact finiteExecution keys")
    draft = commands.add_parser("draft", help="write an unsigned DRAFT record for one manifest",
                                epilog=CLI_EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    draft.add_argument("lane", choices=sorted(LANES))
    draft.add_argument("manifest", type=Path)
    draft.add_argument("--out", type=Path, required=True, help="new record path inside the project")
    draft.add_argument("--project", type=Path, help="project root (or AGENTIC_PROJECT_ROOT)")
    check = commands.add_parser("check", help="validate a record against a manifest and lane")
    check.add_argument("record", type=Path)
    check.add_argument("manifest", type=Path)
    check.add_argument("lane", choices=sorted(LANES))
    check.add_argument("--project", type=Path, help="project root (or AGENTIC_PROJECT_ROOT)")
    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            result: Any = {"schemaVersion": 2, "lanes": LANES,
                           "grantFields": list(GRANT_FIELDS), "note": CLI_EPILOG}
        elif args.command == "draft":
            result = draft_record(args.lane, args.manifest, args.out, args.project)
        else:
            result = {"status": "VALID", **check_record(args.record, args.manifest,
                                                         args.lane, args.project)}
    except (PaidApprovalError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
