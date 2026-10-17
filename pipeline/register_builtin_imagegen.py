#!/usr/bin/env python3
"""Register an existing still PNG in a story-still manifest, from any hand.

This helper never generates an image and never calls a network API. It copies
one source to the manifest's declared target, writes an adjacent provenance
sidecar, and updates that one manifest record plus aggregate status counts.

``--origin imagegen`` (the default) accepts only PNG sources beneath an
explicitly configured built-in ImageGen output root and writes the historical
``.imagegen.json`` sidecar. ``human-made``, ``third-party`` and
``other-tool:NAME`` write a ``.still.json`` sidecar of kind
``supplied_still_registration`` with a truthful provider, the maker or
supplier, and the rights basis. A JPEG, WebP, TIFF or other original that was
converted to the PNG target keeps its chain through ``--original``.

The grounding guard (exact-identity coverage and continuity facts) runs the
same way for every origin. ``--retake`` retires a registered still without
deleting bytes, and ``--batch`` applies the same per-row refusals to a list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, Iterable

if __package__:
    from ._common import image_dimensions, project_root, profile_value, sha256_file as sha256
else:
    from _common import image_dimensions, project_root, profile_value, sha256_file as sha256


REGISTERABLE = {"generation_ready", "failed", "needs_changes"}
SUPPLIED_ORIGINS = {"human-made", "third-party"}
PROVIDER_UPLOAD = {"allowed", "forbidden"}
CANON_PREFIX = "canon:"


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


class GroundingError(ValueError):
    """Raised when a registration claim lacks grounding evidence."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_file(root: Path, value: str, canon_root: Path | None = None) -> Path:
    base, relative, label = root.resolve(), value, "project root"
    if value.startswith(CANON_PREFIX):
        if canon_root is None:
            raise GroundingError(f"canon reference needs an explicit --canon-root: {value}")
        base, relative, label = canon_root.resolve(), value[len(CANON_PREFIX):], "canon root"
    path = (base / relative).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise GroundingError(f"reference escapes {label}: {value}") from exc
    if not path.is_file():
        raise GroundingError(f"grounding reference does not exist: {value}")
    return path


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def expand_execution_coverage(
    root: Path, execution_refs: Iterable[str], canon_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Return every direct/packed source with auditable execution evidence.

    Every missing or escaping reference is reported together in one error.
    """
    root = root.resolve()
    coverage: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for value in stable_unique(str(item) for item in execution_refs):
        try:
            path = repo_file(root, value, canon_root)
        except GroundingError as exc:
            problems.append(str(exc))
            continue
        pack_sidecar = path.with_suffix(".refpack.json")
        if pack_sidecar.is_file():
            pack = load_json(pack_sidecar)
            if pack.get("kind") != "built_in_imagegen_reference_pack":
                raise GroundingError(f"invalid reference-pack kind: {pack_sidecar}")
            if pack.get("pack") != value:
                raise GroundingError(f"reference-pack path mismatch: {pack_sidecar}")
            expected = pack.get("pack_sha256")
            if not isinstance(expected, str) or sha256(path) != expected:
                raise GroundingError(f"reference-pack checksum mismatch: {value}")
            for entry in pack.get("sources", []):
                source = str(entry.get("source", ""))
                try:
                    source_path = repo_file(root, source, canon_root)
                except GroundingError as exc:
                    problems.append(f"{exc} (packed in {value})")
                    continue
                source_digest = entry.get("source_sha256")
                if not isinstance(source_digest, str) or sha256(source_path) != source_digest:
                    raise GroundingError(f"packed-source checksum mismatch: {source}")
                coverage[source] = {
                    "mode": "packed",
                    "execution_ref": value,
                    "pack_sidecar": (
                        value[:-len(path.name)] + pack_sidecar.name
                        if value.startswith(CANON_PREFIX)
                        else pack_sidecar.relative_to(root).as_posix()
                    ),
                }
        else:
            coverage[value] = {"mode": "direct", "execution_ref": value}
    if problems:
        raise GroundingError("; ".join(problems))
    return coverage


def parse_claim_assignments(values: Iterable[str], *, label: str) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    for raw in values:
        name, separator, source = str(raw).partition("=")
        if not separator or not name.strip() or not source.strip():
            raise GroundingError(f"{label} must use NAME=SOURCE, got {raw!r}")
        claims.append({"name": name.strip(), "source": source.strip()})
    return claims


def _identity_aliases(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    aliases: dict[str, dict[str, Any]] = {}
    for item in policy.get("grounding_guard", {}).get("identity_authorities", []):
        names = [str(item.get("name", "")), *map(str, item.get("aliases", []))]
        for name in names:
            if name:
                aliases[name.casefold()] = item
    return aliases


def validate_exact_identity_claims(
    *,
    root: Path,
    policy: dict[str, Any],
    execution_refs: list[str],
    structured_claims: list[dict[str, str]],
    free_text: str,
    canon_root: Path | None = None,
) -> list[dict[str, str]]:
    """Validate and normalize exact-identity claims against execution coverage."""
    coverage = expand_execution_coverage(root, execution_refs, canon_root)
    normalized: list[dict[str, str]] = []
    claimed_names: set[str] = set()
    for claim in structured_claims:
        name = str(claim.get("name", "")).strip()
        source = str(claim.get("source", "")).strip()
        if not name or not source:
            raise GroundingError("exact identity claims require non-empty name and source")
        if source not in coverage:
            raise GroundingError(
                f"exact identity claim for {name} lacks direct/packed execution coverage: {source}"
            )
        evidence = coverage[source]
        normalized.append(
            {
                "name": name,
                "source": source,
                "coverage": str(evidence["mode"]),
                "execution_ref": str(evidence["execution_ref"]),
            }
        )
        claimed_names.add(name.casefold())

    # Named prose claims are checked even for historical notes.  This catches
    # a named identity claim whose canon is absent from execution references.
    aliases = _identity_aliases(policy)
    mentioned: dict[str, dict[str, Any]] = {}
    folded = free_text.casefold()
    for alias, authority in aliases.items():
        alias_pattern = rf"(?<![\w-]){re.escape(alias)}(?![\w-])"
        # Bind "exact" to the named identity rather than to an unrelated set,
        # prop or timing claim elsewhere in the same note.
        if re.search(rf"\bexact(?:ly)?\b.{{0,80}}{alias_pattern}", folded) or re.search(
            rf"{alias_pattern}.{{0,80}}\bexact(?:ly)?\b", folded
        ):
            mentioned[str(authority["name"])] = authority
    generic_identity_claim = bool(
        re.search(r"\bexact(?:ly)?\b.{0,40}\bidentit(?:y|ies)\b", folded)
        or re.search(r"\bidentit(?:y|ies)\b.{0,40}\bexact(?:ly)?\b", folded)
    )
    if mentioned or generic_identity_claim:
        if not mentioned and not normalized:
            raise GroundingError(
                "an unstructured exact-identity claim has no named, source-backed identity claim"
            )
        for canonical_name, authority in mentioned.items():
            accepted = set(map(str, authority.get("sources", [])))
            covered = sorted(accepted.intersection(coverage))
            explicitly_claimed = any(
                item["name"].casefold() == canonical_name.casefold() for item in normalized
            )
            if not covered:
                raise GroundingError(
                    f"prose claims exact {canonical_name} identity, but none of its authority sources "
                    "has direct/packed execution coverage"
                )
            if not explicitly_claimed:
                source = covered[0]
                evidence = coverage[source]
                normalized.append(
                    {
                        "name": canonical_name,
                        "source": source,
                        "coverage": str(evidence["mode"]),
                        "execution_ref": str(evidence["execution_ref"]),
                        "derived_from_named_prose_claim": "true",
                    }
                )
    return normalized


def validate_manifest_identity_coverage(
    *,
    root: Path,
    row: dict[str, Any],
    execution_refs: list[str],
    exact_identity_claims: list[dict[str, str]],
    no_visible_identity: bool = False,
    canon_root: Path | None = None,
) -> list[dict[str, str]]:
    """Make a blanket manifest promise resolve to explicit active identities."""
    prompt = str(row.get("imagegen_prompt", row.get("prompt", "")))
    if not re.search(r"\b(?:copy[- ]?)?exact(?:ly)?\b.{0,80}\bidentit", prompt, re.I):
        return []
    if no_visible_identity:
        return []
    if not exact_identity_claims:
        raise GroundingError(
            "manifest promises exact identities but does not declare which visible identities are active; "
            "use --exact-identity NAME=SOURCE or --no-visible-identity"
        )
    coverage = expand_execution_coverage(root, execution_refs, canon_root)
    active_sources = stable_unique(str(item["source"]) for item in exact_identity_claims)
    missing = [source for source in active_sources if source not in coverage]
    if missing:
        raise GroundingError(
            "manifest promises exact identities without direct/packed execution coverage: "
            + ", ".join(missing)
        )
    return [
        {
            "source": source,
            "coverage": str(coverage[source]["mode"]),
            "execution_ref": str(coverage[source]["execution_ref"]),
        }
        for source in active_sources
    ]


def numbered_still(still_id: str) -> re.Match[str] | None:
    """Parse PROJECT-sNUMBER, also when a split or alternate suffix follows.

    ``demo-s010``, ``demo-s010b`` and ``demo-s010-alt`` all carry line 10. The
    historical exact form is tried first so existing IDs parse as before.
    """
    return re.fullmatch(
        r"(?P<project>[A-Za-z0-9][A-Za-z0-9_-]*)-s(?P<line>\d+)", still_id
    ) or re.fullmatch(
        r"(?P<project>[A-Za-z0-9][A-Za-z0-9_-]*)-s(?P<line>\d+)"
        r"(?P<suffix>[A-Za-z]+|[-_][A-Za-z0-9][A-Za-z0-9_-]*)",
        still_id,
    )


def resolve_continuity_facts(
    policy: dict[str, Any], still_id: str, order: list[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return (required facts, skipped facts with the reason each was skipped).

    A still with no story line number (a plate or canon sheet) skips
    ``line_range`` facts. ``from``/``to`` facts resolve by storyboard order;
    a still that is not on the storyboard skips them.
    """
    result: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    numbered = numbered_still(still_id)
    for fact in policy.get("grounding_guard", {}).get("continuity_facts", []):
        if "still_ids" in fact and still_id not in fact["still_ids"]:
            continue
        if "episode" in fact:
            if not numbered or fact["episode"].casefold() != numbered.group("project").casefold():
                continue
        if "line_range" in fact:
            if not numbered:
                skipped.append({
                    "id": str(fact.get("id")),
                    "reason": "line_range fact skipped: still ID carries no story line number",
                })
                continue
            start, end = fact["line_range"]
            if not int(start) <= int(numbered.group("line")) <= int(end):
                continue
        if "from" in fact or "to" in fact:
            if order is None:
                raise GroundingError(
                    f"continuity fact {fact.get('id')} uses from/to still IDs; pass --storyboard"
                )
            bounds = [str(fact.get("from", order[0] if order else "")),
                      str(fact.get("to", order[-1] if order else ""))]
            unknown = [value for value in bounds if value not in order]
            if unknown:
                raise GroundingError(
                    f"continuity fact {fact.get('id')} names still IDs not on the storyboard: "
                    + ", ".join(unknown)
                )
            if still_id not in order:
                skipped.append({
                    "id": str(fact.get("id")),
                    "reason": "from/to fact skipped: still is not on the storyboard",
                })
                continue
            position = order.index(still_id)
            if not order.index(bounds[0]) <= position <= order.index(bounds[1]):
                continue
        result.append(fact)
    return result, skipped


def required_continuity_facts(
    policy: dict[str, Any], still_id: str, order: list[str] | None = None
) -> list[dict[str, Any]]:
    """Apply explicit shot IDs, numbered ranges, storyboard spans, or project-wide facts."""
    return resolve_continuity_facts(policy, still_id, order)[0]


def validate_continuity_claims(
    *, policy: dict[str, Any], still_id: str, supplied_fact_ids: Iterable[str],
    order: list[str] | None = None,
) -> list[dict[str, str]]:
    supplied = set(map(str, supplied_fact_ids))
    required = required_continuity_facts(policy, still_id, order)
    missing = [str(item["id"]) for item in required if str(item["id"]) not in supplied]
    if missing:
        raise GroundingError(
            f"{still_id} is missing required continuity fact acknowledgement(s): {', '.join(missing)}"
        )
    known = {
        str(item["id"]): item
        for item in policy.get("grounding_guard", {}).get("continuity_facts", [])
    }
    unknown = sorted(supplied.difference(known))
    if unknown:
        raise GroundingError(f"unknown continuity fact id(s): {', '.join(unknown)}")
    return [
        {"id": str(item["id"]), "value": str(item["value"])} for item in required
    ]


def validate_registration_claims(
    *,
    root: Path,
    policy: dict[str, Any],
    still_id: str,
    execution_refs: list[str],
    exact_identity_claims: list[dict[str, str]],
    continuity_fact_ids: Iterable[str],
    note: str,
    row: dict[str, Any] | None = None,
    no_visible_identity: bool = False,
    canon_root: Path | None = None,
    order: list[str] | None = None,
) -> dict[str, Any]:
    identities = validate_exact_identity_claims(
        root=root,
        policy=policy,
        execution_refs=execution_refs,
        structured_claims=exact_identity_claims,
        free_text=note,
        canon_root=canon_root,
    )
    facts = validate_continuity_claims(
        policy=policy, still_id=still_id, supplied_fact_ids=continuity_fact_ids, order=order
    )
    skipped = resolve_continuity_facts(policy, still_id, order)[1]
    manifest_coverage = (
        validate_manifest_identity_coverage(
            root=root,
            row=row,
            execution_refs=execution_refs,
            exact_identity_claims=identities,
            no_visible_identity=no_visible_identity,
            canon_root=canon_root,
        )
        if row is not None
        else []
    )
    evidence = {
        "exact_identity_claims": identities,
        "manifest_identity_coverage": manifest_coverage,
        "continuity_facts": facts,
        "execution_coverage_count": len(
            expand_execution_coverage(root, execution_refs, canon_root)
        ),
    }
    if skipped:
        evidence["continuity_fact_skips"] = skipped
    return evidence


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        fail(f"source is not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_exclusive(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def parse_origin(value: str) -> tuple[str, str]:
    """Return (provider, source-root profile key) for an ``--origin`` value."""
    if value == "imagegen":
        return "openai_builtin_imagegen", "imagegen"
    if value in SUPPLIED_ORIGINS:
        return value, value
    name = value[len("other-tool:"):] if value.startswith("other-tool:") else ""
    if name and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        return name, name
    fail("--origin must be imagegen, human-made, third-party or other-tool:NAME")


def located(path: Path, root: Path, source_root: Path | None, label: str) -> str:
    """Require a supplied file inside its declared source root, else the project."""
    base = source_root or root
    if not inside(path, base):
        where = "the declared source root" if source_root else "the project (or declare a source root)"
        fail(f"{label} must be inside {where}: {path}")
    return path.relative_to(base).as_posix()


def manifest_order(root: Path, storyboard: Path | None) -> list[str] | None:
    if storyboard is None:
        return None
    path = storyboard if storyboard.is_absolute() else root / storyboard
    path = path.resolve()
    if not inside(path, root) or not path.is_file():
        fail("Explicit storyboard must be an existing file inside the project")
    return [str(row.get("still_id")) for row in load_json(path).get("stills", [])]


def canon_root_path(value: Path | None) -> Path | None:
    raw = value or profile_value("stills", "canonRoot")
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        fail(f"canon root must be an existing directory: {path}")
    return path


def reference_hashes(root: Path, refs: Iterable[str], canon_root: Path | None) -> list[dict[str, str]]:
    result = []
    for ref in stable_unique(str(item) for item in refs):
        try:
            result.append({"ref": ref, "sha256": sha256(repo_file(root, ref, canon_root))})
        except GroundingError as exc:
            fail(f"reference cannot be hashed: {exc}")
    return result


def resolve_manifest(root: Path, manifest: Path) -> Path:
    path = manifest if manifest.is_absolute() else root / manifest
    path = path.resolve()
    if not inside(path, root):
        fail("manifest must be inside --root")
    if not path.is_file():
        fail(f"manifest not found: {path}")
    return path


def manifest_record(manifest: dict[str, Any], still_id: str) -> dict[str, Any]:
    matches = [row for row in manifest.get("stills", []) if row.get("still_id") == still_id]
    if len(matches) != 1:
        fail(f"expected exactly one manifest record for {still_id}, found {len(matches)}")
    return matches[0]


def register(
    *,
    root: Path,
    manifest_path: Path,
    still_id: str,
    source_image: Path,
    origin: str = "imagegen",
    source_root: Path | None = None,
    policy_path: Path | None = None,
    note: str | None = None,
    exact_identity: Iterable[str] = (),
    grounding_fact: Iterable[str] = (),
    no_visible_identity: bool = False,
    storyboard: Path | None = None,
    canon_root: Path | None = None,
    maker: str | None = None,
    supplied_by: str | None = None,
    rights_basis: str | None = None,
    provider_upload: str | None = None,
    rights_evidence: Path | None = None,
    created: str | None = None,
    digital_source_type: str | None = None,
    original: Path | None = None,
) -> dict[str, Any]:
    exact_identity = list(exact_identity)
    if no_visible_identity and exact_identity:
        fail("--no-visible-identity cannot be combined with --exact-identity")
    provider, root_key = parse_origin(origin)
    supplied_fields = {
        "--maker": maker, "--supplied-by": supplied_by, "--rights-basis": rights_basis,
        "--provider-upload": provider_upload, "--rights-evidence": rights_evidence,
        "--created": created, "--digital-source-type": digital_source_type,
        "--original": original,
    }
    if origin == "imagegen":
        given = [flag for flag, value in supplied_fields.items() if value]
        if given:
            fail(f"{', '.join(given)} apply only to a non-imagegen --origin")
    else:
        if not (maker or "").strip() and not (supplied_by or "").strip():
            fail(f"--origin {origin} requires --maker or --supplied-by")
        if not (rights_basis or "").strip():
            fail(f"--origin {origin} requires --rights-basis")
        if provider_upload not in PROVIDER_UPLOAD:
            fail(f"--origin {origin} requires --provider-upload {' or '.join(sorted(PROVIDER_UPLOAD))}")

    source = source_image.resolve()
    if not source.is_file():
        fail(f"source image not found: {source}")
    if origin == "imagegen":
        configured = (source_root or os.environ.get("AGENTIC_IMAGEGEN_OUTPUT_ROOT")
                      or profile_value("stills", "sourceRoots", "imagegen")
                      or profile_value("imagegen", "outputRoot"))
        if not configured:
            fail("Provide --source-root or AGENTIC_IMAGEGEN_OUTPUT_ROOT for built-in ImageGen provenance")
        generated_root = Path(configured).expanduser().resolve()
        if not inside(source, generated_root):
            fail(f"source must be beneath built-in ImageGen output root: {generated_root}")
        declared_root: Path | None = generated_root
    else:
        configured = source_root or profile_value("stills", "sourceRoots", root_key)
        declared_root = Path(configured).expanduser().resolve() if configured else None
    try:
        source_format = image_dimensions(source)[0]
    except ValueError:
        source_format = None
    if origin != "imagegen" and source_format in {"jpeg", "webp"}:
        fail(f"the registered target is a PNG; convert {source_format} first and pass the "
             "unconverted file as --original so its hash stays in the chain")
    width, height = png_dimensions(source)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest_record(manifest, still_id)
    status = record.get("status")
    if status not in REGISTERABLE:
        fail(f"{still_id} has non-registerable status {status!r}")

    policy = {}
    if policy_path:
        resolved_policy = (root / policy_path).resolve()
        if not inside(resolved_policy, root) or not resolved_policy.is_file():
            fail("Explicit policy must be an existing file inside the project")
        policy = load_json(resolved_policy)
    execution_refs = list(record.get("imagegen_refs", record.get("refs", {}).get("all", [])))
    try:
        grounding_evidence = validate_registration_claims(
            root=root,
            policy=policy,
            still_id=still_id,
            execution_refs=execution_refs,
            exact_identity_claims=parse_claim_assignments(
                exact_identity, label="--exact-identity"
            ),
            continuity_fact_ids=grounding_fact,
            note=(note or "") if policy_path else "",
            row=record if policy_path else None,
            no_visible_identity=no_visible_identity,
            canon_root=canon_root,
            order=manifest_order(root, storyboard),
        )
    except GroundingError as exc:
        fail(f"grounding guard rejected {still_id}: {exc}")

    target_value = record.get("target_output_path")
    if not isinstance(target_value, str) or not target_value.endswith(".png"):
        fail(f"{still_id} has invalid PNG target_output_path")
    target = (root / target_value).resolve()
    if not inside(target, root):
        fail("target_output_path escapes --root")
    sidecar = target.with_suffix(".imagegen.json" if origin == "imagegen" else ".still.json")
    if target.exists() or sidecar.exists():
        fail(f"refusing to overwrite existing target or sidecar for {still_id}")

    now = datetime.now(timezone.utc).isoformat()
    source_digest = sha256(source)
    manifest_relative = manifest_path.relative_to(root).as_posix()
    execution_prompt = record.get("imagegen_prompt", record.get("prompt", ""))
    prompt_digest = hashlib.sha256(str(execution_prompt).encode("utf-8")).hexdigest()
    canon_hashes = reference_hashes(
        root, [ref for ref in execution_refs if str(ref).startswith(CANON_PREFIX)], canon_root
    )
    if origin == "imagegen":
        provenance: dict[str, Any] = {
            "schema_version": 1,
            "kind": "built_in_imagegen_registration",
            "provider": "openai_builtin_imagegen",
            "tool": "image_gen",
            "still_id": still_id,
            "episode": manifest.get("episode"),
            "registered_at": now,
            "registered_by": "pipeline/register_builtin_imagegen.py",
            "source_image": source.relative_to(declared_root).as_posix(),
            "source_sha256": source_digest,
            "target_output_path": target.relative_to(root).as_posix(),
            "target_sha256": source_digest,
            "width": width,
            "height": height,
            "manifest": manifest_relative,
            "record_fingerprint": record.get("record_fingerprint"),
            "source_fingerprint": record.get("source_fingerprint"),
            "prompt_sha256": prompt_digest,
            "refs": record.get("refs", {}).get("all", []),
            "imagegen_refs": execution_refs,
            "exact_identity_claims": grounding_evidence["exact_identity_claims"],
            "manifest_identity_coverage": grounding_evidence["manifest_identity_coverage"],
            "continuity_facts": grounding_evidence["continuity_facts"],
            "execution_coverage_count": grounding_evidence["execution_coverage_count"],
            "no_visible_identity": no_visible_identity,
        }
        if canon_hashes:
            provenance["canon_refs"] = canon_hashes
    else:
        rights: dict[str, Any] = {"basis": rights_basis.strip(), "providerUpload": provider_upload}
        if rights_evidence:
            evidence_path = (rights_evidence if rights_evidence.is_absolute()
                             else root / rights_evidence).resolve()
            if not inside(evidence_path, root) or not evidence_path.is_file():
                fail("--rights-evidence must be an existing file inside the project")
            rights["evidence"] = {"path": evidence_path.relative_to(root).as_posix(),
                                  "sha256": sha256(evidence_path)}
        provenance = {
            "schema_version": 1,
            "kind": "supplied_still_registration",
            "origin": origin,
            "provider": provider,
            "still_id": still_id,
            "episode": manifest.get("episode"),
            "registered_at": now,
            "registered_by": "pipeline/register_builtin_imagegen.py",
            "source_root": f"stills.sourceRoots.{root_key}" if declared_root else "project",
            "source_image": located(source, root, declared_root, "source image"),
            "source_sha256": source_digest,
            "target_output_path": target.relative_to(root).as_posix(),
            "target_sha256": source_digest,
            "width": width,
            "height": height,
            "manifest": manifest_relative,
            "record_fingerprint": record.get("record_fingerprint"),
            "source_fingerprint": record.get("source_fingerprint"),
            "prompt_sha256": prompt_digest if str(execution_prompt) else None,
            "requested_size": record.get("size"),
            "refs": record.get("refs", {}).get("all", []),
            "reference_hashes": reference_hashes(root, execution_refs, canon_root),
            "rights": rights,
            "exact_identity_claims": grounding_evidence["exact_identity_claims"],
            "manifest_identity_coverage": grounding_evidence["manifest_identity_coverage"],
            "continuity_facts": grounding_evidence["continuity_facts"],
            "execution_coverage_count": grounding_evidence["execution_coverage_count"],
            "no_visible_identity": no_visible_identity,
        }
        for key, value in (("maker", maker), ("supplied_by", supplied_by),
                           ("created", created), ("digital_source_type", digital_source_type)):
            if value and value.strip():
                provenance[key] = value.strip()
        if original:
            original_path = original.resolve()
            if not original_path.is_file():
                fail(f"--original not found: {original_path}")
            try:
                fmt, original_width, original_height = image_dimensions(original_path)
            except ValueError:
                fmt, original_width, original_height = None, None, None
            provenance["original"] = {
                "path": located(original_path, root, declared_root, "--original"),
                "sha256": sha256(original_path),
                "format": fmt or (original_path.suffix.lstrip(".").lower() or None),
                "width": original_width,
                "height": original_height,
            }
    if "continuity_fact_skips" in grounding_evidence:
        provenance["continuity_fact_skips"] = grounding_evidence["continuity_fact_skips"]
    if note:
        provenance["note"] = note

    copied = sidecar_written = False
    try:
        copy_exclusive(source, target)
        copied = True
        if sha256(target) != source_digest:
            raise RuntimeError("copied target checksum does not match source")
        write_json_exclusive(sidecar, provenance)
        sidecar_written = True

        record["status"] = "generated"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["generated_at"] = now
        record["generation_provider"] = provider
        if origin != "imagegen":
            record["still_origin"] = origin
        record["provenance_sidecar"] = sidecar.relative_to(root).as_posix()
        record["grounding_claims"] = {
            "exact_identity_claims": grounding_evidence["exact_identity_claims"],
            "manifest_identity_coverage": grounding_evidence["manifest_identity_coverage"],
            "continuity_facts": grounding_evidence["continuity_facts"],
            "no_visible_identity": no_visible_identity,
        }
        if "continuity_fact_skips" in grounding_evidence:
            record["grounding_claims"]["continuity_fact_skips"] = grounding_evidence[
                "continuity_fact_skips"
            ]
        record.pop("last_error", None)
        manifest["updated_at"] = now
        manifest["status_counts"] = dict(Counter(row.get("status") for row in manifest["stills"]))
        write_json_atomic(manifest_path, manifest)
    except Exception:
        if sidecar_written:
            sidecar.unlink(missing_ok=True)
        if copied:
            target.unlink(missing_ok=True)
        raise

    return {
        "episode": manifest.get("episode"),
        "still_id": still_id,
        "status": "generated",
        "target": target.relative_to(root).as_posix(),
        "sidecar": sidecar.relative_to(root).as_posix(),
        "sha256": source_digest,
        "dimensions": [width, height],
        "status_counts": manifest["status_counts"],
    }


def retake(*, root: Path, manifest_path: Path, still_id: str, note: str | None = None) -> dict[str, Any]:
    """Retire a registered still and its sidecar; never delete bytes.

    The files move to ``art/retired/<still_id>/<short-sha>/``, the manifest row
    returns to ``needs_changes`` with a history entry, and the downstream
    assets that referenced the retired target are listed. Their hash bindings
    then fail loudly until they are remade or explicitly re-pointed.
    """
    if __package__:
        from .asset_registry import asset_key, downstream, load_registry
    else:
        from asset_registry import asset_key, downstream, load_registry
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest_record(manifest, still_id)
    if record.get("status") != "generated":
        fail(f"{still_id} has status {record.get('status')!r}; only a registered still can be retaken")
    target_value = record.get("target_output_path")
    sidecar_value = record.get("provenance_sidecar")
    if not isinstance(target_value, str) or not isinstance(sidecar_value, str):
        fail(f"{still_id} has no registered target and sidecar to retire")
    target = (root / target_value).resolve()
    sidecar = (root / sidecar_value).resolve()
    if not inside(target, root) or not inside(sidecar, root):
        fail("registered target or sidecar escapes --root")
    if not target.is_file() or not sidecar.is_file():
        fail(f"{still_id} target or sidecar is missing; reconcile before a retake")
    target_digest = sha256(target)
    try:
        assets = load_registry(root)
        impact: Any = downstream(assets, asset_key(target_value, root))
    except ValueError as exc:
        impact = {"unavailable": str(exc)}
    retired = root / "art" / "retired" / still_id / target_digest[:12]
    if retired.exists():
        fail(f"retired location already exists: {retired.relative_to(root).as_posix()}")
    retired.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "retired_at": now,
        "target_output_path": target_value,
        "target_sha256": target_digest,
        "provenance_sidecar": sidecar_value,
        "provenance_sidecar_sha256": sha256(sidecar),
        "generation_provider": record.get("generation_provider"),
        "retired_to": retired.relative_to(root).as_posix(),
        "downstream": impact,
    }
    if note:
        entry["note"] = note
    moved: list[tuple[Path, Path]] = []
    try:
        for path in (target, sidecar):
            destination = retired / path.name
            os.rename(path, destination)
            moved.append((path, destination))
        record.setdefault("retake_history", []).append(entry)
        record["status"] = "needs_changes"
        for key in ("generated_at", "generation_provider", "provenance_sidecar",
                    "grounding_claims", "still_origin"):
            record.pop(key, None)
        manifest["updated_at"] = now
        manifest["status_counts"] = dict(Counter(row.get("status") for row in manifest["stills"]))
        write_json_atomic(manifest_path, manifest)
    except Exception:
        for original, destination in reversed(moved):
            os.rename(destination, original)
        raise
    return {"still_id": still_id, "status": "needs_changes", **entry}


BATCH_KEYS = {
    "still_id", "source_image", "origin", "note", "exact_identity", "grounding_fact",
    "no_visible_identity", "maker", "supplied_by", "rights_basis", "provider_upload",
    "rights_evidence", "created", "digital_source_type", "original",
}


def run_batch(path: Path, **shared: Any) -> list[dict[str, Any]]:
    """Register rows in order with the usual refusals; stop at the first failure."""
    doc = load_json(path)
    rows = doc.get("registrations") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not rows:
        fail("--batch file must hold a non-empty 'registrations' list")
    done: list[dict[str, Any]] = []
    for position, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not {"still_id", "source_image"} <= set(row):
            fail(f"batch row {position} needs still_id and source_image")
        unknown = sorted(set(row) - BATCH_KEYS)
        if unknown:
            fail(f"batch row {position} has unknown keys: {', '.join(unknown)}")
        options = dict(row)
        for key in ("source_image", "rights_evidence", "original"):
            if options.get(key):
                value = Path(options[key])
                options[key] = value if value.is_absolute() else path.parent / value
        try:
            done.append(register(**shared, **options))
        except SystemExit as exc:
            fail(f"batch stopped at row {position} ({row['still_id']}): {exc}; "
                 f"registered before it: {[item['still_id'] for item in done]}")
    return done


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path, help="Story generation_manifest.json")
    parser.add_argument("still_id", nargs="?", help="Exact manifest still_id, for example demo-s0001")
    parser.add_argument("source_image", nargs="?", type=Path, help="Existing PNG to register")
    parser.add_argument("--root", type=Path, help="Private project root (or AGENTIC_PROJECT_ROOT)")
    parser.add_argument(
        "--origin", default="imagegen",
        help="imagegen (default), human-made, third-party or other-tool:NAME",
    )
    parser.add_argument(
        "--source-root", type=Path,
        help="Source root for this origin; defaults to the profile's stills.sourceRoots.<provider> "
             "(imagegen also reads imagegen.outputRoot); a supplied still without one must sit in the project",
    )
    parser.add_argument("--policy", type=Path, help="Optional explicit project policy; no automatic discovery")
    parser.add_argument("--storyboard", type=Path,
                        help="Explicit storyboard whose order resolves from/to continuity facts")
    parser.add_argument("--canon-root", type=Path,
                        help="Read-only series canon for canon:<path> references (or profile stills.canonRoot)")
    parser.add_argument("--note", help="Optional human provenance note")
    parser.add_argument(
        "--exact-identity",
        action="append",
        default=[],
        metavar="NAME=SOURCE",
        help="Named exact-identity claim and its authority source; repeat as needed",
    )
    parser.add_argument(
        "--grounding-fact",
        action="append",
        default=[],
        metavar="FACT_ID",
        help="Acknowledge a policy continuity fact; repeat for every applicable fact",
    )
    parser.add_argument(
        "--no-visible-identity",
        action="store_true",
        help="Explicitly declare that the frame contains no identity-bearing character",
    )
    supplied = parser.add_argument_group("non-imagegen origins")
    supplied.add_argument("--maker", help="Who made the image (maker or --supplied-by is required)")
    supplied.add_argument("--supplied-by", help="Who supplied the image")
    supplied.add_argument("--rights-basis", help="Required: the basis on which the film may use it")
    supplied.add_argument("--provider-upload", choices=sorted(PROVIDER_UPLOAD),
                          help="Required: whether it may be uploaded to a generation provider")
    supplied.add_argument("--rights-evidence", type=Path, help="Optional project file recorded by hash")
    supplied.add_argument("--created", help="Optional creation date")
    supplied.add_argument("--digital-source-type", help="Optional IPTC digitalSourceType URI")
    supplied.add_argument("--original", type=Path,
                          help="Unconverted original (JPEG, WebP, TIFF, ...) whose PNG is being registered")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--retake", action="store_true",
                       help="Retire the registered still and sidecar to art/retired/ and reset the row")
    modes.add_argument("--batch", type=Path,
                       help="JSON {'registrations': [...]} applied in order; each row names still_id, "
                            "source_image and any per-row option by its flag name with underscores; "
                            "relative paths resolve from the batch file; stops at the first refusal")
    args = parser.parse_args(argv)
    if args.batch:
        if args.still_id or args.source_image:
            parser.error("--batch takes the manifest only; rows name still_id and source_image")
    elif args.retake:
        if not args.still_id or args.source_image:
            parser.error("--retake takes the manifest and a still_id only")
    elif not args.still_id or not args.source_image:
        parser.error("still_id and source_image are required")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = project_root(args.root)
    manifest_path = resolve_manifest(root, args.manifest)
    if args.retake:
        result: Any = retake(root=root, manifest_path=manifest_path,
                             still_id=args.still_id, note=args.note)
    else:
        shared = dict(
            root=root, manifest_path=manifest_path, source_root=args.source_root,
            policy_path=args.policy, storyboard=args.storyboard,
            canon_root=canon_root_path(args.canon_root),
        )
        if args.batch:
            result = run_batch(args.batch.resolve(), **shared)
        else:
            result = register(
                **shared, still_id=args.still_id, source_image=args.source_image,
                origin=args.origin, note=args.note, exact_identity=args.exact_identity,
                grounding_fact=args.grounding_fact, no_visible_identity=args.no_visible_identity,
                maker=args.maker, supplied_by=args.supplied_by, rights_basis=args.rights_basis,
                provider_upload=args.provider_upload, rights_evidence=args.rights_evidence,
                created=args.created, digital_source_type=args.digital_source_type,
                original=args.original,
            )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
