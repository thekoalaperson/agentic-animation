#!/usr/bin/env python3
"""Register an existing built-in ImageGen PNG in a story-still manifest.

This helper never generates an image and never calls a network API.  It only
accepts sources beneath an explicitly configured ImageGen output root, copies one source to the manifest's declared target, writes
an adjacent ``.imagegen.json`` provenance sidecar, and updates that one
manifest record plus aggregate status counts.
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
    from ._common import project_root, profile_value
else:
    from _common import project_root, profile_value


REGISTERABLE = {"generation_ready", "failed", "needs_changes"}


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GroundingError(ValueError):
    """Raised when a registration claim lacks grounding evidence."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_file(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise GroundingError(f"reference escapes project root: {value}") from exc
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


def expand_execution_coverage(root: Path, execution_refs: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Return every direct/packed source with auditable execution evidence."""
    root = root.resolve()
    coverage: dict[str, dict[str, Any]] = {}
    for value in stable_unique(str(item) for item in execution_refs):
        path = repo_file(root, value)
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
                source_path = repo_file(root, source)
                source_digest = entry.get("source_sha256")
                if not isinstance(source_digest, str) or sha256(source_path) != source_digest:
                    raise GroundingError(f"packed-source checksum mismatch: {source}")
                coverage[source] = {
                    "mode": "packed",
                    "execution_ref": value,
                    "pack_sidecar": pack_sidecar.relative_to(root).as_posix(),
                }
        else:
            coverage[value] = {"mode": "direct", "execution_ref": value}
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
) -> list[dict[str, str]]:
    """Validate and normalize exact-identity claims against execution coverage."""
    coverage = expand_execution_coverage(root, execution_refs)
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
    coverage = expand_execution_coverage(root, execution_refs)
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


def required_continuity_facts(policy: dict[str, Any], still_id: str) -> list[dict[str, Any]]:
    """Apply explicit shot IDs, numbered ranges, or project-wide facts."""
    result = []
    numbered = re.fullmatch(r"(?P<project>[A-Za-z0-9][A-Za-z0-9_-]*)-s(?P<line>\d+)", still_id)
    for fact in policy.get("grounding_guard", {}).get("continuity_facts", []):
        if "still_ids" in fact and still_id not in fact["still_ids"]:
            continue
        if "episode" in fact:
            if not numbered or fact["episode"].casefold() != numbered.group("project").casefold():
                continue
        if "line_range" in fact:
            if not numbered:
                raise GroundingError("Range-based continuity requires a PROJECT-sNUMBER shot ID")
            start, end = fact["line_range"]
            if not int(start) <= int(numbered.group("line")) <= int(end):
                continue
        result.append(fact)
    return result


def validate_continuity_claims(
    *, policy: dict[str, Any], still_id: str, supplied_fact_ids: Iterable[str]
) -> list[dict[str, str]]:
    supplied = set(map(str, supplied_fact_ids))
    required = required_continuity_facts(policy, still_id)
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
) -> dict[str, Any]:
    identities = validate_exact_identity_claims(
        root=root,
        policy=policy,
        execution_refs=execution_refs,
        structured_claims=exact_identity_claims,
        free_text=note,
    )
    facts = validate_continuity_claims(
        policy=policy, still_id=still_id, supplied_fact_ids=continuity_fact_ids
    )
    manifest_coverage = (
        validate_manifest_identity_coverage(
            root=root,
            row=row,
            execution_refs=execution_refs,
            exact_identity_claims=identities,
            no_visible_identity=no_visible_identity,
        )
        if row is not None
        else []
    )
    return {
        "exact_identity_claims": identities,
        "manifest_identity_coverage": manifest_coverage,
        "continuity_facts": facts,
        "execution_coverage_count": len(expand_execution_coverage(root, execution_refs)),
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Story generation_manifest.json")
    parser.add_argument("still_id", help="Exact manifest still_id, for example demo-s0001")
    parser.add_argument("source_image", type=Path, help="Existing built-in ImageGen PNG")
    parser.add_argument("--root", type=Path, help="Private project root (or AGENTIC_PROJECT_ROOT)")
    parser.add_argument("--source-root", type=Path, help="Actual built-in ImageGen output root")
    parser.add_argument("--policy", type=Path, help="Optional explicit project policy; no automatic discovery")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_visible_identity and args.exact_identity:
        fail("--no-visible-identity cannot be combined with --exact-identity")
    root = project_root(args.root)
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path = manifest_path.resolve()
    if not inside(manifest_path, root):
        fail("manifest must be inside --root")
    if not manifest_path.is_file():
        fail(f"manifest not found: {manifest_path}")

    source = args.source_image.resolve()
    if not source.is_file():
        fail(f"source image not found: {source}")
    source_root = (args.source_root or os.environ.get("AGENTIC_IMAGEGEN_OUTPUT_ROOT")
                   or profile_value("imagegen", "outputRoot"))
    if not source_root:
        fail("Provide --source-root or AGENTIC_IMAGEGEN_OUTPUT_ROOT for built-in ImageGen provenance")
    generated_root = Path(source_root).expanduser().resolve()
    if not inside(source, generated_root):
        fail(f"source must be beneath built-in ImageGen output root: {generated_root}")
    width, height = png_dimensions(source)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [row for row in manifest.get("stills", []) if row.get("still_id") == args.still_id]
    if len(matches) != 1:
        fail(f"expected exactly one manifest record for {args.still_id}, found {len(matches)}")
    record = matches[0]
    status = record.get("status")
    if status not in REGISTERABLE:
        fail(f"{args.still_id} has non-registerable status {status!r}")

    policy = {}
    if args.policy:
        policy_path = (root / args.policy).resolve()
        if not inside(policy_path, root) or not policy_path.is_file():
            fail("Explicit policy must be an existing file inside the project")
        policy = load_json(policy_path)
    try:
        grounding_evidence = validate_registration_claims(
            root=root,
            policy=policy,
            still_id=args.still_id,
            execution_refs=list(
                record.get("imagegen_refs", record.get("refs", {}).get("all", []))
            ),
            exact_identity_claims=parse_claim_assignments(
                args.exact_identity, label="--exact-identity"
            ),
            continuity_fact_ids=args.grounding_fact,
            note=(args.note or "") if args.policy else "",
            row=record if args.policy else None,
            no_visible_identity=args.no_visible_identity,
        )
    except GroundingError as exc:
        fail(f"grounding guard rejected {args.still_id}: {exc}")

    target_value = record.get("target_output_path")
    if not isinstance(target_value, str) or not target_value.endswith(".png"):
        fail(f"{args.still_id} has invalid PNG target_output_path")
    target = (root / target_value).resolve()
    if not inside(target, root):
        fail("target_output_path escapes --root")
    sidecar = target.with_suffix(".imagegen.json")
    if target.exists() or sidecar.exists():
        fail(f"refusing to overwrite existing target or sidecar for {args.still_id}")

    now = datetime.now(timezone.utc).isoformat()
    source_digest = sha256(source)
    source_relative = source.relative_to(generated_root).as_posix()
    manifest_relative = manifest_path.relative_to(root).as_posix()
    execution_prompt = record.get("imagegen_prompt", record.get("prompt", ""))
    prompt_digest = hashlib.sha256(str(execution_prompt).encode("utf-8")).hexdigest()
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "kind": "built_in_imagegen_registration",
        "provider": "openai_builtin_imagegen",
        "tool": "image_gen",
        "still_id": args.still_id,
        "episode": manifest.get("episode"),
        "registered_at": now,
        "registered_by": "pipeline/register_builtin_imagegen.py",
        "source_image": source_relative,
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
        "imagegen_refs": record.get("imagegen_refs", record.get("refs", {}).get("all", [])),
        "exact_identity_claims": grounding_evidence["exact_identity_claims"],
        "manifest_identity_coverage": grounding_evidence["manifest_identity_coverage"],
        "continuity_facts": grounding_evidence["continuity_facts"],
        "execution_coverage_count": grounding_evidence["execution_coverage_count"],
        "no_visible_identity": args.no_visible_identity,
    }
    if args.note:
        provenance["note"] = args.note

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
        record["generation_provider"] = "openai_builtin_imagegen"
        record["provenance_sidecar"] = sidecar.relative_to(root).as_posix()
        record["grounding_claims"] = {
            "exact_identity_claims": grounding_evidence["exact_identity_claims"],
            "manifest_identity_coverage": grounding_evidence["manifest_identity_coverage"],
            "continuity_facts": grounding_evidence["continuity_facts"],
            "no_visible_identity": args.no_visible_identity,
        }
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

    print(
        json.dumps(
            {
                "episode": manifest.get("episode"),
                "still_id": args.still_id,
                "status": "generated",
                "target": target.relative_to(root).as_posix(),
                "sidecar": sidecar.relative_to(root).as_posix(),
                "sha256": source_digest,
                "dimensions": [width, height],
                "status_counts": manifest["status_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
