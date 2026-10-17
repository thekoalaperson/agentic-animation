#!/usr/bin/env python3
"""Write a still derived from a registered image or an accepted clip.

The parent must be a registered asset (a still sidecar, a reference pack or an
accepted clip's sidecar) whose recorded hash matches its bytes, and not a
retired one. Its origin, provider, maker or supplier and rights are carried
into the derived sidecar, so a crop of a still that may not be uploaded stays
marked that way. The grounding guard runs on the row's references plus the
parent, as a registration's does (``--policy``, ``--exact-identity``,
``--grounding-fact``, ``--no-visible-identity``, ``--storyboard``).

Operations:
  exact-frame  one frame of a video by index (decoded on the qualified GPU worker)
  crop         a WIDTHxHEIGHT region placed by --anchor or by --x/--y
  pad          a WIDTHxHEIGHT canvas holding the parent at --anchor, filled with --color
  scale        a resize to WIDTHxHEIGHT that keeps the parent's aspect ratio

The output goes to the manifest row's ``target_output_path`` with a
``.derive.json`` sidecar recording the parent path and hash, the operation and
its parameters, the tool hash and the output hash. The manifest row is updated
the way the registrar updates it, so production inventory reads it unchanged.
Nothing is overwritten. Crop, pad and scale run on any host.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from ._common import atomic_new_bytes, extract_exact_frame, project_root, sha256_file as sha256
    from .asset_registry import asset_key, load_registry
    from .register_builtin_imagegen import (
        REGISTERABLE, GroundingError, canon_root_path, fail, inside, load_json, manifest_order,
        manifest_record, parse_claim_assignments, resolve_manifest, validate_registration_claims,
        write_json_atomic, write_json_exclusive,
    )
else:
    from _common import atomic_new_bytes, extract_exact_frame, project_root, sha256_file as sha256
    from asset_registry import asset_key, load_registry
    from register_builtin_imagegen import (
        REGISTERABLE, GroundingError, canon_root_path, fail, inside, load_json, manifest_order,
        manifest_record, parse_claim_assignments, resolve_manifest, validate_registration_claims,
        write_json_atomic, write_json_exclusive,
    )

ANCHORS = {
    "center": (0.5, 0.5), "top": (0.5, 0.0), "bottom": (0.5, 1.0),
    "left": (0.0, 0.5), "right": (1.0, 0.5), "top-left": (0.0, 0.0),
    "top-right": (1.0, 0.0), "bottom-left": (0.0, 1.0), "bottom-right": (1.0, 1.0),
}
# Declarations a derived still inherits from its parent's sidecar, under the
# names a supplied-still sidecar uses.
INHERITED = ("origin", "provider", "maker", "supplied_by", "rights", "digital_source_type")


def size(value: str) -> tuple[int, int]:
    width, separator, height = value.lower().partition("x")
    if not separator or not width.isdigit() or not height.isdigit() or int(width) < 1 or int(height) < 1:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT, for example 1344x768")
    return int(width), int(height)


def placed(outer: tuple[int, int], inner: tuple[int, int], anchor: str) -> tuple[int, int]:
    fx, fy = ANCHORS[anchor]
    return round((outer[0] - inner[0]) * fx), round((outer[1] - inner[1]) * fy)


def derive_image(parent: Path, operation: str, *, out_size: tuple[int, int],
                 anchor: str | None = None, x: int | None = None, y: int | None = None,
                 color: str | None = None) -> tuple[bytes, dict[str, Any]]:
    """Return PNG bytes and the exact parameters used. Never guesses a placement."""
    from PIL import Image
    with Image.open(parent) as opened:
        image = opened.convert("RGBA" if opened.mode in ("RGBA", "LA", "P") else "RGB")
    width, height = out_size
    params: dict[str, Any] = {"width": width, "height": height,
                              "parentWidth": image.width, "parentHeight": image.height}
    if operation == "crop":
        if width > image.width or height > image.height:
            fail("crop size exceeds the parent image")
        if (x is None) != (y is None):
            fail("crop offsets need both --x and --y")
        if x is None and anchor is None:
            fail("crop needs an explicit --anchor or --x/--y")
        if x is not None and anchor is not None:
            fail("give --anchor or --x/--y, not both")
        if x is None:
            x, y = placed((image.width, image.height), (width, height), anchor)
            params["anchor"] = anchor
        if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
            fail("crop box falls outside the parent image")
        params.update(x=x, y=y)
        result = image.crop((x, y, x + width, y + height))
    elif operation == "pad":
        if width < image.width or height < image.height:
            fail("pad size is smaller than the parent image")
        if anchor is None or not color:
            fail("pad needs an explicit --anchor and --color")
        from PIL import ImageColor
        try:
            fill = ImageColor.getrgb(color)
        except ValueError:
            fail(f"unreadable --color: {color}")
        left, top = placed((width, height), (image.width, image.height), anchor)
        result = Image.new(image.mode, (width, height), fill if image.mode == "RGB" else (*fill[:3], 255))
        result.paste(image, (left, top))
        params.update(anchor=anchor, color=color, x=left, y=top)
    elif operation == "scale":
        if abs(width / height - image.width / image.height) > max(1 / width, 1 / height) * 2:
            fail("scale must keep the parent's aspect ratio; crop or pad first")
        result = image.resize((width, height), Image.Resampling.LANCZOS)
        params["resample"] = "lanczos"
    else:
        fail(f"unknown image operation: {operation}")
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue(), params


def tool_record(operation: str) -> dict[str, Any]:
    record: dict[str, Any] = {"path": "pipeline/derive_still.py", "sha256": sha256(Path(__file__))}
    if operation == "exact-frame":
        try:
            record["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True,
                                              text=True).stdout.splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            record["ffmpeg"] = None
    else:
        import PIL
        record["pillow"] = PIL.__version__
    return record


def registered_parent(root: Path, parent: Path, digest: str) -> dict[str, Any]:
    """Return the parent's registry row, refusing a loose, changed or retired file."""
    try:
        assets = load_registry(root)
    except ValueError as exc:
        fail(f"asset registry is inconsistent; reconcile it before deriving: {exc}")
    key = asset_key(parent.relative_to(root).as_posix(), root)
    row = assets.get(key)
    if row is None:
        fail(f"parent is not a registered asset; register it first: {key}")
    if row.get("retired"):
        fail(f"parent is a retired asset: {key}")
    if row.get("sha256") != digest:
        fail(f"parent's registered hash does not match its bytes: {key}")
    return row


def inherited(root: Path, sidecar_value: str) -> dict[str, Any]:
    """The origin, provider, maker or supplier and rights the parent declared."""
    record = load_json(root / sidecar_value)
    carried = {key: record[key] for key in INHERITED if key in record}
    carried["inherited_from"] = record.get("inherited_from", sidecar_value)
    return carried


def derive(*, root: Path, manifest_path: Path, still_id: str, parent: Path, operation: str,
           out_size: tuple[int, int] | None = None, frame: int | None = None,
           anchor: str | None = None, x: int | None = None, y: int | None = None,
           color: str | None = None, note: str | None = None,
           policy_path: Path | None = None, exact_identity: list[str] | tuple[str, ...] = (),
           grounding_fact: list[str] | tuple[str, ...] = (), no_visible_identity: bool = False,
           storyboard: Path | None = None, canon_root: Path | None = None,
           extractor=extract_exact_frame) -> dict[str, Any]:
    parent = (parent if parent.is_absolute() else root / parent).resolve()
    if not inside(parent, root) or not parent.is_file():
        fail("parent must be an existing file inside the project")
    if no_visible_identity and exact_identity:
        fail("--no-visible-identity cannot be combined with --exact-identity")
    parent_digest = sha256(parent)
    parent_row = registered_parent(root, parent, parent_digest)
    parent_value = parent.relative_to(root).as_posix()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest_record(manifest, still_id)
    if record.get("status") not in REGISTERABLE:
        fail(f"{still_id} has non-registerable status {record.get('status')!r}")
    policy: dict[str, Any] = {}
    if policy_path:
        resolved_policy = (root / policy_path).resolve()
        if not inside(resolved_policy, root) or not resolved_policy.is_file():
            fail("Explicit policy must be an existing file inside the project")
        policy = load_json(resolved_policy)
    row_refs = list(record.get("imagegen_refs", record.get("refs", {}).get("all", [])))
    try:
        grounding = validate_registration_claims(
            root=root, policy=policy, still_id=still_id,
            execution_refs=[*row_refs, parent_value],
            exact_identity_claims=parse_claim_assignments(exact_identity, label="--exact-identity"),
            continuity_fact_ids=grounding_fact,
            note=(note or "") if policy_path else "",
            row=record if policy_path else None,
            no_visible_identity=no_visible_identity,
            canon_root=canon_root, order=manifest_order(root, storyboard),
        )
    except GroundingError as exc:
        fail(f"grounding guard rejected {still_id}: {exc}")
    target_value = record.get("target_output_path")
    if not isinstance(target_value, str) or not target_value.endswith(".png"):
        fail(f"{still_id} has invalid PNG target_output_path")
    target = (root / target_value).resolve()
    if not inside(target, root):
        fail("target_output_path escapes --root")
    sidecar = target.with_suffix(".derive.json")
    if target.exists() or sidecar.exists():
        fail(f"refusing to overwrite existing target or sidecar for {still_id}")

    if operation == "exact-frame":
        if frame is None or frame < 0:
            fail("exact-frame needs a non-negative --frame index")
        if out_size or anchor or x is not None or color:
            fail("exact-frame takes only --frame")
        params: dict[str, Any] = {"frameIndex": frame}
        try:
            extractor(parent, frame, target)
        except Exception as exc:  # the worker rule and empty decodes both refuse
            target.unlink(missing_ok=True)
            fail(f"exact-frame refused: {exc}")
    else:
        if out_size is None:
            fail(f"{operation} needs --size WIDTHxHEIGHT")
        raw, params = derive_image(parent, operation, out_size=out_size, anchor=anchor,
                                   x=x, y=y, color=color)
        atomic_new_bytes(target, raw)
    output_digest = sha256(target)
    from PIL import Image
    with Image.open(target) as written:
        width, height = written.size
    now = datetime.now(timezone.utc).isoformat()
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "kind": "derived_still",
        "still_id": still_id,
        "episode": manifest.get("episode"),
        "derived_at": now,
        "parent": {"path": parent_value, "sha256": parent_digest,
                   "sidecar": {"path": parent_row["sidecar"],
                               "sha256": sha256(root / parent_row["sidecar"])}},
        **inherited(root, parent_row["sidecar"]),
        "operation": {"name": operation, **params},
        "tool": tool_record(operation),
        "target_output_path": target.relative_to(root).as_posix(),
        "target_sha256": output_digest,
        "width": width,
        "height": height,
        "manifest": manifest_path.relative_to(root).as_posix(),
        "refs": row_refs,
        "exact_identity_claims": grounding["exact_identity_claims"],
        "manifest_identity_coverage": grounding["manifest_identity_coverage"],
        "continuity_facts": grounding["continuity_facts"],
        "execution_coverage_count": grounding["execution_coverage_count"],
        "no_visible_identity": no_visible_identity,
    }
    if "continuity_fact_skips" in grounding:
        provenance["continuity_fact_skips"] = grounding["continuity_fact_skips"]
    if note:
        provenance["note"] = note
    try:
        write_json_exclusive(sidecar, provenance)
        record["status"] = "generated"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["generated_at"] = now
        record["generation_provider"] = "derived"
        if "origin" in provenance:
            record["still_origin"] = provenance["origin"]
        record["provenance_sidecar"] = sidecar.relative_to(root).as_posix()
        record["grounding_claims"] = {
            "exact_identity_claims": grounding["exact_identity_claims"],
            "manifest_identity_coverage": grounding["manifest_identity_coverage"],
            "continuity_facts": grounding["continuity_facts"],
            "no_visible_identity": no_visible_identity,
        }
        if "continuity_fact_skips" in grounding:
            record["grounding_claims"]["continuity_fact_skips"] = grounding["continuity_fact_skips"]
        record.pop("last_error", None)
        manifest["updated_at"] = now
        manifest["status_counts"] = dict(Counter(row.get("status") for row in manifest["stills"]))
        write_json_atomic(manifest_path, manifest)
    except Exception:
        sidecar.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    return {"still_id": still_id, "status": "generated",
            "target": provenance["target_output_path"],
            "sidecar": sidecar.relative_to(root).as_posix(),
            "sha256": output_digest, "dimensions": [width, height]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="Story generation_manifest.json")
    parser.add_argument("still_id", help="Manifest row that receives the derived still")
    parser.add_argument("parent", type=Path, help="Registered image or accepted clip in the project")
    parser.add_argument("--operation", required=True, choices=("exact-frame", "crop", "pad", "scale"))
    parser.add_argument("--root", type=Path, help="Private project root (or AGENTIC_PROJECT_ROOT)")
    parser.add_argument("--size", type=size, help="Output WIDTHxHEIGHT for crop, pad and scale")
    parser.add_argument("--frame", type=int, help="Frame index for exact-frame")
    parser.add_argument("--anchor", choices=sorted(ANCHORS), help="Explicit placement for crop and pad")
    parser.add_argument("--x", type=int, help="Crop left offset (with --y)")
    parser.add_argument("--y", type=int, help="Crop top offset (with --x)")
    parser.add_argument("--color", help="Pad fill, for example #000000")
    parser.add_argument("--note", help="Optional provenance note")
    parser.add_argument("--policy", type=Path, help="Optional explicit project policy; no discovery")
    parser.add_argument("--storyboard", type=Path,
                        help="Explicit storyboard whose order resolves from/to continuity facts")
    parser.add_argument("--canon-root", type=Path,
                        help="Read-only series canon for canon:<path> references (or profile stills.canonRoot)")
    parser.add_argument("--exact-identity", action="append", default=[], metavar="NAME=SOURCE",
                        help="Named exact-identity claim and its authority source; repeat as needed")
    parser.add_argument("--grounding-fact", action="append", default=[], metavar="FACT_ID",
                        help="Acknowledge a policy continuity fact; repeat for every applicable fact")
    parser.add_argument("--no-visible-identity", action="store_true",
                        help="Explicitly declare that the frame contains no identity-bearing character")
    args = parser.parse_args(argv)
    root = project_root(args.root)
    result = derive(root=root, manifest_path=resolve_manifest(root, args.manifest),
                    still_id=args.still_id, parent=args.parent, operation=args.operation,
                    out_size=args.size, frame=args.frame, anchor=args.anchor, x=args.x,
                    y=args.y, color=args.color, note=args.note, policy_path=args.policy,
                    exact_identity=args.exact_identity, grounding_fact=args.grounding_fact,
                    no_visible_identity=args.no_visible_identity, storyboard=args.storyboard,
                    canon_root=canon_root_path(args.canon_root))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
