#!/usr/bin/env python3
"""Compose registered images into one labelled reference pack.

Each source must already be a registered asset whose recorded hash matches its
bytes. The pack is a contact image with one labelled cell per source, plus the
``.refpack.json`` sidecar that the still registrar validates as packed
execution coverage. Labels come from the policy's identity authorities (an
authority whose ``sources`` list the image), else from ``PATH=LABEL``, else
from the file name. Nothing is overwritten; this runs on any host.
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from ._common import atomic_new_bytes, project_root, sha256_file as sha256
    from .asset_registry import asset_key, load_registry
else:
    from _common import atomic_new_bytes, project_root, sha256_file as sha256
    from asset_registry import asset_key, load_registry

CELL = 512
CAPTION = 40


def policy_labels(policy: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for authority in policy.get("grounding_guard", {}).get("identity_authorities", []):
        for source in authority.get("sources", []):
            labels.setdefault(str(source), str(authority.get("name", "")))
    return labels


def compose(paths: list[Path], labels: list[str]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    columns = min(len(paths), 3)
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * CELL, rows * (CELL + CAPTION)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:  # Pillow without FreeType sizing
        font = ImageFont.load_default()
    for index, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        image.thumbnail((CELL, CELL))
        left = (index % columns) * CELL
        top = (index // columns) * (CELL + CAPTION)
        sheet.paste(image, (left + (CELL - image.width) // 2, top + (CELL - image.height) // 2))
        draw.text((left + 8, top + CELL + 8), label, fill="black", font=font)
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue()


def make_pack(*, root: Path, output: Path, sources: list[str],
              policy: dict[str, Any] | None = None) -> dict[str, Any]:
    root = root.resolve()
    output = (output if output.is_absolute() else root / output).resolve()
    if not output.is_relative_to(root) or output.suffix != ".png":
        raise ValueError("pack must be a .png path inside the project")
    sidecar = output.with_suffix(".refpack.json")
    if output.exists() or sidecar.exists():
        raise ValueError(f"refusing to overwrite an existing pack: {output.relative_to(root)}")
    assets = load_registry(root)
    from_policy = policy_labels(policy or {})
    entries, paths, labels = [], [], []
    problems = []
    for raw in sources:
        value, _, explicit = raw.partition("=")
        key = asset_key(value, root)
        path = (root / key).resolve()
        row = assets.get(key)
        if row is None:
            problems.append(f"not a registered asset: {value}")
            continue
        if not path.is_file() or row.get("sha256") != sha256(path):
            problems.append(f"registered hash does not match the bytes: {value}")
            continue
        label = from_policy.get(key) or explicit.strip() or Path(key).stem
        entries.append({"source": key, "source_sha256": row["sha256"], "label": label})
        paths.append(path)
        labels.append(label)
    if problems:
        raise ValueError("; ".join(problems))
    if not entries:
        raise ValueError("a reference pack needs at least one source")
    atomic_new_bytes(output, compose(paths, labels))
    record = {
        "schema_version": 1,
        "kind": "built_in_imagegen_reference_pack",
        "pack": output.relative_to(root).as_posix(),
        "pack_sha256": sha256(output),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "pipeline/make_reference_pack.py",
        "sources": entries,
    }
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="Pack PNG inside the project, e.g. art/refs/pack-a.png")
    parser.add_argument("sources", nargs="+", help="Registered project image paths, optionally PATH=LABEL")
    parser.add_argument("--root", type=Path, help="Private project root (or AGENTIC_PROJECT_ROOT)")
    parser.add_argument("--policy", type=Path, help="Optional explicit policy for labels")
    args = parser.parse_args(argv)
    root = project_root(args.root)
    policy = None
    if args.policy:
        path = (root / args.policy).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit("Explicit policy must be an existing file inside the project")
        policy = json.loads(path.read_text(encoding="utf-8"))
    try:
        record = make_pack(root=root, output=args.output, sources=args.sources, policy=policy)
    except ValueError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps({"pack": record["pack"], "sha256": record["pack_sha256"],
                      "sources": len(record["sources"])}))


if __name__ == "__main__":
    main()
