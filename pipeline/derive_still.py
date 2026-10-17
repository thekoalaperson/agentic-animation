#!/usr/bin/env python3
"""Write a still derived from a registered image or an accepted clip.

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
    from .register_builtin_imagegen import (
        REGISTERABLE, fail, inside, manifest_record, resolve_manifest,
        write_json_atomic, write_json_exclusive,
    )
else:
    from _common import atomic_new_bytes, extract_exact_frame, project_root, sha256_file as sha256
    from register_builtin_imagegen import (
        REGISTERABLE, fail, inside, manifest_record, resolve_manifest,
        write_json_atomic, write_json_exclusive,
    )

ANCHORS = {
    "center": (0.5, 0.5), "top": (0.5, 0.0), "bottom": (0.5, 1.0),
    "left": (0.0, 0.5), "right": (1.0, 0.5), "top-left": (0.0, 0.0),
    "top-right": (1.0, 0.0), "bottom-left": (0.0, 1.0), "bottom-right": (1.0, 1.0),
}
PARENT_SIDECARS = (".imagegen.json", ".still.json", ".derive.json", ".vid.json",
                   ".ingest.json", ".kie.json")


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


def parent_sidecar(root: Path, parent: Path) -> dict[str, str] | None:
    stem = parent.with_suffix("")
    found = [Path(str(stem) + suffix) for suffix in PARENT_SIDECARS
             if Path(str(stem) + suffix).is_file()]
    if len(found) > 1:
        fail(f"parent has more than one sidecar: {[p.name for p in found]}")
    if not found:
        return None
    return {"path": found[0].relative_to(root).as_posix(), "sha256": sha256(found[0])}


def derive(*, root: Path, manifest_path: Path, still_id: str, parent: Path, operation: str,
           out_size: tuple[int, int] | None = None, frame: int | None = None,
           anchor: str | None = None, x: int | None = None, y: int | None = None,
           color: str | None = None, note: str | None = None,
           extractor=extract_exact_frame) -> dict[str, Any]:
    parent = (parent if parent.is_absolute() else root / parent).resolve()
    if not inside(parent, root) or not parent.is_file():
        fail("parent must be an existing file inside the project")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest_record(manifest, still_id)
    if record.get("status") not in REGISTERABLE:
        fail(f"{still_id} has non-registerable status {record.get('status')!r}")
    target_value = record.get("target_output_path")
    if not isinstance(target_value, str) or not target_value.endswith(".png"):
        fail(f"{still_id} has invalid PNG target_output_path")
    target = (root / target_value).resolve()
    if not inside(target, root):
        fail("target_output_path escapes --root")
    sidecar = target.with_suffix(".derive.json")
    if target.exists() or sidecar.exists():
        fail(f"refusing to overwrite existing target or sidecar for {still_id}")
    parent_digest = sha256(parent)

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
        "parent": {"path": parent.relative_to(root).as_posix(), "sha256": parent_digest,
                   "sidecar": parent_sidecar(root, parent)},
        "operation": {"name": operation, **params},
        "tool": tool_record(operation),
        "target_output_path": target.relative_to(root).as_posix(),
        "target_sha256": output_digest,
        "width": width,
        "height": height,
        "manifest": manifest_path.relative_to(root).as_posix(),
    }
    if note:
        provenance["note"] = note
    try:
        write_json_exclusive(sidecar, provenance)
        record["status"] = "generated"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["generated_at"] = now
        record["generation_provider"] = "derived"
        record["provenance_sidecar"] = sidecar.relative_to(root).as_posix()
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
    args = parser.parse_args(argv)
    root = project_root(args.root)
    result = derive(root=root, manifest_path=resolve_manifest(root, args.manifest),
                    still_id=args.still_id, parent=args.parent, operation=args.operation,
                    out_size=args.size, frame=args.frame, anchor=args.anchor, x=args.x,
                    y=args.y, color=args.color, note=args.note)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
