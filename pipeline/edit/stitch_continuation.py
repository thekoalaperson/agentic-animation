#!/usr/bin/env python3
"""Join a generated continuation without duplicating its anchor frame.

Image-to-video continuation normally conditions on the preceding clip's final
frame, so frame zero of the continuation repeats that boundary. This tool
removes the repeated frame, validates matching geometry/cadence, joins the
clips, and writes a lineage sidecar beside the output.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


def probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,pix_fmt,avg_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": stream.get("pix_fmt"),
        "fps": str(Fraction(stream["avg_frame_rate"])),
        "frames": int(stream["nb_read_frames"]),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("continuation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--keep-boundary-frame", action="store_true")
    parser.add_argument("--crf", type=int, default=15)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for path in (args.base, args.continuation):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    if args.output.exists() and not args.force:
        parser.error(f"output exists (use --force or version it): {args.output}")

    base = probe(args.base)
    continuation = probe(args.continuation)
    for field in ("width", "height", "fps"):
        if base[field] != continuation[field]:
            parser.error(
                f"{field} mismatch: base={base[field]} continuation={continuation[field]}"
            )

    skip = 0 if args.keep_boundary_frame else 1
    expected_frames = base["frames"] + continuation["frames"] - skip
    filter_graph = (
        "[0:v:0]setpts=PTS-STARTPTS[v0];"
        f"[1:v:0]trim=start_frame={skip},setpts=PTS-STARTPTS[v1];"
        "[v0][v1]concat=n=2:v=1:a=0,format=yuv420p[vout]"
    )
    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(args.base), "-i",
        str(args.continuation), "-filter_complex", filter_graph, "-map", "[vout]",
        "-an", "-c:v", "libx264", "-crf", str(args.crf), "-preset", args.preset,
        "-r", base["fps"], "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output),
    ]
    subprocess.run(command, check=True)

    result = probe(args.output)
    if result["frames"] != expected_frames:
        raise RuntimeError(
            f"frame-count verification failed: expected {expected_frames}, "
            f"found {result['frames']}"
        )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(args.output), "-f", "null", "-"],
        check=True,
    )
    manifest = {
        "operation": "continuation-stitch",
        "base": str(args.base),
        "continuation": str(args.continuation),
        "output": str(args.output),
        "boundaryFrameRemoved": not args.keep_boundary_frame,
        "baseFrames": base["frames"],
        "continuationFrames": continuation["frames"],
        "outputFrames": result["frames"],
        "width": result["width"],
        "height": result["height"],
        "fps": result["fps"],
        "sha256": sha256(args.output),
        "command": command,
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".stitch.json")
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output} ({result['frames']} frames at {result['fps']} fps)")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
