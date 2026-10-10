#!/usr/bin/env python3
"""Non-generative, versioned motion/lighting polish for approved video sources.

Examples:
  # Whole clip 1.5x slower, native delivery FPS and synchronized audio.
  local_polish.py in.mp4 out.mp4 --slow 1.5

  # Keep the opening native; slow only source seconds 4 through 8.
  local_polish.py in.mp4 out.mp4 --slow 1.5 --window 4:8

  # Add a soft warm pulse and a two-frame lightning peak on output time.
  local_polish.py in.mp4 out.mp4 \
    --warm-flash 4.35,#ffc84a,0.24,0.70 \
    --lightning-peak 4.43,#fff2bf,0.90,2

Effect timestamps are measured on the retimed OUTPUT timeline. The script keeps
the source untouched and writes OUT.edit.json with exact lineage and filters.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import shlex
import subprocess
import sys


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "stream=index,codec_type,r_frame_rate,width,height,sample_rate,channels",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    fps_fraction = Fraction(video["r_frame_rate"])
    return {
        "duration": float(data["format"]["duration"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": float(fps_fraction),
        "fpsExpression": (str(fps_fraction.numerator) if fps_fraction.denominator == 1
                          else f"{fps_fraction.numerator}/{fps_fraction.denominator}"),
        "audio": audio,
    }


def atempo_chain(speed: float) -> str:
    if speed <= 0:
        raise ValueError("audio speed must be positive")
    parts = []
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    parts.append(f"atempo={speed:.9f}")
    return ",".join(parts)


def parse_window(value: str) -> tuple[float, float]:
    try:
        start, end = (float(v) for v in value.split(":", 1))
    except Exception as exc:
        raise argparse.ArgumentTypeError("window must be START:END") from exc
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("window requires 0 <= START < END")
    return start, end


def parse_effect(value: str, lightning: bool = False) -> dict:
    fields = value.split(",")
    if len(fields) != 4:
        kind = "START,COLOR,OPACITY,FRAMES" if lightning else "START,COLOR,OPACITY,DURATION"
        raise argparse.ArgumentTypeError(f"effect must be {kind}")
    start, color, opacity, tail = fields
    color = color.strip().lstrip("#")
    if len(color) != 6 or any(c not in "0123456789abcdefABCDEF" for c in color):
        raise argparse.ArgumentTypeError("COLOR must be a six-digit hex value")
    result = {"startSeconds": float(start), "color": f"#{color.lower()}",
              "opacity": float(opacity)}
    if result["startSeconds"] < 0 or not 0 < result["opacity"] <= 1:
        raise argparse.ArgumentTypeError("effect start/opacity is invalid")
    if lightning:
        result["frames"] = int(tail)
        if result["frames"] not in (1, 2):
            raise argparse.ArgumentTypeError("lightning peaks are limited to one or two frames")
    else:
        result["durationSeconds"] = float(tail)
        if result["durationSeconds"] <= 0:
            raise argparse.ArgumentTypeError("flash duration must be positive")
    return result


def build_timing(meta: dict, slow: float, window: tuple[float, float] | None,
                 preserve_audio: bool) -> tuple[list[str], str, str | None, float, dict]:
    fps = meta["fpsExpression"]
    duration = meta["duration"]
    interpolation = "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    filters: list[str] = []

    if slow == 1:
        filters.append(f"[0:v:0]setpts=PTS-STARTPTS,fps={fps}[vtime]")
        if preserve_audio:
            filters.append("[0:a:0]asetpts=PTS-STARTPTS[atime]")
        return filters, "vtime", "atime" if preserve_audio else None, duration, {
            "slowFactor": 1, "window": None, "intermediateFps": None,
        }

    if slow <= 1:
        raise ValueError("--slow is a duration multiplier and must be >= 1")

    if window is None:
        filters.append(
            f"[0:v:0]setpts=PTS-STARTPTS,{interpolation},setpts={slow:.9f}*PTS,fps={fps}[vtime]"
        )
        if preserve_audio:
            filters.append(
                f"[0:a:0]asetpts=PTS-STARTPTS,{atempo_chain(1 / slow)}[atime]"
            )
        return filters, "vtime", "atime" if preserve_audio else None, duration * slow, {
            "slowFactor": slow, "window": None, "intermediateFps": 60,
        }

    start, end = window
    if end > duration + 0.02:
        raise ValueError(f"window ends at {end:.3f}s but source is {duration:.3f}s")
    end = min(end, duration)
    segments: list[tuple[float, float, bool]] = []
    if start > 0:
        segments.append((0, start, False))
    segments.append((start, end, True))
    if end < duration - 0.001:
        segments.append((end, duration, False))

    count = len(segments)
    vsrc = [f"vsrc{n}" for n in range(count)]
    filters.append(f"[0:v:0]split={count}" + "".join(f"[{x}]" for x in vsrc))
    if preserve_audio:
        asrc = [f"asrc{n}" for n in range(count)]
        filters.append(f"[0:a:0]asplit={count}" + "".join(f"[{x}]" for x in asrc))
    else:
        asrc = []

    for n, (seg_start, seg_end, slowed) in enumerate(segments):
        vchain = f"[{vsrc[n]}]trim=start={seg_start:.9f}:end={seg_end:.9f},setpts=PTS-STARTPTS"
        if slowed:
            vchain += f",{interpolation},setpts={slow:.9f}*PTS"
        vchain += f",fps={fps}[vseg{n}]"
        filters.append(vchain)
        if preserve_audio:
            achain = f"[{asrc[n]}]atrim=start={seg_start:.9f}:end={seg_end:.9f},asetpts=PTS-STARTPTS"
            if slowed:
                achain += f",{atempo_chain(1 / slow)}"
            achain += f"[aseg{n}]"
            filters.append(achain)

    filters.append("".join(f"[vseg{n}]" for n in range(count)) +
                   f"concat=n={count}:v=1:a=0[vtime]")
    if preserve_audio:
        filters.append("".join(f"[aseg{n}]" for n in range(count)) +
                       f"concat=n={count}:v=0:a=1[atime]")
    output_duration = duration + (end - start) * (slow - 1)
    return filters, "vtime", "atime" if preserve_audio else None, output_duration, {
        "slowFactor": slow, "window": {"startSeconds": start, "endSeconds": end},
        "intermediateFps": 60,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--slow", type=float, default=1.0,
                        help="duration multiplier; 1.5 means 1.5x slower")
    parser.add_argument("--window", type=parse_window,
                        help="source window START:END to slow; omitted means whole clip")
    parser.add_argument("--warm-flash", action="append", default=[],
                        type=lambda v: parse_effect(v, False),
                        metavar="START,COLOR,OPACITY,DURATION")
    parser.add_argument("--lightning-peak", action="append", default=[],
                        type=lambda v: parse_effect(v, True),
                        metavar="START,COLOR,OPACITY,FRAMES")
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.output.exists() and not args.force:
        parser.error(f"output exists (use --force or version the filename): {args.output}")
    if args.window and args.slow == 1:
        parser.error("--window requires --slow greater than 1")

    meta = probe(args.input)
    preserve_audio = bool(meta["audio"]) and not args.mute
    filters, vlabel, alabel, output_duration, timing = build_timing(
        meta, args.slow, args.window, preserve_audio
    )

    current = vlabel
    effects = [*(('warm', e) for e in args.warm_flash),
               *(('lightning', e) for e in args.lightning_peak)]
    for n, (kind, effect) in enumerate(effects):
        color = "0x" + effect["color"].lstrip("#")
        if kind == "warm":
            length = effect["durationSeconds"]
            attack = min(0.12, length / 4)
            decay = max(0.001, length - attack)
            filters.append(
                f"color=c={color}@{effect['opacity']}:s={meta['width']}x{meta['height']}:"
                f"r={meta['fpsExpression']}:d={length},format=rgba,"
                f"fade=t=in:st=0:d={attack}:alpha=1,"
                f"fade=t=out:st={attack}:d={decay}:alpha=1,"
                f"setpts=PTS+{effect['startSeconds']}/TB[fx{n}]"
            )
        else:
            length = effect["frames"] / meta["fps"]
            filters.append(
                f"color=c={color}@{effect['opacity']}:s={meta['width']}x{meta['height']}:"
                f"r={meta['fpsExpression']}:d={length:.9f},format=rgba,"
                f"setpts=PTS+{effect['startSeconds']}/TB[fx{n}]"
            )
        filters.append(f"[{current}][fx{n}]overlay=eof_action=pass:shortest=0[vfx{n}]")
        current = f"vfx{n}"

    filters.append(f"[{current}]format=yuv420p[vout]")
    command = ["ffmpeg", "-y", "-i", str(args.input), "-filter_complex", ";".join(filters),
               "-map", "[vout]"]
    if preserve_audio and alabel:
        command += ["-map", f"[{alabel}]", "-c:a", "aac", "-b:a", "192k"]
    command += ["-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
                "-pix_fmt", "yuv420p", "-r", meta["fpsExpression"],
                "-movflags", "+faststart", str(args.output)]

    manifest = {
        "operation": "local-motion-and-illumination-polish",
        "source": str(args.input),
        "output": str(args.output),
        "sourceDurationSeconds": meta["duration"],
        "outputDurationSecondsExpected": output_duration,
        "deliveryFps": meta["fpsExpression"],
        "timing": timing,
        "warmFlashes": args.warm_flash,
        "lightningPeaks": args.lightning_peak,
        "audio": "muted" if args.mute else ("synchronized" if preserve_audio else "none"),
        "generativeCostUsd": 0,
        "command": command,
    }

    print(" ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    manifest_path = Path(str(args.output) + ".edit.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"local_polish: {exc}", file=sys.stderr)
        raise SystemExit(2)
