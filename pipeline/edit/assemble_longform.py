#!/usr/bin/env python3
"""Build a deterministic long-form picture, Hindi voice stem, and wave-shaped score.

The episode owns the edit plan; this shared renderer owns media mechanics. It
does not generate media, retime speech, or introduce still-image padding.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np


SR = 48_000
FPS = 24


def run(cmd, **kwargs):
    return subprocess.run([str(x) for x in cmd], check=True, **kwargs)


def probe_duration(path: Path) -> float:
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ], text=True).strip())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bound_path(ep: Path, raw: str, label: str) -> Path:
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (ep / path).resolve()
    try:
        resolved.relative_to(ep.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes episode root: {resolved}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{label} is missing: {resolved}")
    return resolved


def verify_sha(path: Path, expected: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected or "").lower()):
        raise RuntimeError(f"{label} has no valid declared SHA-256")
    actual = sha(path)
    if actual != str(expected).lower():
        raise RuntimeError(f"{label} hash changed: {path}")
    return actual


def audio(path: Path, filters: str | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0"]
    if filters:
        cmd += ["-af", filters]
    proc = run(cmd + ["-ar", SR, "-ac", 2, "-f", "f32le", "pipe:1"], capture_output=True)
    return np.frombuffer(proc.stdout, np.float32).reshape(-1, 2).copy()


def write_audio(path: Path, samples: np.ndarray) -> None:
    partial = path.with_suffix(".partial.wav")
    run([
        "ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", SR,
        "-ac", 2, "-i", "pipe:0", "-c:a", "pcm_s24le", partial,
    ], input=samples.astype(np.float32).tobytes())
    partial.replace(path)


def place(dst: np.ndarray, src: np.ndarray, at: float) -> None:
    start = max(0, round(at * SR))
    end = min(len(dst), start + len(src))
    if end > start:
        dst[start:end] += src[: end - start]


def fade(samples: np.ndarray, seconds: float) -> np.ndarray:
    n = min(len(samples) // 2, round(seconds * SR))
    if n:
        curve = np.sin(np.linspace(0, math.pi / 2, n, dtype=np.float32))[:, None]
        samples[:n] *= curve
        samples[-n:] *= curve[::-1]
    return samples


def validate_picture_ownership(plan: dict, segments: list[dict]) -> None:
    """Keep an authored shared-picture chain on exactly its complete utterances."""
    contracts=plan.get('pictureOwnershipContracts',[])
    if not isinstance(contracts,list):
        raise RuntimeError('Picture ownership contracts must be an array')
    for contract in contracts:
        owner=contract.get('ownerSceneId')
        ids=contract.get('coveredSceneIds',[])
        events=plan.get('events',[])
        positions=[i for i,e in enumerate(events) if e['id'] in ids]
        if (not owner or not ids or len(set(ids))!=len(ids) or len(positions)!=len(ids)
                or positions!=list(range(positions[0],positions[0]+len(ids)))
                or [events[i]['id'] for i in positions]!=ids
                or any(events[i]['kind'] not in ('narration','picture') for i in positions)):
            raise RuntimeError('Picture ownership must cover ordered contiguous narration or picture events')
        start=round(events[positions[0]]['start']*FPS)
        end=round(events[positions[-1]]['end']*FPS)
        owned=[s for s in segments if s.get('sceneId')==owner]
        cursor=start
        for segment in owned:
            if round(segment['timelineStart']*FPS)!=cursor:
                raise RuntimeError('Picture ownership starts late/early or has a gap')
            cursor+=segment['frames']
        if not owned or cursor!=end:
            raise RuntimeError('Picture ownership no longer matches its complete narration events')


def build_picture_segments(ep: Path, plan: dict) -> list[dict]:
    result = []
    cursor_frames = 0
    for block in plan["pictureBlocks"]:
        block_end_frames = round((block["start"] + block["duration"]) * FPS)
        remaining_frames = block_end_frames - cursor_frames
        if block["kind"] == "dialogue-fixed-sync":
            candidate = block["candidates"][0]
            if isinstance(candidate, str):
                candidate = {"path": candidate}
            # A picture window names its own source, hash and role; the native
            # part keeps the fixed-sync role it always had.
            role = candidate["role"] if candidate.get("pictureWindow") else "dialogue-fixed-sync"
            candidates = [dict(candidate, role=role)]
        else:
            candidates = block["candidates"]
        for candidate in candidates:
            if remaining_frames <= 0:
                break
            source = bound_path(ep, candidate["path"], f"picture source {block['id']}")
            source_sha = verify_sha(
                source, candidate.get("sourceSha256", ""), f"picture source {block['id']}"
            )
            source_start = float(candidate.get("sourceStart", 0.0))
            speed = float(candidate.get("speed", 1.0))
            if not math.isfinite(speed) or not 0.5 <= speed <= 4.0:
                raise RuntimeError(
                    f"Unsafe explicit speed in {block['id']}: {speed}; allowed 0.5..4.0"
                )
            natural = probe_duration(source)
            if not 0 <= source_start < natural:
                raise RuntimeError(
                    f"Invalid sourceStart in {block['id']}: {source_start} for {source} ({natural}s)"
                )
            output_duration = float(candidate.get("duration", (natural - source_start) / speed))
            source_duration = float(
                candidate.get("sourceDuration", output_duration * speed)
            )
            source_duration = min(source_duration, natural - source_start)
            if source_duration <= 0:
                raise RuntimeError(f"Empty source window in {block['id']}: {source}")
            declared_frames = int(candidate.get("frames", round(output_duration * FPS)))
            frames = min(remaining_frames, declared_frames)
            use = frames / FPS
            if frames <= 0:
                continue
            used_source_duration = min(source_duration, use * speed)
            rendered = {
                "id": f"p{len(result)+1:03d}", "block": block["id"],
                "source": str(source), "sourceStart": source_start,
                "sourceDuration": used_source_duration, "speed": speed,
                "duration": use, "frames": frames, "timelineStart": cursor_frames / FPS,
                "role": candidate.get("role", "picture"), "sourceSha256": source_sha,
            }
            if candidate.get('sceneId'):
                rendered['sceneId']=candidate['sceneId']
            if candidate.get('crop') is not None:
                crop = candidate['crop']
                if (not isinstance(crop, dict) or set(crop) != {'x', 'y', 'width', 'height'}
                        or any(isinstance(v, bool) or not isinstance(v, int) or v % 2 for v in crop.values())
                        or crop['x'] < 0 or crop['y'] < 0 or crop['width'] < 128 or crop['height'] < 72):
                    raise RuntimeError('Invalid explicit picture crop')
                rendered['crop'] = dict(crop)
            if candidate.get('continuationOf') is not None:
                link=candidate['continuationOf']
                if not isinstance(link,dict) or not result:
                    raise RuntimeError('Continuation has no immediate rendered predecessor')
                parent=result[-1]
                parent_source=bound_path(ep,link.get('source',''),'continuation parent')
                endpoint=link.get('endFrameExclusive')
                actual_endpoint=(parent['sourceStart']+parent['sourceDuration'])*FPS
                if (isinstance(endpoint,bool) or not isinstance(endpoint,int)
                        or str(parent_source)!=parent['source']
                        or link.get('sourceSha256')!=parent['sourceSha256']
                        or abs(actual_endpoint-endpoint)>1e-3
                        or abs(source_start*FPS-1)>1e-3
                        or speed!=1.0 or parent['speed']!=1.0):
                    raise RuntimeError('Continuation parent source/hash/endpoint or native cadence changed before rendering')
                rendered['continuationOf']=dict(link)
            result.append(rendered)
            cursor_frames += frames
            remaining_frames -= frames
        if remaining_frames:
            raise RuntimeError(f"Picture deficit in {block['id']}: {remaining_frames} frames")
    validate_picture_ownership(plan,result)
    return result


TRANSITION_OUT_KEYS = ("fadeOut", "dissolveOut")
TRANSITION_IN_KEYS = ("fadeIn", "dissolveIn")
DIP_COLOURS = {"dip-to-black": "black", "dip-to-white": "white"}


def apply_transitions(segments: list[dict], transitions: list[dict]) -> None:
    """Mark the two picture segments on each side of an authored transition.

    A dip fades the outgoing segment's last frames to a colour and the incoming
    segment's first frames up from it. A dissolve names how many unused frames
    each side lends to the overlap; ``group_dissolve_runs`` renders the pair
    as one run. Segments without a transition are left exactly as they were.
    """
    for transition in transitions:
        boundary = transition.get("timelineFrame")
        cut = next((k for k in range(len(segments) - 1)
                    if round((segments[k]["timelineStart"] + segments[k]["duration"]) * FPS) == boundary
                    and round(segments[k + 1]["timelineStart"] * FPS) == boundary), None)
        if isinstance(boundary, bool) or not isinstance(boundary, int) or cut is None:
            raise RuntimeError(f"Transition after {transition.get('after')} has no picture cut to sit on")
        outgoing, incoming = segments[cut], segments[cut + 1]
        kind, frames = transition.get("kind"), transition.get("frames")
        shown_out, shown_in = transition.get("outgoingFrames"), transition.get("incomingFrames")
        tail, head = transition.get("outgoingUnusedTailFrames"), transition.get("incomingUnusedHeadFrames")
        counts = (frames, shown_out, shown_in, tail, head)
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts) or frames < 2
                or shown_out > outgoing["frames"] or shown_in > incoming["frames"]):
            raise RuntimeError(f"Transition after {transition.get('after')} has invalid frame counts")
        if kind in DIP_COLOURS:
            if (tail, head) != (0, 0) or shown_out + shown_in != frames or shown_out >= outgoing["frames"]:
                raise RuntimeError("A dip uses only frames the cut already shows, ending on its colour")
            if shown_out:
                outgoing["fadeOut"] = {"frames": shown_out, "colour": DIP_COLOURS[kind]}
            if shown_in:
                incoming["fadeIn"] = {"frames": shown_in, "colour": DIP_COLOURS[kind]}
        elif kind == "dissolve":
            if tail + head != frames or (shown_out, shown_in) != (head, tail):
                raise RuntimeError("A dissolve overlaps exactly the unused frames each side lends")
            outgoing["dissolveOut"] = {"frames": frames, "unusedTailFrames": tail}
            incoming["dissolveIn"] = {"frames": frames, "unusedHeadFrames": head}
        else:
            raise RuntimeError(f"Unknown transition kind {kind!r}")


def group_dissolve_runs(runs: list[dict]) -> list[dict]:
    """Render runs joined by dissolves as one run of two or more sources.

    Each member keeps its own source window and gains only the unused frames
    it lends to the overlap. The joined run covers exactly the members'
    delivery frames, so the concat list and the frame-count guard are unchanged.
    """
    result: list[dict] = []
    for run in runs:
        if result and "dissolveIn" in run:
            group = result[-1]
            if "dissolveOut" not in (group["sources"][-1] if "sources" in group else group):
                raise RuntimeError("Dissolve has no outgoing run beside it")
            if "sources" not in group:
                group = {"sources": [dict(group)], "dissolves": []}
                result[-1] = group
            group["sources"].append(dict(run))
            group["dissolves"].append(int(run["dissolveIn"]["frames"]))
            continue
        result.append(run)
    grouped = []
    for item in result:
        if "sources" not in item:
            # An authored dissolve renders or refuses; it never falls back to a cut.
            if any(key in item for key in ("dissolveIn", "dissolveOut")):
                raise RuntimeError(f"Dissolve at {item['id']} has no run on its other side")
            grouped.append(item)
            continue
        parts = []
        for part in item["sources"]:
            speed = float(part["speed"])
            head = int(part.get("dissolveIn", {}).get("unusedHeadFrames", 0))
            tail = int(part.get("dissolveOut", {}).get("unusedTailFrames", 0))
            extended = dict(part, sourceStart=float(part["sourceStart"]) - head * speed / FPS,
                            sourceDuration=float(part["sourceDuration"]) + (head + tail) * speed / FPS,
                            frames=int(part["frames"]) + head + tail, shownFrames=int(part["frames"]))
            if extended["sourceStart"] < -1e-9:
                raise RuntimeError("Dissolve head reaches before its source")
            parts.append(extended)
        first, last = item["sources"][0], item["sources"][-1]
        grouped.append({
            "id": f"{first['id']}-dissolve-{last['id']}",
            "role": "dissolve-transition-run",
            "timelineStart": first["timelineStart"],
            "duration": sum(float(p["duration"]) for p in item["sources"]),
            "frames": sum(int(p["frames"]) for p in item["sources"]),
            "sources": parts,
            "dissolves": item["dissolves"],
            "memberSegmentIds": [m for p in item["sources"] for m in p["memberSegmentIds"]],
            "memberBlocks": [m for p in item["sources"] for m in p["memberBlocks"]],
            "memberRoles": [m for p in item["sources"] for m in p["memberRoles"]],
        })
    return grouped


def coalesce_render_segments(segments: list[dict]) -> list[dict]:
    """Merge exact adjacent windows for rendering while preserving the role ledger.

    ``picture-segments.json`` remains scene/block granular because downstream SFX
    anchors use those semantic roles. Encoding two source-contiguous windows
    separately creates an avoidable GOP/decoder boundary and can make a true H3
    continuation appear to restart. A render run is merged only when its source
    bytes, speed, source clock and delivery clock are all contiguous.
    """
    runs: list[dict] = []
    for segment in segments:
        current = dict(segment)
        current["memberSegmentIds"] = [segment["id"]]
        current["memberBlocks"] = [segment["block"]]
        current["memberRoles"] = [segment.get("role", "picture")]
        if not runs:
            runs.append(current)
            continue
        prior = runs[-1]
        same_source = (
            prior["source"] == current["source"]
            and prior["sourceSha256"] == current["sourceSha256"]
        )
        source_contiguous = abs(
            float(prior["sourceStart"])
            + float(prior["sourceDuration"])
            - float(current["sourceStart"])
        ) <= 1e-6
        timeline_contiguous = abs(
            float(prior["timelineStart"])
            + float(prior["duration"])
            - float(current["timelineStart"])
        ) <= 1e-6
        same_speed = abs(float(prior["speed"]) - float(current["speed"])) <= 1e-9
        # An authored transition is a boundary the picture must keep.
        transition_between = any(key in prior for key in TRANSITION_OUT_KEYS) or any(
            key in current for key in TRANSITION_IN_KEYS)
        if not (same_source and source_contiguous and timeline_contiguous and same_speed
                and prior.get('crop') == current.get('crop')) or transition_between:
            runs.append(current)
            continue
        for key in TRANSITION_OUT_KEYS:
            if key in current:
                prior[key] = current[key]
        prior["id"] = f"{prior['memberSegmentIds'][0]}-{current['id']}"
        prior["sourceDuration"] = float(prior["sourceDuration"]) + float(
            current["sourceDuration"]
        )
        prior["duration"] = float(prior["duration"]) + float(current["duration"])
        prior["frames"] = int(prior["frames"]) + int(current["frames"])
        prior["memberSegmentIds"].extend(current["memberSegmentIds"])
        prior["memberBlocks"].extend(current["memberBlocks"])
        prior["memberRoles"].extend(current["memberRoles"])
        prior["role"] = "source-contiguous-role-preserving-render-run"
    return runs


def frame_filters(frame: dict | None, width: int, height: int) -> tuple[str, str]:
    """Filters for an authored film-wide matte and stepped cadence.

    ``aspectRatio`` crops the fitted picture to that shape and letterboxes or
    pillarboxes it back into the unchanged delivery container. ``cadence``
    holds each picture frame for that many delivery frames; the delivery clock
    and frame count are unchanged.
    """
    if not frame:
        return "", ""
    matte = ""
    ratio = frame.get("aspectRatio")
    if ratio is not None:
        if ratio > width / height:
            inner = max(2, round(width / ratio / 2) * 2)
            matte = f"crop={width}:{inner}:0:{(height - inner) // 2},pad={width}:{height}:0:{(height - inner) // 2}:black,"
        elif ratio < width / height:
            inner = max(2, round(height * ratio / 2) * 2)
            matte = f"crop={inner}:{height}:{(width - inner) // 2}:0,pad={width}:{height}:{(width - inner) // 2}:0:black,"
    cadence = frame.get("cadence")
    stepped = f"fps={FPS}/{cadence}:eof_action=pass,fps={FPS}," if cadence else ""
    return matte, stepped


def render_segment(item: dict, cache: Path, width: int, height: int, frame: dict | None = None) -> Path:
    if "sources" in item:
        return render_dissolve_run(item, cache, width, height, frame)
    identity = {
        "source": item["sourceSha256"], "start": item["sourceStart"],
        "sourceDuration": item["sourceDuration"], "speed": item["speed"],
        "frames": item["frames"], "width": width, "height": height,
        "crop": item.get("crop"),
    }
    fades = {key: item[key] for key in ("fadeIn", "fadeOut") if key in item}
    if fades:
        identity["fades"] = fades  # absent for an ordinary cut, so its cache key is unchanged
    if frame:
        identity["frame"] = frame
    matte, stepped = frame_filters(frame, width, height)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    output = cache / f"{item['id']}-{key}.mp4"
    if output.exists():
        return output
    partial = output.with_suffix(".partial.mp4")
    crop_filter = ""
    if item.get("crop") is not None:
        crop = item["crop"]
        geometry = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", item["source"]], text=True))["streams"][0]
        if (crop["x"] + crop["width"] > geometry["width"]
                or crop["y"] + crop["height"] > geometry["height"]):
            raise RuntimeError("Authored crop exceeds actual source geometry")
        crop_filter = f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']},"
    vf = (
        f"trim=duration={item['sourceDuration']:.9f},"
        f"setpts=(PTS-STARTPTS)/{item['speed']:.9f},"
        f"{crop_filter}"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"{matte}fps={FPS},{stepped}"
        + "".join(f"fade=t=in:s=0:n={fade['frames']}:c={fade['colour']}," for fade in [item.get("fadeIn")] if fade)
        # ffmpeg reaches the colour one frame after a fade-out's last frame, so
        # start one frame earlier: the segment's final frame is then the colour,
        # as a fade-in's first frame is.
        + "".join(f"fade=t=out:s={max(0, int(item['frames']) - fade['frames'] - 1)}:n={fade['frames']}:c={fade['colour']},"
                  for fade in [item.get("fadeOut")] if fade)
        + "format=yuv420p"
    )
    run([
        "ffmpeg", "-y", "-v", "error", "-ss", item["sourceStart"],
        "-i", item["source"], "-an", "-vf", vf, "-frames:v", item["frames"],
        "-c:v", "libx264", "-preset", "veryfast", "-crf", 18,
        "-threads", 3, "-g", FPS * 2, "-pix_fmt", "yuv420p", partial,
    ])
    partial.replace(output)
    return output


def render_dissolve_run(item: dict, cache: Path, width: int, height: int, frame: dict | None = None) -> Path:
    """Render each source of a dissolve run, then overlap them with ``xfade``."""
    parts = [render_segment(dict(part, id=f"{item['id']}-s{index}"), cache, width, height, frame)
             for index, part in enumerate(item["sources"])]
    key = hashlib.sha256(json.dumps({"parts": [path.name for path in parts], "dissolves": item["dissolves"],
                                     "frames": item["frames"]}).encode()).hexdigest()[:20]
    output = cache / f"{item['id']}-{key}.mp4"
    if output.exists():
        return output
    graph, label, length = [], "[i0]", int(item["sources"][0]["frames"])
    for index in range(len(parts)):
        graph.append(f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={FPS}[i{index}]")
    for index, frames in enumerate(item["dissolves"], 1):
        offset = length - frames
        graph.append(f"{label}[i{index}]xfade=transition=fade:duration={frames / FPS:.9f}:"
                     f"offset={offset / FPS:.9f}[x{index}]")
        label = f"[x{index}]"
        length += int(item["sources"][index]["frames"]) - frames
    if length != int(item["frames"]):
        raise RuntimeError(f"Dissolve run {item['id']} does not cover its delivery frames")
    partial = output.with_suffix(".partial.mp4")
    inputs = [arg for path in parts for arg in ("-i", path)]
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex",
         ";".join(graph) + f";{label}format=yuv420p[out]", "-map", "[out]", "-an",
         "-frames:v", item["frames"], "-c:v", "libx264", "-preset", "veryfast", "-crf", 18,
         "-threads", 3, "-g", FPS * 2, "-pix_fmt", "yuv420p", partial])
    partial.replace(output)
    return output


def build_picture(ep: Path, work: Path, plan: dict, width: int, height: int) -> Path:
    segments = build_picture_segments(ep, plan)
    if plan.get("transitions"):
        apply_transitions(segments, plan["transitions"])
    (work / "picture-segments.json").write_text(json.dumps(segments, indent=2) + "\n")
    render_runs = coalesce_render_segments(segments)
    if plan.get("transitions"):
        render_runs = group_dissolve_runs(render_runs)
    (work / "picture-render-runs.json").write_text(
        json.dumps(render_runs, indent=2) + "\n"
    )
    cache = work / f"segments-{height}p"
    cache.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rendered = list(
            pool.map(lambda x: render_segment(x, cache, width, height, plan.get("frame")), render_runs)
        )
    listing = work / f"concat-{height}p.txt"
    # MP4 format.duration rounds some24fps clips to milliseconds. Repeated
    # implicit concat durations accumulate a picture/audio drift across a film.
    # Author each start from its exact delivery-frame count instead.
    listing.write_text("".join(f"file '{path}'\nduration {item['frames']/FPS:.12f}\n"
                               for path, item in zip(rendered, render_runs)))
    output = work / f"picture-{height}p.mp4"
    partial = output.with_suffix(".partial.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", 0, "-i", listing,
         "-c", "copy", "-bsf:v", "setts=pts=round(PTS*TB*24)/(TB*24):dts=round(DTS*TB*24)/(TB*24)",
         "-an", "-movflags", "+faststart", partial])
    # Explicit durations can still round a join by one stream tick. Quantize
    # timestamps only, then fail before publishing a non-24fps body. Pixels and
    # compressed payload remain unchanged; every source cache stays reusable.
    probe=json.loads(subprocess.check_output([
        "ffprobe","-v","error","-show_streams","-of","json",partial],text=True))['streams']
    if len(probe)!=1 or probe[0]['codec_type']!='video' or probe[0]['avg_frame_rate']!='24/1' or int(probe[0]['nb_frames'])!=sum(r['frames'] for r in render_runs):
        raise RuntimeError('Assembled picture lost exact frame count or24fps clock; source caches preserved')
    partial.replace(output)
    return output


def place_audio_overlaps(stem: np.ndarray, source: Path, event: dict, source_start: float,
                         gain: float) -> dict:
    """Play an authored lead or tail of the event's own native audio across a cut.

    The kept window is decoded, levelled and placed exactly as without an
    overlap; each overlap is decoded on its own from the native audio beyond
    that window, at the same gain, and ramped with the same sine curve as
    ``fade`` so the bed enters or leaves smoothly. It never exceeds its
    authored length, so the "voice exceeds event" bound still holds.
    """
    placed = {}
    for key in ("audioLead", "audioTail"):
        overlap = event.get(key)
        if not overlap:
            continue
        seconds = float(overlap["seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError(f"{key} of {event['id']} must be positive")
        if key == "audioLead":
            start, timeline_at = source_start - seconds, float(event["start"]) - seconds
        else:
            start = source_start + float(event["audioSourceDuration"])
            timeline_at = float(event["start"]) + float(event["audioSourceDuration"])
        if start < -1e-9 or timeline_at < -1e-9:
            raise RuntimeError(f"{key} of {event['id']} reaches before its clip or the film")
        piece = audio(source, f"atrim=start={max(start, 0):.9f}:duration={seconds:.9f},asetpts=PTS-STARTPTS")
        room = round(seconds * SR)
        if len(piece) > room + 1024:
            raise RuntimeError(f"{key} of {event['id']} exceeds its authored length")
        piece = piece[:room] * gain
        n = len(piece)
        if n:
            curve = np.sin(np.linspace(0, math.pi / 2, n, dtype=np.float32))[:, None]
            piece *= curve if key == "audioLead" else curve[::-1]
        place(stem, piece, timeline_at)
        placed[key + "Seconds"] = round(n / SR, 6)
    return placed


def build_voice(ep: Path, work: Path, plan: dict) -> Path:
    total = round(plan["runtimeSeconds"] * SR)
    stem = np.zeros((total, 2), np.float32)
    levels = []
    for event in plan["events"]:
        if event.get("kind") == "picture":
            # Wordless picture carries no voice. Keep its row so the level audit
            # still lists every event in clock order; the stem stays silent here.
            if event.get("audio") is not None:
                raise RuntimeError(f"picture event {event['id']} must not carry voice audio")
            levels.append({"event": event["id"], "kind": "picture", "voice": None})
            continue
        source = bound_path(ep, event["audio"], f"voice source {event['id']}")
        verify_sha(source, event.get("audioSha256", ""), f"voice source {event['id']}")
        admission = event.get("admission")
        if event.get("kind") == "dialogue":
            if not isinstance(admission, dict):
                raise RuntimeError(f"dialogue {event['id']} lacks admission provenance")
            for path_key, sha_key in (
                ("sidecar", "sidecarSha256"),
                ("reviewExport", "reviewExportSha256"),
            ):
                evidence_path = bound_path(
                    ep, admission.get(path_key, ""), f"dialogue {event['id']} {path_key}"
                )
                verify_sha(
                    evidence_path, admission.get(sha_key, ""),
                    f"dialogue {event['id']} {path_key}",
                )
        source_start = float(event.get("audioSourceStart", 0.0))
        source_duration = event.get("audioSourceDuration")
        filters = None
        if source_start > 0 or source_duration is not None and event.get("kind") == "dialogue":
            filters = f"atrim=start={source_start:.9f}"
            if source_duration is not None:
                filters += f":duration={float(source_duration):.9f}"
            filters += ",asetpts=PTS-STARTPTS"
        samples = audio(source, filters)
        room = round(float(event["duration"]) * SR)
        # A voice event is never silently clipped to make arithmetic pass.
        # One AAC frame of container tail is tolerated only outside an explicitly
        # measured speech window; anything larger is a broken timing contract.
        if len(samples) > room + 1024:
            raise RuntimeError(
                f"Voice exceeds event {event['id']}: {len(samples) / SR:.6f}s > "
                f"{event['duration']:.6f}s"
            )
        if len(samples) > room:
            samples = samples[:room]
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples * samples)))
        # A shared speech target retains native dynamics while removing clip jumps.
        gain = min(10 ** ((-20 - 20 * math.log10(max(rms, 1e-7))) / 20), 0.92 / max(peak, 1e-7), 3.0)
        samples *= gain
        place(stem, samples, event["start"])
        level = {"event": event["id"], "rms": rms, "peak": peak, "gain": gain}
        level.update(place_audio_overlaps(stem, source, event, source_start, gain))
        levels.append(level)
    output = work / "voice-hi-48k24.wav"
    write_audio(output, stem)
    (work / "voice-levels.json").write_text(json.dumps(levels, indent=2) + "\n")
    return output


def wave_envelope(plan: dict, wave: dict, length: int) -> tuple[np.ndarray, dict]:
    env = np.zeros(length, np.float32)
    by_shot = {row["shot"]: row for row in wave["shots"]}
    source_energy = [float(row["music_energy"]) for row in wave["shots"]]
    lo, hi = min(source_energy), max(source_energy)
    breaths = []
    rendered_points = []
    for event in plan["events"]:
        row = by_shot[event["shot"]]
        start = round(event["start"] * SR)
        end = min(length, round(event["end"] * SR))
        # Expand the authored wave rather than compressing the entire episode
        # into a five-decibel continuous bed.  The floor here is audible but
        # restrained; explicit editorial breaths below can reach near-silence.
        normalized = (float(row["music_energy"]) - lo) / max(hi - lo, 1e-6)
        energy = 0.14 + 0.82 * normalized ** 1.55
        if event["kind"] == "dialogue":
            energy *= 0.78
        for breath in plan.get("musicBreaths", []):
            first, last = breath["shots"]
            if first <= event["shot"] <= last:
                energy = min(energy, float(breath["level"]))
                breaths.append({
                    "shot": event["shot"], "event": event["id"],
                    "start": event["start"], "end": event["end"],
                    "level": float(breath["level"]), "reason": breath["reason"],
                })
                break
        env[start:end] = energy
        rendered_points.append({
            "shot": event["shot"], "sourceMusicEnergy": float(row["music_energy"]),
            "envelopeTarget": float(energy), "kind": event["kind"],
        })
    # Smooth shot boundaries over 300 ms in linear time. A full-film convolution
    # with a 14,400-sample kernel needlessly multiplies the render workload.
    n = round(0.3 * SR)
    if n > 1:
        padded = np.pad(env, (n // 2, (n - 1) // 2))
        cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
        env = ((cumulative[n:] - cumulative[:-n]) / n).astype(np.float32)
    # A dramatic vacuum is absolute; smoothing must never leak across its edge.
    for breath in breaths:
        if breath['level'] == 0:
            env[round(breath['start'] * SR):round(breath['end'] * SR)] = 0
    point_source = np.array([x["sourceMusicEnergy"] for x in rendered_points])
    point_rendered = np.array([x["envelopeTarget"] for x in rendered_points])
    unmasked = np.array([x["shot"] not in {b["shot"] for b in breaths} for x in rendered_points])
    correlation_all = float(np.corrcoef(point_source, point_rendered)[0, 1])
    correlation_unmasked = float(np.corrcoef(point_source[unmasked], point_rendered[unmasked])[0, 1])
    audit = {
        "method": "nonlinear dramatic-wave conform with authored editorial breath masks",
        "sourceRange": [lo, hi],
        "renderedRange": [float(env.min()), float(env.max())],
        "scoreActivePercent": round(float(np.mean(env > 0.20) * 100), 3),
        "lowLevelPercent": round(float(np.mean(env <= 0.10) * 100), 3),
        "nearSilentPercent": round(float(np.mean(env <= 0.05) * 100), 3),
        "correlationWithMusicEnergyAllEvents": round(correlation_all, 5),
        "correlationWithMusicEnergyOutsideAuthoredBreaths": round(correlation_unmasked, 5),
        "authoredBreaths": breaths,
        "eventTargets": rendered_points,
    }
    return env, audit


def phrase_entries(source: np.ndarray) -> list[int]:
    """Find quiet phrase boundaries usable as alternate reprise entries."""
    mono = np.mean(source, axis=1)
    window = max(1, round(1.5 * SR))
    hop = max(1, round(0.5 * SR))
    candidates = []
    for pos in range(round(12 * SR), max(round(12 * SR) + 1, len(source) - round(12 * SR)), hop):
        chunk = mono[pos:pos + window]
        if len(chunk) == window:
            candidates.append((float(np.sqrt(np.mean(chunk * chunk))), pos))
    entries = [0]
    for _, pos in sorted(candidates):
        if all(abs(pos - existing) >= round(12 * SR) for existing in entries):
            entries.append(pos)
        if len(entries) == 4:
            break
    return entries


def extend_score(source: np.ndarray, duration: float, crossfade_seconds: float = 5.0) -> tuple[np.ndarray, list[dict]]:
    target = round(duration * SR)
    if len(source) >= target:
        return source[:target].copy(), []
    overlap = min(round(crossfade_seconds * SR), len(source) // 5)
    out = source.copy()
    entries = phrase_entries(source)
    joins = []
    reprise = 0
    while len(out) < target:
        reprise += 1
        offset = entries[reprise % len(entries)]
        piece = source[offset:].copy()
        if overlap and len(out) > overlap:
            n = min(overlap, len(piece), len(out))
            join_at = len(out) - n
            curve = np.sin(np.linspace(0, math.pi / 2, n, dtype=np.float32))[:, None]
            out[-n:] = out[-n:] * curve[::-1] + piece[:n] * curve
            out = np.concatenate([out, piece[n:]], axis=0)
            joins.append({
                "timelineSeconds": round(join_at / SR, 3),
                "sourceEntrySeconds": round(offset / SR, 3),
                "crossfadeSeconds": round(n / SR, 3),
                "selection": "low-RMS quiet-energy splice candidate; musical phrase not verified",
            })
        else:
            out = np.concatenate([out, piece], axis=0)
    return out[:target], joins


def conform_score_section(source: np.ndarray, movement: dict) -> tuple[np.ndarray, list[dict], dict]:
    """An authored recording cut must never silently become an automatic reprise."""
    duration = float(movement["timelineDuration"])
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Score placement needs a positive finite duration")
    explicit = "sourceStart" in movement or "sourceDuration" in movement
    if explicit:
        if not all(k in movement for k in ("sourceStart", "sourceDuration")):
            raise RuntimeError("Explicit score cuts need both sourceStart and sourceDuration")
        start, length = float(movement["sourceStart"]), float(movement["sourceDuration"])
        if not all(math.isfinite(v) for v in (start, length)) or start < 0 or length <= 0:
            raise RuntimeError("Invalid explicit score recording window")
        first, count = round(start * SR), round(length * SR)
        if count != round(duration * SR) or count < 1:
            raise RuntimeError("Explicit score cut must cover its placement exactly without retiming")
        if first + count > len(source):
            raise RuntimeError("Explicit score cut exceeds actual decoded recording; no silent padding/reprise")
        conformed, joins = source[first:first + count].copy(), []
        selection = dict(method="authored-recording-section-no-reprise-no-retiming",
                         sourceStartSample=first, sourceSamples=count,
                         sourceStartSeconds=first/SR, sourceEndSeconds=(first+count)/SR)
    else:
        conformed, joins = extend_score(source, duration)
        selection = dict(method="selected-take-with-quiet-energy-reprise-candidates")
    gain_db = float(movement.get("gainDb", 0))
    if not math.isfinite(gain_db) or not -24 <= gain_db <= 6:
        raise RuntimeError("Score section gain outside -24..6dB")
    conformed *= 10 ** (gain_db / 20)
    fades = [float(movement.get(k, min(2.5, duration / 2)))
             for k in ("fadeInSeconds", "fadeOutSeconds")]
    if not all(math.isfinite(v) and 0 <= v <= duration for v in fades) or sum(fades) > duration + 1/SR:
        raise RuntimeError("Score section fades overlap or exceed the section")
    for seconds, tail in zip(fades, (False, True)):
        n = round(seconds * SR)
        if n:
            curve = np.sin(np.linspace(0, math.pi / 2, n, dtype=np.float32))[:, None]
            if tail:
                conformed[-n:] *= curve[::-1]
            else:
                conformed[:n] *= curve
    selection.update(gainDb=gain_db, fadeInSeconds=fades[0], fadeOutSeconds=fades[1])
    return conformed, joins, selection


def validate_score_bus(stem: np.ndarray) -> float:
    if not np.all(np.isfinite(stem)):
        raise RuntimeError("Score bus contains non-finite samples")
    peak = float(np.max(np.abs(stem)))
    if peak > 0.98:
        raise RuntimeError(f"Score bus peak {peak:.6f} exceeds PCM headroom; revise gains/overlaps")
    return peak


def build_music(ep: Path, work: Path, plan: dict, wave: dict) -> Path:
    total = round(plan["runtimeSeconds"] * SR)
    stem = np.zeros((total, 2), np.float32)
    decisions = []
    for movement in plan["scorePlacements"]:
        source_path = bound_path(ep, movement["track"], f"score {movement['movement']}")
        verify_sha(
            source_path, movement.get("trackSha256", ""),
            f"score {movement['movement']} track",
        )
        sidecar_path = bound_path(
            ep, movement.get("sidecar", ""), f"score {movement['movement']} sidecar"
        )
        verify_sha(
            sidecar_path, movement.get("sidecarSha256", ""),
            f"score {movement['movement']} sidecar",
        )
        review_path = bound_path(
            ep, movement.get("reviewExport", ""),
            f"score {movement['movement']} review export",
        )
        verify_sha(
            review_path, movement.get("reviewExportSha256", ""),
            f"score {movement['movement']} review export",
        )
        execution_path = bound_path(
            ep, movement.get("executionManifest", ""),
            f"score {movement['movement']} execution manifest",
        )
        verify_sha(
            execution_path, movement.get("executionManifestSha256", ""),
            f"score {movement['movement']} execution manifest",
        )
        normalization = plan.get("musicNormalization", {"method": "legacy-loudnorm"})
        level_audit = {"method": normalization.get("method")}
        if normalization.get("method") == "static-rms":
            source = audio(source_path)
            target = float(normalization.get("targetRmsDbfs", -24))
            maximum_gain = float(normalization.get("maximumGainDb", 9))
            peak_cap = float(normalization.get("peakCapDbfs", -4))
            if not -36 <= target <= -18 or not 0 <= maximum_gain <= 12 or not -12 <= peak_cap <= -3:
                raise RuntimeError("Unsafe static music level contract")
            rms = float(np.sqrt(np.mean(source.astype(np.float64) ** 2)))
            peak = float(np.max(np.abs(source)))
            if rms < 1e-7:
                raise RuntimeError("Selected score source is silent")
            gain_db = min(target - 20 * math.log10(rms), maximum_gain,
                          peak_cap - 20 * math.log10(max(peak, 1e-7)))
            source *= 10 ** (gain_db / 20)
            level_audit.update(sourceRmsDbfs=20 * math.log10(rms), gainDb=gain_db,
                               targetRmsDbfs=target, peakCapDbfs=peak_cap,
                               dynamicCompression=False)
        elif normalization.get("method") == "legacy-loudnorm":
            source = audio(source_path, "loudnorm=I=-24:TP=-3:LRA=10")
        else:
            raise RuntimeError("Unknown explicit music normalization method")
        conformed, joins, section = conform_score_section(source, movement)
        start = float(movement["timelineStart"])
        if not math.isfinite(start) or start < 0 or round(start*SR) + len(conformed) > total:
            raise RuntimeError("Score placement escapes the measured film clock")
        place(stem, conformed, movement["timelineStart"])
        decisions.append({
            "movement": movement["movement"], "take": movement["take"],
            "source": str(source_path), "sourceDuration": len(source) / SR,
            "timelineDuration": movement["timelineDuration"],
            **section,
            "reviewScope": "Authored structural choice; no verified phrase/harmony or listening claim",
            "repriseJoins": joins,
            "normalization": level_audit,
        })
    env, envelope_audit = wave_envelope(plan, wave, total)
    stem *= env[:, None]
    envelope_audit['verifiedPcmPeak'] = validate_score_bus(stem)
    envelope_audit['pcmHeadroomGuard'] = 'fail before write above0.98; never hard-clip a working score'
    output = work / "music-wave-48k24.wav"
    write_audio(output, stem)
    (work / "music-conform.json").write_text(json.dumps(decisions, indent=2) + "\n")
    (work / "music-wave-audit.json").write_text(json.dumps(envelope_audit, indent=2) + "\n")
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("--plan", default="work/assembly-v1/edit-plan.json")
    ap.add_argument("--work", default="work/assembly-v1", help="Versioned component/cache directory")
    ap.add_argument("--wave", default="work/dramatic-wave-v1/dramatic-wave.json")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--stage", choices=("all", "picture", "audio"), default="all")
    args = ap.parse_args()
    ep = args.episode.resolve()
    if platform.system() != 'Linux':
        raise RuntimeError('Production rendering runs on the qualified GPU worker, not the Mac')
    subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], check=True, capture_output=True)
    work = (ep / args.work).resolve()
    work.relative_to(ep / 'work')
    work.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ep / args.plan).read_text())
    if plan.get('renderable') is not True:
        raise RuntimeError('A complete source-admitted renderable plan is required')
    wave = json.loads((ep / args.wave).read_text()) if args.stage in ('all','audio') else None
    width = round(args.height * 16 / 9 / 2) * 2
    outputs = {}
    if args.stage in ("all", "picture"):
        outputs["picture"] = str(build_picture(ep, work, plan, width, args.height))
    if args.stage in ("all", "audio"):
        outputs["voice"] = str(build_voice(ep, work, plan))
        outputs["music"] = str(build_music(ep, work, plan, wave))
    audit = {
        "episode": plan["episode"], "runtimeSeconds": plan["runtimeSeconds"],
        "resolution": f"{width}x{args.height}", "outputs": outputs,
        **({"frame": plan["frame"]} if plan.get("frame") else {}),
    }
    (work / f"assembly-components-{args.height}p.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
