"""Restore explicitly reviewed source regions in a neural-upscaled video chunk."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


KINDS = {"source-preserving-sky-repair", "source-preserving-background-repair", "source-preserving-regional-detail-repair"}


def validate_recipe(recipe):
    allowed = {"approved", "approval", "kind", "sourceContainerSha256", "originalChunkSha256", "chunkStartFrame", "chunkFrames", "globalFrameRange", "regions", "reason", "requiredSemantics", "reviewEvidence", "fadeInFrames", "fadeOutFrames"}
    unknown = set(recipe) - allowed
    if unknown:
        raise ValueError(f"Unsupported recipe fields: {unknown}")
    if recipe.get("approved") is not True or recipe.get("kind") not in KINDS:
        raise ValueError("Explicit regional repair approval required")
    start, count = recipe["chunkStartFrame"], recipe["chunkFrames"]
    lo, hi = recipe["globalFrameRange"]
    if not all(type(v) is int for v in (start, count, lo, hi)) or not 0 <= start <= lo < hi <= start + count:
        raise ValueError("Repair range exceeds chunk")
    if not recipe["regions"]:
        raise ValueError("Empty repair regions")
    for region in recipe["regions"]:
        unknown = set(region) - {"outerBox", "outerBoxKeyframes", "featherPixels", "featherFrameEdges", "globalFrameRange", "fadeInFrames", "fadeOutFrames", "reason"}
        if unknown:
            raise ValueError(f"Unsupported region fields: {unknown}")
        if type(region.get('featherFrameEdges', True)) is not bool:
            raise ValueError('featherFrameEdges must be boolean')
        rlo, rhi = region.get("globalFrameRange", [lo, hi])
        if not all(type(v) is int for v in (rlo, rhi)) or not lo <= rlo < rhi <= hi:
            raise ValueError("Region interval exceeds recipe")
        fades = [region.get(k, recipe.get(k, 0)) for k in ("fadeInFrames", "fadeOutFrames")]
        if any(type(v) is not int or v < 0 for v in fades) or sum(fades) > rhi-rlo:
            raise ValueError("Invalid temporal fade")
        if ("outerBox" in region) == ("outerBoxKeyframes" in region):
            raise ValueError("Specify one static box or keyframed boxes")
        points = region.get("outerBoxKeyframes", [])
        if points:
            if any(set(p) != {"frame", "outerBox"} or type(p["frame"]) is not int for p in points):
                raise ValueError("Invalid keyframe fields")
            times = [p["frame"] for p in points]
            if times != sorted(set(times)) or times[0] > rlo or times[-1] < rhi-1:
                raise ValueError("Keyframes must be ordered and cover region interval")
        boxes = [p["outerBox"] for p in points] if points else [region["outerBox"]]
        feather = region["featherPixels"]
        if type(feather) is not int or feather <= 0:
            raise ValueError("Invalid feather")
        for box in boxes:
            if len(box) != 4 or not all(type(x) is int for x in box):
                raise ValueError("Invalid box coordinates")
            x0,y0,x1,y1 = box
            if not (0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080 and min(x1-x0,y1-y0) > 2*feather):
                raise ValueError("Invalid repair geometry")
    return lo, hi


def masks_at(recipe, ordinal):
    for region in recipe["regions"]:
        lo, hi = region.get("globalFrameRange", recipe["globalFrameRange"])
        if not lo <= ordinal < hi:
            continue
        gain = 1.0
        fade_in = region.get("fadeInFrames", recipe.get("fadeInFrames", 0))
        fade_out = region.get("fadeOutFrames", recipe.get("fadeOutFrames", 0))
        if fade_in and ordinal < lo + fade_in:
            gain *= (1-np.cos(np.pi*(ordinal-lo)/fade_in))/2
        if fade_out and ordinal >= hi-fade_out:
            gain *= (1-np.cos(np.pi*(hi-ordinal)/fade_out))/2
        if "outerBoxKeyframes" in region:
            points = region["outerBoxKeyframes"]
            times = [p["frame"] for p in points]
            box = [int(np.rint(np.interp(ordinal, times, [p["outerBox"][k] for p in points]))) for k in range(4)]
        else:
            box = region["outerBox"]
        x0,y0,x1,y1 = box
        yy, xx = np.mgrid[y0:y1, x0:x1]
        frame_edges=region.get('featherFrameEdges',True)
        distances=[d for d,at_edge in ((xx-x0,x0==0),(x1-1-xx,x1==1920),
                                      (yy-y0,y0==0),(y1-1-yy,y1==1080)) if frame_edges or not at_edge]
        distance = np.minimum.reduce(distances).astype(np.float32) if distances else np.full(xx.shape,region['featherPixels'],np.float32)
        ramp = np.clip(distance / region["featherPixels"], 0, 1)
        alpha = (((1-np.cos(np.pi*ramp))/2)*gain)[...,None]
        yield x0,y0,x1,y1,alpha


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("chunk", type=Path)
    ap.add_argument("recipe", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    recipe = json.loads(a.recipe.read_text())
    lo, hi = validate_recipe(recipe)
    if a.output.exists() or a.output.resolve() in {a.source.resolve(), a.chunk.resolve()}:
        raise ValueError("Use a new output path; original and earlier repairs are immutable")
    source_hash, chunk_hash = sha(a.source), sha(a.chunk)
    if source_hash != recipe["sourceContainerSha256"] or chunk_hash != recipe["originalChunkSha256"]:
        raise ValueError("Repair source identity mismatch")
    count, start = recipe["chunkFrames"], recipe["chunkStartFrame"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    hd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(a.chunk), "-an",
                           "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    source = subprocess.Popen([
        "ffmpeg", "-v", "error", "-reinit_filter", "0", "-i", str(a.source), "-an",
        "-vf", f"trim=start_frame={lo}:end_frame={hi},setpts=N/(24*TB),scale=1920:1080:flags=lanczos",
        "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"
    ], stdout=subprocess.PIPE)
    partial = a.output.with_suffix(".partial.mp4")
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
        "-r", "24", "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-threads", "16", "-g", "48", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)
    ], stdin=subprocess.PIPE)
    touched = 0
    processed = 0
    try:
        for i in range(count):
            raw = hd.stdout.read(1920*1080*3)
            if len(raw) != 1920*1080*3:
                raise EOFError("Short HD chunk")
            if lo <= start+i < hi:
                clean = source.stdout.read(1920*1080*3)
                if len(clean) != len(raw):
                    raise EOFError("Short clean source range")
                frame = np.frombuffer(raw, np.uint8).reshape(1080,1920,3).copy()
                reference = np.frombuffer(clean, np.uint8).reshape(1080,1920,3)
                masks = list(masks_at(recipe, start+i))
                for x0,y0,x1,y1,alpha in masks:
                    frame[y0:y1,x0:x1] = np.rint(frame[y0:y1,x0:x1]*(1-alpha) + reference[y0:y1,x0:x1]*alpha).astype(np.uint8)
                raw = frame.tobytes()
                touched += bool(masks)
                processed += 1
            encoder.stdin.write(raw)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Repair encoder failed")
        for process in (hd, source):
            if process.stdout.read(1):
                raise RuntimeError("Unexpected trailing source frame")
            process.stdout.close()
            if process.wait():
                raise RuntimeError("Repair decoder failed")
    finally:
        for process in (hd,source,encoder):
            if process.poll() is None:
                process.kill()
                process.wait()
    info = json.loads(subprocess.check_output([
        "ffprobe","-v","error","-threads","8","-count_frames","-select_streams","v:0","-show_streams","-of","json",str(partial)
    ], text=True))["streams"][0]
    if (int(info["width"]),int(info["height"]),int(info["nb_read_frames"]),info["r_frame_rate"]) != (1920,1080,count,"24/1"):
        raise RuntimeError("Repaired chunk geometry/frame contract failed")
    decoded = subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(partial),"-f","null","-"], capture_output=True,text=True,check=True)
    expected_touched = sum(any(r.get("globalFrameRange", [lo,hi])[0] <= f < r.get("globalFrameRange", [lo,hi])[1] for r in recipe["regions"]) for f in range(lo,hi))
    if decoded.stderr.strip() or processed != hi-lo or touched != expected_touched or sha(a.source) != source_hash or sha(a.chunk) != chunk_hash:
        raise RuntimeError("Repair verification failed")
    partial.replace(a.output)
    audit = {**recipe,"source":str(a.source),"originalChunk":str(a.chunk),
             "output":str(a.output),"outputSha256":sha(a.output),"recipeSha256":sha(a.recipe),
             "outputFrames":count,"repairedFrames":touched,"decode":"passed",
             "processedRangeFrames":processed,
             "repairedFramesMeaning":"Union of active region intervals, including fade boundaries; gaps in the decoding envelope are excluded.",
             "rendererVersion":"source-regions-keyframes-v2",
             "scope":"Only approved feathered regions restored from same source ordinals; one chunk reencoded, other chunks remain untouched. This is an explicit regional exception, not a whole-film resize fallback."}
    a.output.with_suffix(".json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()
