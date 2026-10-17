#!/usr/bin/env python3
"""Verify packet clocks and decode without implicit filter reinitialization."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("master", type=Path)
ap.add_argument("output", type=Path)
ap.add_argument("--gpu-finishing-probe", action="store_true", help="Strict H.264 packet clock plus complete decode/count on the GPU worker; no decoded-frame timestamp claim")
ap.add_argument("--episode", type=Path, help="Project root; also records the master's project-relative path so the audit survives a move")
a = ap.parse_args()
data = json.loads(subprocess.check_output([
    "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
    "-show_entries", "packet=pts_time,dts_time", "-of", "json", str(a.master)], text=True))["packets"]
pts = sorted(float(x["pts_time"]) for x in data)
dts = [float(x["dts_time"]) for x in data]
error = max(abs(t - pts[0] - i / 24) for i, t in enumerate(pts))
assert all(dts[i] > dts[i-1] for i in range(1, len(dts))), "Non-increasing decode timestamp"
assert error < .002, f"Presentation clock deviates from24fps by{error}"
gpu_decode = None
if a.gpu_finishing_probe:
    import gpu_finishing_probe as gpu_probe
    _, strict_clock = gpu_probe.inspect_h264_mp4(a.master, len(pts))
    gpu_decode = gpu_probe.decode_h264_worker(a.master, len(pts))
else:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-reinit_filter", "0", "-i", str(a.master), "-vsync", "0", "-f", "null", "-"], capture_output=True, text=True, check=True)
    if proc.stderr.strip():
        raise RuntimeError(f"Clean passthrough decode failed: {proc.stderr}")
h = hashlib.sha256()
with a.master.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
result = {"master": str(a.master.resolve()), "sha256": h.hexdigest(),
          "frames": len(pts), "fps": 24, "firstVideoPtsSeconds": pts[0],
          "maxPresentationClockErrorSeconds": error, "decodeTimestampsStrictlyIncreasing": True,
          "decode": "passed with empty error log, fixed filter graph, passthrough frames and source encoder time base",
          "passed": True}
if a.episode is not None:
    result["masterEpisodeRelative"] = a.master.resolve().relative_to(a.episode.resolve()).as_posix()
if gpu_decode is not None:
    result.update(decode="GCP GPU-worker H.264 complete decode passed; exact decoded count, no dropped/duplicated frames",
                  frameTimingBasis="packet presentation timestamps; not decoded-frame timestamps",
                  strictPacketClock=strict_clock, gpuDecode=gpu_decode)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
