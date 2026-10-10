"""Opt-in H.264/MP4 finishing checks on the qualified GPU worker.

Packet timestamps are not described as decoded-frame timestamps. A metadata
frame count is only a preflight; decode_h264_worker independently counts every
decoded frame and fails on decoder errors. The retained L4 compute driver has
no libnvcuvid, so decoding explicitly uses FFmpeg's H.264 decoder on that worker;
this is not represented as CUDA decoding. No media work is performed on Mac.
No media operations run during import or unit tests.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def container(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]))


def validate_packet_clock(video, rows, expected_frames=None):
    require(video.get("codec_name") == "h264" and video.get("pix_fmt") == "yuv420p",
            "GPU finishing fast path is restricted to qualified 8-bit H.264")
    require(Fraction(video["r_frame_rate"]) == Fraction(24) and
            Fraction(video["avg_frame_rate"]) == Fraction(24), "Expected exact 24fps metadata")
    tb = Fraction(video["time_base"])
    require(tb > 0 and rows, "Missing positive packet clock")
    for row in rows:
        require(all(type(row.get(k)) is int for k in ("pts", "dts", "duration")), "Incomplete integer packet clock")
        require(row["duration"] * tb == Fraction(1, 24), "Video packet duration is not one 24fps frame")
    require(all(b["dts"] > a["dts"] for a, b in zip(rows, rows[1:])), "Non-increasing video DTS")
    pts = sorted(row["pts"] for row in rows)
    require(all((b - a) * tb == Fraction(1, 24) for a, b in zip(pts, pts[1:])),
            "Duplicate/gapped/non-CFR packet presentation clock")
    declared = video.get("nb_frames")
    if declared not in (None, "N/A", ""):
        require(str(declared).isdigit() and int(declared) == len(rows), "MP4 sample count differs from packet count")
    if expected_frames is not None:
        require(type(expected_frames) is int and expected_frames > 0 and len(rows) == expected_frames,
                "Packet count differs from exact expected frame contract")
    times = [float(p * tb) for p in pts]
    timeline = "\n".join(f"{v:.6f}" for v in times).encode("ascii")
    return dict(frames=len(rows), timeBase=str(tb), firstSeconds=round(times[0], 6),
                lastSeconds=round(times[-1], 6), maxCadenceErrorSeconds=0.0,
                timelineSha256=hashlib.sha256(timeline).hexdigest(),
                integerPacketClockSha256=canonical_sha(rows),
                basis="integer packet PTS/DTS/duration; not decoded-frame timestamps",
                frameCountBasis="MP4 nb_frames corroborated by packets" if declared not in (None, "N/A", "") else "packet count; decoded count pending",
                decodedFrameCountVerified=False)


def inspect_h264_mp4(path, expected_frames=None, doc=None):
    path = Path(path)
    doc = container(path) if doc is None else doc
    require(path.suffix.lower() == ".mp4" and "mp4" in doc["format"]["format_name"].split(","),
            "GPU finishing fast path requires an MP4 container")
    videos = [s for s in doc["streams"] if s.get("codec_type") == "video"]
    require(len(videos) == 1, "Expected exactly one H.264 picture stream")
    rows = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts,dts,duration", "-of", "json", str(path)
    ]))["packets"]
    return doc, validate_packet_clock(videos[0], rows, expected_frames)


def probe_media(path):
    """Match mixer probe geometry without its unconditional CPU frame decode."""
    doc = container(path)
    duration = float(doc["format"]["duration"])
    result = dict(durationSeconds=duration, audio=[], video=[])
    for s in doc["streams"]:
        if s.get("codec_type") == "audio":
            result["audio"].append(dict(codec=s["codec_name"], sampleRate=int(s["sample_rate"]),
                channels=int(s["channels"]), durationSeconds=float(s.get("duration", duration))))
        elif s.get("codec_type") == "video":
            _, clock = inspect_h264_mp4(path, doc=doc)
            result["video"].append(dict(codec=s["codec_name"], width=int(s["width"]), height=int(s["height"]),
                frameRate=s["r_frame_rate"], frames=clock["frames"], durationSeconds=float(s.get("duration", duration)),
                frameCountBasis=clock["frameCountBasis"], packetClock=clock))
    return result


def decode_command(path):
    return ["ffmpeg", "-v", "error", "-nostats", "-xerror", "-err_detect", "explode",
            "-c:v", "h264", "-threads", "3",
            "-reinit_filter", "0", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0?",
            "-sn", "-dn", "-vsync", "0", "-c:v", "wrapped_avframe", "-progress", "pipe:1",
            "-f", "null", "-"]


def parse_decode_progress(stdout, expected_frames):
    block, final = {}, None
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        block[key.strip()] = value.strip()
        if key.strip() == "progress":
            if value.strip() == "end":
                final = block.copy()
            block = {}
    require(final is not None, "GPU decode did not report progress=end")
    require(final.get("frame", "").isdigit() and int(final["frame"]) == expected_frames,
            "Actual GPU decoded frame count differs from expected")
    require(final.get("dup_frames") == "0" and final.get("drop_frames") == "0", "GPU decode duplicated/dropped frames")
    return final


def decode_h264_worker(path, expected_frames):
    require(platform.system() == "Linux", "GPU finishing decode belongs on the Linux GPU worker")
    require(type(expected_frames) is int and expected_frames > 0, "Positive exact decoded frame contract required")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    require(gpu, "Missing GPU device")
    path = Path(path)
    before = sha256(path)
    tool_sha = sha256(Path(__file__))
    binary = shutil.which("ffmpeg")
    require(binary is not None, "ffmpeg unavailable")
    binary_sha = sha256(binary)
    command = decode_command(path)
    start = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=600, check=True)
    require(not proc.stderr.strip(), "Actual GPU decode emitted errors")
    final = parse_decode_progress(proc.stdout, expected_frames)
    require(sha256(path) == before and sha256(Path(__file__)) == tool_sha and sha256(binary) == binary_sha,
            "Source or decode tool changed during verification")
    return dict(sourceSha256=before, frames=expected_frames, actualDecodedFrameCount=int(final["frame"]),
                decoder="h264", hardwareAcceleration="none; explicit CPU decode on qualified GPU worker",
                hardwareOutputFormat="yuv420p", gpu=gpu,
                audioDecode="native decoder when an audio stream exists", command=command,
                frameTimingBasis="packet presentation timestamps inspected separately; no decoded timestamp claim",
                errorLogEmpty=True, errorLogSha256=hashlib.sha256(proc.stderr.encode()).hexdigest(),
                progressSha256=hashlib.sha256(proc.stdout.encode()).hexdigest(),
                duplicatedFrames=0, droppedFrames=0, toolSha256=tool_sha, ffmpegBinarySha256=binary_sha,
                elapsedSeconds=round(time.monotonic() - start, 3), passed=True)
