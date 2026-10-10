"""Copy approved audio onto reconstructed picture without changing A/V offsets."""
from fractions import Fraction
import hashlib
import json
import math
import subprocess


def streams(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)
    ], text=True))["streams"]


def start(stream):
    return int(stream["start_pts"]) * Fraction(stream["time_base"])


def packet_hash(path, selection):
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", selection,
        "-c", "copy", "-f", "hash", "-hash", "sha256", "-"
    ], text=True).strip().split("=", 1)[-1]


def source_audio_identity(path, picture_only=False):
    """Explicit silent HD mode must not silently discard an existing audio track."""
    rows=streams(path)
    videos=[s for s in rows if s['codec_type']=='video']
    audios=[s for s in rows if s['codec_type']=='audio']
    if len(videos)!=1 or len(rows)!=len(videos)+len(audios):
        raise ValueError('Expected one picture and no unrelated streams')
    if picture_only:
        if audios:
            raise ValueError('Picture-only mode requires genuinely silent input/output')
        return None
    if len(audios)!=1:
        raise ValueError('Audio-preserving HD requires exactly one audio stream')
    return packet_hash(path,'0:a:0')


def audio_packet_clock(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets",
        "-show_entries", "packet=pts,dts,duration,size", "-of", "json", str(path)
    ], text=True))["packets"]
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return {"packets": len(data), "sha256": hashlib.sha256(canonical).hexdigest()}


def remux_approved_audio(picture, source, output):
    source_streams = streams(source)
    v = next(s for s in source_streams if s["codec_type"] == "video")
    a = next(s for s in source_streams if s["codec_type"] == "audio")
    pv = next(s for s in streams(picture) if s["codec_type"] == "video")
    video_start, audio_start = start(v), start(a)
    tick = Fraction(v["time_base"])
    movie_scale = math.lcm(tick.denominator, Fraction(a["time_base"]).denominator)
    if movie_scale > 2**31 - 1:
        raise ValueError("Source time bases exceed safe MP4 timescale")
    source_audio_hash = packet_hash(source, "0:a:0")
    # Do not use -shortest: it can discard approved tail AAC packets. Both
    # complete streams belong to the approved source clock and are audited.
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-copyts",
        "-itsoffset", f"{float(video_start - start(pv)):.12f}",
        "-i", str(picture), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
        "-avoid_negative_ts", "disabled", "-movie_timescale", str(movie_scale),
        "-video_track_timescale", str(tick.denominator),
        "-movflags", "+faststart", str(output)
    ], check=True)
    final_streams = streams(output)
    fv = next(s for s in final_streams if s["codec_type"] == "video")
    fa = next(s for s in final_streams if s["codec_type"] == "audio")
    relative_error = abs((start(fv) - start(fa)) - (video_start - audio_start))
    if relative_error > tick:
        raise ValueError(f"Approved relative A/V start changed by {float(relative_error)}s")
    if abs(start(fv) - video_start) > tick or abs(start(fa) - audio_start) > Fraction(a["time_base"]):
        raise ValueError("Approved absolute stream start changed")
    output_audio_hash = packet_hash(output, "0:a:0")
    if source_audio_hash != output_audio_hash:
        raise ValueError("Approved AAC payload changed")
    source_clock, output_clock = audio_packet_clock(source), audio_packet_clock(output)
    if Fraction(a["time_base"]) != Fraction(fa["time_base"]) or source_clock != output_clock:
        raise ValueError("Approved AAC packet clocks or tail changed")
    return {
        "sourceVideoStartSeconds": float(video_start),
        "sourceAudioStartSeconds": float(audio_start),
        "outputVideoStartSeconds": float(start(fv)),
        "outputAudioStartSeconds": float(start(fa)),
        "relativeStartErrorSeconds": float(relative_error),
        "sourceVideoTickSeconds": float(tick),
        "movieTimescale": movie_scale,
        "sourceAudioPacketSha256": source_audio_hash,
        "outputAudioPacketSha256": output_audio_hash,
        "approvedAudioPacketsUnchanged": True,
        "audioPacketClock": source_clock,
        "approvedAudioPacketClocksUnchanged": True,
        "approvedStreamStartsPreserved": True,
    }
