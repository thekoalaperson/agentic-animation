#!/usr/bin/env python3
"""GPU-worker-only, source-bound audio revision; never render picture or publish.

The config is a contract, not an approval. --premaster-sha256 may bind its one
not-yet-rendered input; the resolved contract is retained before processing.
--check-config and the unit tests perform no media operations.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys

SR = 48000
FPS = 24
SPF = SR // FPS
SHA = re.compile(r"[0-9a-f]{64}\Z")
INPUTS = {"editPlan", "old720", "old1080", "shortPicture", "shortSpec", "voice", "bodyPremaster"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def inside(ep, value, *, work=False):
    raw = Path(value)
    require(not raw.is_absolute(), "Config paths must be episode-relative")
    path = (ep / raw).resolve()
    try:
        relative = path.relative_to(ep.resolve())
        if work:
            path.relative_to((ep / "work").resolve())
    except ValueError as exc:
        raise ValueError("Path escapes its authorized episode/work root") from exc
    require("deliverables" not in relative.parts, "Canonical deliverables are never adapter inputs/outputs")
    require(path != ep.resolve() and (not work or path != (ep / "work").resolve()), "Broad root target")
    return path


def resolved_config(config, premaster_sha=None):
    c = copy.deepcopy(config)
    require(c.get("schemaVersion") == 1 and c.get("kind") == "source-bound-audio-only-delivery-revision", "Wrong adapter contract")
    require(c.get("fps") == FPS and c.get("language") == "hi", "Only the qualified Hindi 24fps route is supported")
    require(c.get("backgroundDerivation") == "unity-voice-linear-premaster-minus-unchanged-voice", "Unity-voice premaster contract required")
    require(set(c.get("inputs", {})) == INPUTS, "Missing/unrecognized input binding")
    for key, row in c["inputs"].items():
        require(set(row) == {"path", "sha256"} and isinstance(row["path"], str), "Malformed input binding")
        if key == "bodyPremaster" and premaster_sha:
            require(row["sha256"] in (None, premaster_sha), "CLI hash conflicts with already-bound premaster")
            row["sha256"] = premaster_sha
        require(isinstance(row["sha256"], str) and SHA.fullmatch(row["sha256"]), f"{key} requires its exact SHA-256")
    require(isinstance(c.get("outputRoot"), str), "Missing versioned outputRoot")
    require(type(c.get("gpuFinishingProbe", False)) is bool, "gpuFinishingProbe must be an explicit boolean")
    peak = c.get("targetTruePeakDbtp", -1.5)
    require(type(peak) in (int, float) and math.isfinite(peak) and -3 <= peak <= -1.5,
            "Final true-peak target must remain between -3 and -1.5 dBTP")
    return c


def integer(value, label):
    require(type(value) is int and value > 0, label + " must be a positive integer")
    return value


def frame(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), label + " is not finite")
    n = round(value * FPS)
    require(abs(n / FPS - value) < 0.000002, label + " is not on the 24fps clock")
    return n


def validate_plans(plan, spec):
    require(plan.get("language") == "hi" and plan.get("deliveryFps") == FPS, "Unexpected long-film clock")
    events = plan.get("events", [])
    require(events, "Missing voice events")
    lookup, order, cursor = {}, [], 0
    for e in events:
        require(e["id"] not in lookup, "Duplicate voice event")
        lo, hi = frame(e["start"], "event start"), frame(e["end"], "event end")
        require(lo == cursor and hi > lo, "Long-film event gap/overlap")
        require(frame(e["duration"], "event duration") == hi - lo, "Event duration mismatch")
        require(0 <= e["subtitleAudioAt"] < e["duration"] and 0 < e["subtitleAudioDuration"] <= e["duration"] - e["subtitleAudioAt"] + 0.001, "Invalid whole-speech window")
        lookup[e["id"]] = e
        order.append(e["id"])
        cursor = hi
    require(frame(plan["runtimeSeconds"], "runtime") == cursor, "Long-film runtime mismatch")
    require(spec.get("kind") == "authored-motion-short" and spec.get("language") == "hi" and spec.get("fps") == FPS, "Wrong Short spec")
    cta = integer(spec["ctaFrames"], "ctaFrames")
    short_body = integer(spec["bodyFrames"], "Short bodyFrames")
    require(integer(spec["totalFrames"], "Short totalFrames") == short_body + cta, "Short ending clock mismatch")
    require(cta == 156, "This qualified route retains the existing 6.5s ending")
    require(spec.get("beats") and spec["beats"][0]["id"] == order[0], "Preserve the complete original opening")
    short_cursor, indexes = 0, []
    for beat in spec["beats"]:
        require(beat["id"] in lookup, "Unknown Short scene")
        e = lookup[beat["id"]]
        indexes.append(order.index(beat["id"]))
        n = integer(beat["frames"], "beat frames")
        require(type(beat["sourceStartFrame"]) is int and type(beat["startFrame"]) is int, "Short ordinals must be integers")
        require(beat["sourceStartFrame"] == frame(e["start"], "source scene start"), "Changed Short source interval")
        require(n == frame(e["duration"], "whole scene duration"), "Short must retain the whole scene including its tail")
        require(beat["startFrame"] == short_cursor, "Short gap/overlap")
        require(beat["spokenTextHi"] == e["subtitleTextHi"] and abs(beat["audioDuration"] - e["subtitleAudioDuration"]) < 0.000001, "Changed Short speech")
        require(e["subtitleAudioAt"] == 0, "Qualified Short route expects speech at scene origin")
        short_cursor += n
    require(indexes == sorted(set(indexes)), "Short scene order/repetition changed")
    require(short_cursor == short_body, "Short does not exhaust its body")
    return {"bodyFrames": cursor, "fullFrames": cursor + cta, "shortFrames": short_body + cta, "ctaFrames": cta}


def short_slices(spec):
    beats = spec["beats"]
    rows = []
    for i, b in enumerate(beats):
        previous = beats[i - 1] if i else None
        following = beats[i + 1] if i + 1 < len(beats) else None
        fade_in = previous is None or previous["sourceStartFrame"] + previous["frames"] != b["sourceStartFrame"]
        fade_out = following is None or b["sourceStartFrame"] + b["frames"] != following["sourceStartFrame"]
        rows.append(dict(id=b["id"], sourceStartSample=b["sourceStartFrame"] * SPF,
                         outputStartSample=b["startFrame"] * SPF, samples=b["frames"] * SPF,
                         fadeInSamples=round(.18 * SR) if fade_in else 0,
                         fadeOutSamples=round((.6 if following is None else .18) * SR) if fade_out else 0))
    return rows


def short_audio(voice, premix, spec):
    import numpy as np
    require(voice.shape == premix.shape and voice.ndim == 2 and voice.shape[1] == 2, "Voice/premaster sample geometry mismatch")
    require(np.isfinite(voice).all() and np.isfinite(premix).all(), "Nonfinite input audio")
    output = np.zeros((spec["totalFrames"] * SPF, 2), dtype=np.float32)
    dry = output.copy()
    for row in short_slices(spec):
        lo, dest, n = row["sourceStartSample"], row["outputStartSample"], row["samples"]
        v = voice[lo:lo + n]
        require(len(v) == n and dest + n <= len(output), "Short source/output overflow")
        bg = premix[lo:lo + n].astype(np.float64) - v.astype(np.float64)
        for name in ("fadeInSamples", "fadeOutSamples"):
            require(row[name] <= n, "Background fade exceeds its selected scene")
        if row["fadeInSamples"]:
            f = row["fadeInSamples"]
            bg[:f] *= np.linspace(0, 1, f)[:, None]
        if row["fadeOutSamples"]:
            f = row["fadeOutSamples"]
            bg[-f:] *= np.linspace(1, 0, f)[:, None]
        dry[dest:dest + n] = v
        output[dest:dest + n] = v.astype(np.float64) + bg
    return output, dry


def packets(path, stream):
    result = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", stream,
        "-show_packets", "-show_entries", "packet=pts,dts,duration", "-of", "json", str(path)])
    return json.loads(result)["packets"]


def packet_interval(rows, timebase):
    require(rows, "Missing encoded packets")
    tb = Fraction(timebase)
    for row in rows:
        require(all(type(row.get(k)) is int for k in ("pts", "dts", "duration")) and row["duration"] > 0, "Incomplete packet clock")
    return min(Fraction(r["pts"]) * tb for r in rows), max(Fraction(r["pts"] + r["duration"]) * tb for r in rows)


def clock_preserving_mux(picture, aac, output, video_tb, audio_tb, shift):
    scale = math.lcm(Fraction(video_tb).denominator, Fraction(audio_tb).denominator)
    require(scale <= 2**31 - 1, "Unsafe MP4 timescale")
    return ["ffmpeg", "-v", "error", "-copyts", "-i", str(picture),
            "-itsoffset", f"{float(shift):.12f}", "-i", str(aac), "-map", "0:v:0", "-map", "1:a:0",
            "-c", "copy", "-avoid_negative_ts", "disabled", "-movie_timescale", str(scale),
            "-video_track_timescale", str(Fraction(video_tb).denominator), "-movflags", "+faststart", str(output)]


def encode_full_aac(premaster, output, normalizer):
    # Default MP4 movie timescale1000 rounds fractional-ms sample endings.
    # Regression evidence: an encoded AAC tail omitted32 samples; retain complete source timing.
    # A48k movie clock represents every sample without adding/trimming PCM.
    # FFmpeg7 loudnorm's final-block timestamp jumped400 samples on this
    # otherwise contiguous PCM. Derive PTS from emitted48k sample count after
    # explicit resampling; no samples are padded, removed or speed-changed.
    clocked = normalizer + ",aresample=48000,asetpts=N/SR/TB"
    return ["ffmpeg", "-v", "error", "-i", str(premaster), "-vn", "-af", clocked,
            "-ar", str(SR), "-ac", "2", "-c:a", "aac", "-b:a", "256k",
            "-movie_timescale", str(SR), str(output)]


def require_gpu():
    require(platform.system() == "Linux", "All media processing belongs on the qualified GPU worker, never Mac")
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, capture_output=True, text=True)
    require(result.stdout.strip(), "No GPU worker detected")


def execute(ep, config_path, c):
    require_gpu()  # Before any media import/probe/decode or output directory.
    import numpy as np
    import mix_absolute_sfx_longform as mix
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "open_media"))
    import hd_audio_remux as remux
    fast = c.get("gpuFinishingProbe", False)
    if fast:
        import gpu_finishing_probe as gpu_probe
    probe_media = gpu_probe.probe_media if fast else mix.probe_media

    scripts = Path(__file__).resolve().parent
    tool_paths = [Path(__file__).resolve(), scripts / "master_review_audio.py", scripts / "mix_absolute_sfx_longform.py",
                  scripts.parent / "open_media/hd_audio_remux.py", scripts / "audit_master_timestamps.py",
                  scripts / "audit_master_voice_sync.py", scripts / "audit_processed_audio_fidelity.py"]
    if fast:
        tool_paths.append(scripts / "gpu_finishing_probe.py")
    tool_hashes = {str(p): digest(p) for p in tool_paths}
    original_config_sha = digest(config_path)
    paths = {k: inside(ep, v["path"]) for k, v in c["inputs"].items()}
    out = inside(ep, c["outputRoot"], work=True)
    require(not out.exists(), "Preserve earlier attempts; choose an absent versioned outputRoot")
    require(all(out not in p.parents and out != p for p in paths.values()), "Output root contains an input")
    for k, path in paths.items():
        require(path.is_file() and digest(path) == c["inputs"][k]["sha256"], f"Changed/missing input: {k}")
    plan, spec = read(paths["editPlan"]), read(paths["shortSpec"])
    clock = validate_plans(plan, spec)
    probes = {k: probe_media(paths[k]) for k in ("old720", "old1080", "shortPicture", "voice", "bodyPremaster")}
    for k, dimensions, frames, audio_count in (("old720", (1280, 720), clock["fullFrames"], 1),
                                               ("old1080", (1920, 1080), clock["fullFrames"], 1),
                                               ("shortPicture", (1080, 1920), clock["shortFrames"], 0)):
        p = probes[k]
        require(len(p["video"]) == 1 and len(p["audio"]) == audio_count, f"Unexpected streams in {k}")
        v = p["video"][0]
        require((v["width"], v["height"], v["frames"], Fraction(v["frameRate"])) == (*dimensions, frames, Fraction(FPS)), f"Changed picture contract: {k}")
    for key in ("voice", "bodyPremaster"):
        mix.verify_audio_probe(probes[key], clock["bodyFrames"] / FPS, clock["bodyFrames"] * SPF, key,
                               codec="pcm_f32le" if key == "bodyPremaster" else None)
    old_streams = remux.streams(paths["old720"])
    ov = next(s for s in old_streams if s["codec_type"] == "video")
    oa = next(s for s in old_streams if s["codec_type"] == "audio")
    require(int(oa["sample_rate"]) == SR and int(oa["channels"]) == 2 and oa["codec_name"] == "aac", "Old audio is not qualified 48k stereo AAC")
    old_audio_interval = packet_interval(packets(paths["old720"], "a:0"), oa["time_base"])
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "resolved-config.json", c)

    def run(cmd):
        subprocess.run([str(x) for x in cmd], check=True)

    def preserve_picture(source, target):
        sv = next(s for s in remux.streams(source) if s["codec_type"] == "video")
        tv = next(s for s in remux.streams(target) if s["codec_type"] == "video")
        require(Fraction(sv["time_base"]) == Fraction(tv["time_base"]), "Picture timebase changed")
        before, after = packets(source, "v:0"), packets(target, "v:0")
        require(before == after, "Native picture PTS/DTS/duration changed")
        source_hash = remux.packet_hash(source, "0:v:0")
        require(source_hash == remux.packet_hash(target, "0:v:0"), "Picture payload changed")
        return dict(sourceSha256=digest(source), picturePayloadSha256=source_hash,
                    picturePacketClockSha256=hashlib.sha256(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    packets=len(before), timeBase=sv["time_base"], exactPicturePacketsAndClockPreserved=True)

    full_wav = out / "full-premaster-48kf32.wav"
    run(["ffmpeg", "-v", "error", "-i", paths["bodyPremaster"], "-af", f"apad=pad_len={clock['ctaFrames'] * SPF}",
         "-ar", SR, "-ac", 2, "-c:a", "pcm_f32le", full_wav])
    mix.verify_audio_probe(probe_media(full_wav), clock["fullFrames"] / FPS, clock["fullFrames"] * SPF, "full premaster", codec="pcm_f32le")
    peak_target = c.get("targetTruePeakDbtp", -1.5)
    settings = dict(targetIntegratedLufs=-14, targetTruePeakDbtp=peak_target, targetLra=11)
    measured = mix.loudness(full_wav, settings)
    norm = (f"loudnorm=I=-14:TP={peak_target}:LRA=11:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true")
    aac = out / "full-new-audio.m4a"
    run(encode_full_aac(full_wav, aac, norm))
    na = next(s for s in remux.streams(aac) if s["codec_type"] == "audio")
    new_interval = packet_interval(packets(aac, "a:0"), na["time_base"])
    shift = old_audio_interval[0] - new_interval[0]
    require(abs(shift) <= Fraction(1, 10), "Unexpected encoder priming/start offset")
    # Complete PCM length is frame-derived, not the old AAC/container duration.
    # A split old body/end-card may contain different AAC packet boundaries.
    require(abs(new_interval[1] + shift - old_audio_interval[1]) <= Fraction(1, SR), "New AAC does not cover the original complete audio clock")
    prefix = ep.name.split("-", 1)[0].upper()
    full720 = out / f"{prefix}_HI_720P_AUDIO_REVISION.mp4"
    run(clock_preserving_mux(paths["old720"], aac, full720, ov["time_base"], na["time_base"], shift))
    full720_picture = preserve_picture(paths["old720"], full720)
    delivered_a = next(s for s in remux.streams(full720) if s["codec_type"] == "audio")
    delivered_interval = packet_interval(packets(full720, "a:0"), delivered_a["time_base"])
    require(all(abs(a - b) <= Fraction(1, SR) for a, b in zip(delivered_interval, old_audio_interval)), "Original AAC start/end not preserved")
    require(abs(remux.start(delivered_a) - remux.start(oa)) <= Fraction(1, SR), "Audio stream start changed")
    require(remux.packet_hash(aac, "0:a:0") == remux.packet_hash(full720, "0:a:0"), "AAC re-encoded during clock rebasing")
    delivered = mix.loudness(full720, settings)
    require(abs(float(delivered["input_i"]) + 14) <= .5 and float(delivered["input_tp"]) <= -1, "Full-film delivered loudness needs investigation")
    full1080 = out / f"{prefix}_HI_1080P_AUDIO_REVISION.mp4"
    hd_audio = remux.remux_approved_audio(paths["old1080"], full720, full1080)
    hd_picture = preserve_picture(paths["old1080"], full1080)
    write_new(out / "full-master-receipt.json", dict(kind="picture-preserving-full-audio-revision",
        configSha256=digest(out / "resolved-config.json"), inputs=c["inputs"], frames=clock["fullFrames"], fps=FPS,
        bodyFrames=clock["bodyFrames"], silentTailSamples=clock["ctaFrames"] * SPF,
        premasterSha256=digest(full_wav), encodedAacSha256=digest(aac), normalizer=norm,
        measured=measured, delivered=delivered, measuredPacketRebaseSeconds=float(shift),
        oldAudioPacketIntervalSeconds=list(map(float, old_audio_interval)),
        full720=dict(path=str(full720.relative_to(ep)), sha256=digest(full720), picture=full720_picture),
        full1080=dict(path=str(full1080.relative_to(ep)), sha256=digest(full1080), picture=hd_picture, audioCopy=hd_audio),
        finalApproved=False, publicationAuthorized=False))

    voice, premix = mix.decode_samples(paths["voice"]), mix.decode_samples(paths["bodyPremaster"])
    require(len(voice) == clock["bodyFrames"] * SPF and len(premix) == len(voice), "Decoded body samples differ")
    short_mix, short_voice = short_audio(voice, premix, spec)
    short_pcm, short_dry = out / "short-premaster-48kf32.wav", out / "short-voice-48kf32.wav"
    for path, samples in ((short_pcm, short_mix), (short_dry, short_voice)):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0",
                        "-c:a", "pcm_f32le", str(path)], input=samples.astype("<f4").tobytes(), check=True)
    lookup = {e["id"]: e for e in plan["events"]}
    voice_events = []
    for beat in spec["beats"]:
        e = copy.deepcopy(lookup[beat["id"]])
        e.update(start=beat["startFrame"] / FPS, duration=beat["frames"] / FPS,
                 end=(beat["startFrame"] + beat["frames"]) / FPS)
        voice_events.append(e)
    voice_plan = out / "short-voice-audit-plan.json"
    write_new(voice_plan, dict(kind="whole-performance-short-voice-clock", sourceSpecSha256=c["inputs"]["shortSpec"]["sha256"],
                              sourceEditPlanSha256=c["inputs"]["editPlan"]["sha256"], runtimeSeconds=spec["totalFrames"] / FPS,
                              events=voice_events))
    write_new(out / "short-audio-receipt.json", dict(kind="unchanged-picture-whole-voice-audio-revision", inputs=c["inputs"],
        slices=short_slices(spec), premasterSha256=digest(short_pcm), voiceSha256=digest(short_dry),
        backgroundDerivation=c["backgroundDerivation"], sourceVoiceFaded=False,
        ctaSamples=spec["ctaFrames"] * SPF, ctaAudio="digital silence", finalApproved=False))
    del voice, premix, short_mix, short_voice
    products = [("720", full720, paths["voice"], paths["editPlan"]), ("1080", full1080, paths["voice"], paths["editPlan"])]
    for label, lufs in (("yt", -14), ("reels", -10)):
        target = out / f"{prefix}_SHORT_HI_{label.upper()}.mp4"
        config = dict(kind="source-bound-review-master", inputs={
            "picture": c["inputs"]["shortPicture"],
            "premaster": dict(path=str(short_pcm.relative_to(ep)), sha256=digest(short_pcm))},
            expectedVideoFrames=spec["totalFrames"], output=str(target.relative_to(ep)),
            targetIntegratedLufs=lufs, targetTruePeakDbtp=-1.5)
        config['sampleClock48k'] = True
        if fast:
            config["gpuFinishingProbe"] = True
        if label == "reels":
            config["socialCompression"] = True
        path = out / f"master-{label}.json"
        write_new(path, config)
        run([sys.executable, scripts / "master_review_audio.py", ep, path.relative_to(ep)])
        preserve_picture(paths["shortPicture"], target)
        products.append((label, target, short_dry, voice_plan))
    screening = []
    for label, target, dry, audit_plan in products:
        run([sys.executable, scripts / "audit_master_timestamps.py", target, out / f"clock-{label}.json"]
            + (["--gpu-finishing-probe"] if fast else []))
        result = subprocess.run([sys.executable, str(scripts / "audit_master_voice_sync.py"), str(target), str(dry),
            str(audit_plan), str(out / f"voice-sync-{label}.json"), "--speech-band", "--measured-speech-window"])
        receipt = read(out / f"voice-sync-{label}.json")  # Missing receipt is a real tool error.
        require(result.returncode == 0 or receipt.get("passed") is False, "Unexplained speech verifier failure")
        screening.append(dict(product=label, masterSha256=digest(target), voiceScreenPassed=receipt["passed"],
                              timingPassed=receipt["timingPassed"], voicePresencePassed=receipt["voicePresencePassed"]))
    reels_receipt = read(products[-1][1].with_suffix(".json"))
    if reels_receipt.get("encodedCorrections"):
        run([sys.executable, scripts / "audit_processed_audio_fidelity.py", ep,
             products[-1][1].with_suffix(".json").relative_to(ep), voice_plan.relative_to(ep), (out / "reels-processed-fidelity").relative_to(ep)])
    for k, path in paths.items():
        require(digest(path) == c["inputs"][k]["sha256"], f"Input changed during execution: {k}")
    require(digest(config_path) == original_config_sha, "Config changed during execution")
    require(all(digest(Path(p)) == h for p, h in tool_hashes.items()), "Tool changed during execution")
    result = dict(kind="source-bound-audio-only-delivery-revision", computeHost=socket.gethostname(),
        sourceConfigSha256=original_config_sha, resolvedConfigSha256=digest(out / "resolved-config.json"), toolHashes=tool_hashes,
        products=[dict(id=l, path=str(p.relative_to(ep)), sha256=digest(p)) for l, p, _, _ in products],
        speechScreening=screening, status="rendered-audited-not-published", allVoiceScreensPassed=all(x["voiceScreenPassed"] for x in screening),
        requiresSpeechScreenInvestigation=not all(x["voiceScreenPassed"] for x in screening),
        finalApproved=False, publicationAuthorized=False,
        note="Failed dry-voice screens remain failures; processed-fidelity evidence is additional, not a substituted voice/taste approval.")
    write_new(out / "audio-revision-receipt.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--premaster-sha256", help="Exact digest from the completed mixer receipt; never discovers or guesses the digest")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check-config", action="store_true")
    actions.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    ep = args.episode.resolve()
    config_path = inside(ep, args.config, work=True)
    c = resolved_config(read(config_path), args.premaster_sha256)
    for binding in c["inputs"].values():
        inside(ep, binding["path"])
    inside(ep, c["outputRoot"], work=True)
    if args.check_config:
        for key in ("editPlan", "shortSpec"):
            binding = c["inputs"][key]
            require(digest(inside(ep, binding["path"])) == binding["sha256"], f"Changed metadata input: {key}")
        clock = validate_plans(read(inside(ep, c["inputs"]["editPlan"]["path"])), read(inside(ep, c["inputs"]["shortSpec"]["path"])))
        print(json.dumps(dict(status="metadata-contract-valid", clock=clock, metadataHashesVerified=True,
                              mediaInspected=False, mediaHashesVerified=False, publicationAuthorized=False)))
    else:
        execute(ep, config_path, c)


if __name__ == "__main__":
    main()
