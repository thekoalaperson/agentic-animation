#!/usr/bin/env python3
"""Measure final-master voice lag against every exact-timeline source event."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

SR = 8000


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, speech_band=False):
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0"]
    if speech_band:
        command += ["-af", "highpass=f=300,lowpass=f=3500"]
    raw = subprocess.check_output(command + ["-ar", str(SR), "-ac", "1", "-f", "f32le", "-"])
    return np.frombuffer(raw, np.float32)


def compare(reference, candidate):
    # Valid cross-correlation via FFT; each lag has its own energy denominator.
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    reference -= reference.mean()
    candidate -= candidate.mean()
    n = len(reference)
    size = 1 << (len(candidate) + n - 2).bit_length()
    correlation = np.fft.irfft(np.fft.rfft(candidate, size) * np.fft.rfft(reference[::-1], size), size)
    dots = correlation[n - 1:len(candidate)]
    energy = np.r_[0., np.cumsum(candidate * candidate)]
    energies = energy[n:] - energy[:-n]
    scores = dots / np.sqrt(np.maximum(energies * np.sum(reference * reference), 1e-20))
    position = int(np.argmax(scores))
    return position, float(scores[position])


def main():
    global SR
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("voice", type=Path)
    ap.add_argument("plan", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--max-lag", type=float, default=.05)
    ap.add_argument("--min-correlation", type=float, default=.85)
    ap.add_argument("--speech-band", action="store_true", help="Additional 300–3500Hz voice-presence check; retain broadband diagnostic separately")
    ap.add_argument('--measured-speech-window',action='store_true',
                    help='Use the measured full speech-file window, excluding authored post-speech musical breath')
    ap.add_argument('--analysis-sample-rate', type=int, choices=(8000, 48000), default=8000,
                    help='Use native 48kHz to resolve fractional-8kHz-sample AAC offsets without correlation quantization loss')
    ap.add_argument("--episode", type=Path,
                    help="Project root; also records the master's project-relative path so the audit survives a move")
    args = ap.parse_args()
    SR = args.analysis_sample_rate
    plan = json.loads(args.plan.read_text())
    voice, master = load(args.voice, args.speech_band), load(args.master, args.speech_band)
    if len(master) / SR < plan["runtimeSeconds"] - .05:
        raise RuntimeError("Master audio ends before the authored voice timeline")
    rows = []
    radius = round(.15 * SR)
    skipped = []
    for event in plan["events"]:
        if event.get("kind") == "picture":
            skipped.append(event["id"])  # wordless picture has no voice to align
            continue
        lo,hi=event['start'],event['end']
        if args.measured_speech_window:
            lo=event['start']+event['subtitleAudioAt']
            hi=lo+event['subtitleAudioDuration']
            if not event['start']<=lo<hi<=event['end']+.001:
                raise RuntimeError('Measured speech window lies outside its event')
        start = round((lo + .2) * SR)
        end = round((hi - .2) * SR)
        if end <= start:
            raise RuntimeError(f"Cannot evaluate short event {event['id']}")
        begin = max(0, start - radius)
        stop = min(len(master), end + radius)
        position, correlation = compare(voice[start:end], master[begin:stop])
        lag = (begin + position - start) / SR
        rows.append({"event": event["id"], "kind": event["kind"], "lagSeconds": lag,
                     'windowSeconds':[lo+.2,hi-.2],
                     "correlation": round(correlation, 6),
                     "pass": abs(lag) <= args.max_lag and correlation >= args.min_correlation})
    result = {"master": str(args.master.resolve()), "voice": str(args.voice.resolve()),
              "masterSha256": sha(args.master), "voiceSha256": sha(args.voice),
              "planSha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "method": f"{SR}Hz mono normalized FFT cross-correlation, every event interior; positive lag means master is later",
              "analysisSampleRate": SR,
              "spectralBand": "300–3500Hz" if args.speech_band else "broadband",
              'windowBasis':'measured full speech-file interior; excludes musical breath' if args.measured_speech_window else 'whole event interior including any post-speech breath',
              "events": len(rows), "maxAllowedLagSeconds": args.max_lag,
              "minimumAllowedCorrelation": args.min_correlation,
              "maxAbsoluteLagSeconds": max(abs(x["lagSeconds"]) for x in rows),
              "minimumCorrelation": min(x["correlation"] for x in rows),
              "timingPassed": all(abs(x["lagSeconds"]) <= args.max_lag for x in rows),
              "voicePresencePassed": all(x["correlation"] >= args.min_correlation for x in rows),
              "passed": all(x["pass"] for x in rows), "results": rows}
    if skipped:
        result["pictureEventsWithoutVoice"] = skipped
    if args.episode is not None:
        result["masterEpisodeRelative"] = args.master.resolve().relative_to(args.episode.resolve()).as_posix()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "results"}, indent=2))
    if not result["passed"]:
        raise RuntimeError("Final voice alignment/correlation requires investigation")


if __name__ == "__main__":
    main()
