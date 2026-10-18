#!/usr/bin/env python3
"""Transcribe dialogue clips on a worker and compare each with its frozen line.

Advisory evidence for a person to listen to, never a rejection or an approval.
For each clip of a dialogue execution manifest it records the transcript, the
span of transcribed speech, the words of the frozen line it did not hear, the
words it heard that the line does not contain, and the clip's measured frame
rate and duration. ASR mishears names, soft words and ambience, so every
finding is a listening cue. Whether a second voice speaks is not checked here.

The transcript spans feed prepare_dialogue_review.py --measure-speech-windows
--transcript-evidence, so an energy-based window never cuts heard speech.
Runs on a qualified Linux worker with faster-whisper or openai-whisper.
Unexercised: only the comparison and the measurement core, with the Whisper
call replaced, are tested; there is no recorded worker run, so qualify it on
one clip before reading its findings. The thresholds are Whisper's library
defaults (no-speech 0.6, log-prob -1.0, compression ratio 2.4), beam 5 as in
faster-whisper (openai-whisper's library call defaults to greedy), and
conditioning on previous text turned off. They are not tuned on any clips
and are not the effects screen's stricter values.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Callable, Iterable
import unicodedata

if __package__:
    from .audit_sfx_speech import SimpleSegment, atomic_json, sha256
else:
    from audit_sfx_speech import SimpleSegment, atomic_json, sha256

# Whisper library defaults, except conditioning on previous text is off.
THRESHOLDS = {"noSpeechThreshold": 0.6, "logProbThreshold": -1.0,
              "compressionRatioThreshold": 2.4, "conditionOnPreviousText": False,
              "beamSize": 5}
PUNCTUATION = re.compile(r"[।॥.,!?;:\"'‘’“”()\[\]\-–—…]")
JOINERS = re.compile(r"[‌‍]")
NUKTA = "़"


def words(text: str) -> list[str]:
    """NFC words with dandas and punctuation removed; joiners and nukta kept."""
    return PUNCTUATION.sub(" ", unicodedata.normalize("NFC", text)).split()


def loose(word: str) -> str:
    return JOINERS.sub("", unicodedata.normalize("NFD", word).replace(NUKTA, ""))


def compare_line(frozen: str, heard: str) -> dict[str, Any]:
    expected, actual = words(frozen), words(heard)
    matcher = difflib.SequenceMatcher(a=[loose(w) for w in expected],
                                      b=[loose(w) for w in actual], autojunk=False)
    missing, extra, spelling = [], [], []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            missing.extend(expected[a0:a1])
        if tag in ("insert", "replace"):
            extra.extend(actual[b0:b1])
        if tag == "equal":
            spelling.extend([e, a] for e, a in zip(expected[a0:a1], actual[b0:b1]) if e != a)
    return {"exactAfterNormalization": expected == actual,
            "sameIgnoringNuktaAndJoiners": not missing and not extra,
            "missingWords": missing, "extraWords": extra,
            "nuktaOrJoinerDifferences": spelling}


def probe(path: Path) -> dict[str, Any]:
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                      "-of", "json", str(path)], capture_output=True, text=True,
                                     check=True, timeout=120).stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"fps": video.get("r_frame_rate"), "durationSeconds": float(info["format"]["duration"])}


def measure(jobs: Iterable[tuple[dict[str, Any], Path, str]], output: Path, *,
            model_name: str, language: str, manifest_sha256: str,
            transcribe: Callable[[Path], tuple[Iterable[Any], Any]],
            probe_fn: Callable[[Path], dict[str, Any]] = probe) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite transcript evidence: {output}")
    items = []
    for row, video, digest in jobs:
        if sha256(video) != digest:
            raise ValueError(f"clip changed before transcription: {row['id']}")
        segments, info = transcribe(video)
        heard = [{"start": round(float(getattr(s, "start", 0.0)), 3),
                  "end": round(float(getattr(s, "end", 0.0)), 3),
                  "text": str(getattr(s, "text", "")).strip(),
                  "avgLogprob": round(float(getattr(s, "avg_logprob", -999.0)), 4),
                  "noSpeechProbability": round(float(getattr(s, "no_speech_prob", 1.0)), 4)}
                 for s in segments]
        heard = [s for s in heard if s["text"]]
        transcript = " ".join(s["text"] for s in heard)
        items.append({
            "id": row["id"], "source": str(video), "sourceSha256": digest,
            "frozenLine": row["hindi"], "transcript": transcript, "segments": heard,
            "transcriptSpan": ({"start": heard[0]["start"], "end": heard[-1]["end"]}
                               if heard else None),
            "detectedLanguage": getattr(info, "language", None),
            "comparison": compare_line(row["hindi"], transcript),
            "secondVoice": "not checked",
            "measured": probe_fn(video),
        })
    report = {"schemaVersion": 1, "kind": "dialogue-speech-transcripts",
              "advisoryOnly": True, "creativeApproval": False,
              "model": model_name, "language": language, "thresholds": THRESHOLDS,
              "computeHost": platform.node(), "executionManifestSha256": manifest_sha256,
              "policy": ("Listening cues only. A mismatch is a reason to listen, never a "
                         "rejection; ASR mishears names, soft words and ambience."),
              "items": items}
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="dialogue execution manifest")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, required=True)
    parser.add_argument("--ids", help="explicit comma-separated subset")
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON file below the project's work/")
    parser.add_argument("--language", default="hi", help="language the frozen lines are in")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--backend", choices=("auto", "faster-whisper", "openai-whisper"),
                        default="auto")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Speech transcription runs on a qualified Linux worker")
    if __package__:
        from .prepare_dialogue_review import select_jobs, verified_jobs
    else:
        from prepare_dialogue_review import select_jobs, verified_jobs
    root, manifest = args.project.resolve(), args.manifest.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root / "work"):
        raise SystemExit("Transcript evidence belongs below the project's work/")
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    jobs = [(row, video, digest) for row, video, digest, _ in verified_jobs(
        plan, manifest, root, manifest.parent, select_jobs(plan, args.ids, args.max_jobs))]
    backend = args.backend
    if backend == "auto":
        backend = ("faster-whisper" if importlib.util.find_spec("faster_whisper")
                   else "openai-whisper" if importlib.util.find_spec("whisper") else None)
        if backend is None:
            raise SystemExit("Neither faster_whisper nor openai-whisper is installed")
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="cuda", compute_type="float16")

        def transcribe(path: Path):
            return model.transcribe(
                str(path), language=args.language, beam_size=THRESHOLDS["beamSize"],
                condition_on_previous_text=False,
                no_speech_threshold=THRESHOLDS["noSpeechThreshold"],
                log_prob_threshold=THRESHOLDS["logProbThreshold"],
                compression_ratio_threshold=THRESHOLDS["compressionRatioThreshold"])
    else:
        import whisper
        model = whisper.load_model(args.model, device="cuda")

        def transcribe(path: Path):
            result = model.transcribe(
                str(path), language=args.language, fp16=True,
                condition_on_previous_text=False, beam_size=THRESHOLDS["beamSize"],
                no_speech_threshold=THRESHOLDS["noSpeechThreshold"],
                logprob_threshold=THRESHOLDS["logProbThreshold"],
                compression_ratio_threshold=THRESHOLDS["compressionRatioThreshold"])
            info = type("WhisperInfo", (), {"language": result.get("language")})()
            return [SimpleSegment(row) for row in result.get("segments", [])], info

    report = measure(jobs, output, model_name=f"{backend}:{args.model}", language=args.language,
                     manifest_sha256=sha256(manifest), transcribe=transcribe)
    print(json.dumps({"items": len(report["items"]), "output": str(output),
                      "withDifferences": sum(not x["comparison"]["sameIgnoringNuktaAndJoiners"]
                                             for x in report["items"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
