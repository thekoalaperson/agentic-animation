#!/usr/bin/env python3
"""Conservative, hash-bound unintended-speech screen for generated SFX.

This is an automatic quarantine aid, not a creative approval.  Whisper can
hallucinate words from noisy effects, so the report preserves every score and
requires a later isolated/context audition.  A credible hit is nevertheless a
hard reason not to auto-admit the source into a film mix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
from typing import Any, Callable, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def under_deliverables(path: Path) -> bool:
    return "deliverables" in path.resolve().parts


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def credible_rows(segments: Iterable[Any]) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        text = str(getattr(segment, "text", "")).strip()
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", start))
        average = float(getattr(segment, "avg_logprob", -999.0))
        no_speech = float(getattr(segment, "no_speech_prob", 1.0))
        if text and end - start >= 0.30 and average >= -0.80 and no_speech <= 0.45:
            rows.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                    "avgLogprob": round(average, 4),
                    "noSpeechProbability": round(no_speech, 4),
                }
            )
    return rows


class SimpleSegment:
    """Attribute adapter for openai-whisper's segment dictionaries."""

    def __init__(self, row: dict[str, Any]):
        self.text = row.get("text", "")
        self.start = row.get("start", 0.0)
        self.end = row.get("end", self.start)
        self.avg_logprob = row.get("avg_logprob", -999.0)
        self.no_speech_prob = row.get("no_speech_prob", 1.0)


def audit(
    directory: Path,
    output: Path,
    model_name: str,
    transcribe: Callable[[Path], tuple[Iterable[Any], Any]],
) -> dict[str, Any]:
    directory = directory.resolve()
    output = output.resolve()
    if under_deliverables(output):
        raise ValueError("SFX audits belong in work/, never deliverables/")
    wavs = sorted(directory.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV sources in {directory}")
    results = []
    for wav in wavs:
        segments, info = transcribe(wav)
        credible = credible_rows(segments)
        results.append(
            {
                "file": wav.name,
                "path": str(wav),
                "sha256": sha256(wav),
                "status": "quarantine" if credible else "no-credible-speech-detected",
                "detectedLanguage": getattr(info, "language", None),
                "languageProbability": round(float(getattr(info, "language_probability", 0.0)), 4),
                "credibleSpeech": credible,
            }
        )
    payload = {
        "schemaVersion": 1,
        "route": "offline-whisper-unintended-speech-screen",
        "model": model_name,
        "policy": (
            "Automatic quarantine aid only. Any credible hit blocks automatic source admission; "
            "all sources still require isolated and in-context listening because ASR can hallucinate on SFX."
        ),
        "sourceDirectory": str(directory),
        "count": len(results),
        "quarantined": sum(row["status"] == "quarantine" for row in results),
        "results": results,
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="small")
    parser.add_argument("--backend", choices=("auto", "faster-whisper", "openai-whisper"), default="auto")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Production SFX inference requires a qualified Linux GPU worker")
    if under_deliverables(args.output):
        raise SystemExit("SFX audits belong in work/, never deliverables/")
    backend = args.backend
    if backend == "auto":
        if importlib.util.find_spec("faster_whisper"):
            backend = "faster-whisper"
        elif importlib.util.find_spec("whisper"):
            backend = "openai-whisper"
        else:
            raise SystemExit("Neither faster_whisper nor openai-whisper is installed")

    if backend == "faster-whisper":
        from faster_whisper import WhisperModel

        model = WhisperModel(
            args.model, device=args.device,
            compute_type="float16" if args.device == "cuda" else "int8",
        )

        def transcribe(path: Path):
            return model.transcribe(
                str(path),
                language=None,
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=False,
                no_speech_threshold=0.55,
                log_prob_threshold=-0.8,
                compression_ratio_threshold=2.4,
            )
    else:
        import whisper

        model = whisper.load_model(args.model, device=args.device)

        def transcribe(path: Path):
            result = model.transcribe(
                str(path),
                fp16=args.device == "cuda",
                condition_on_previous_text=False,
                no_speech_threshold=0.55,
                logprob_threshold=-0.8,
                compression_ratio_threshold=2.4,
            )
            segments = [SimpleSegment(row) for row in result.get("segments", [])]
            info = type("WhisperInfo", (), {
                "language": result.get("language"), "language_probability": 0.0
            })()
            return segments, info

    result = audit(args.directory, args.output, f"{backend}:{args.model}", transcribe)
    print(
        json.dumps(
            {"count": result["count"], "quarantined": result["quarantined"], "audit": str(args.output.resolve())},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
