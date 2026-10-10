#!/usr/bin/env python3
"""Secondary AudioSet warning pass for generated text-to-audio SFX.

This deliberately never issues the final reject verdict: AudioSet labels are
broad and can confuse designed textures with voices or music.  The selector
uses these warnings alongside Whisper and signal checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path



MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SUSPECT = (
    "speech", "conversation", "narration", "monologue", "singing", "choir",
    "chant", "vocal", "music", "song", "humming", "whispering", "scream",
    "yell", "shout",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--manifest", type=Path, help="Finite source-bound jobs; supports known-content controls")
    args = ap.parse_args()
    import librosa
    import torch
    from transformers import pipeline

    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Audio QA must run on the qualified GPU worker")

    classifier = pipeline(
        "audio-classification", model=MODEL, device=0, top_k=30,
    )
    rows = []
    jobs = (json.loads(args.manifest.read_text())["jobs"] if args.manifest else
            [{"output": str(path.relative_to(args.root))} for path in sorted(args.root.rglob("*.wav"))])
    for job in jobs:
        path = args.root / job["output"]
        if path.name.startswith("."):
            continue  # macOS archive metadata is not audio.
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if job.get("sha256") and job["sha256"] != source_sha:
            raise RuntimeError(f"Bound QA source changed: {path}")
        sr = 16_000
        samples, _ = librosa.load(path, sr=sr, mono=True)
        window = 10 * sr
        hop = 9 * sr
        starts = list(range(0, max(1, len(samples) - window + 1), hop))
        final_start = max(0, len(samples) - window)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        windows = []
        by_label = {}
        for start in sorted(set(starts)):
            chunk = samples[start:start + window]
            result = classifier({"array": chunk, "sampling_rate": sr})
            warned = [
                {**x, "windowStartSeconds": round(start / sr, 3)}
                for x in result
                if x["score"] >= args.threshold
                and any(term in x["label"].lower() for term in SUSPECT)
            ]
            for item in warned:
                if item["label"] not in by_label or item["score"] > by_label[item["label"]]["score"]:
                    by_label[item["label"]] = item
            windows.append({
                "startSeconds": round(start / sr, 3),
                "topLabels": result[:10], "suspicious": warned,
            })
        suspicious = sorted(by_label.values(), key=lambda x: x["score"], reverse=True)
        rows.append({
            "file": path.name,
            "sha256": source_sha,
            "job": job.get("id"),
            "controlExpectation": job.get("controlExpectation"),
            "warning": bool(suspicious),
            "suspicious": suspicious,
            "windows": windows,
        })
        print(path.name, "WARN" if suspicious else "clear", suspicious[:3], flush=True)

    payload = {
        "model": MODEL,
        "modelRevision": getattr(classifier.model.config, "_commit_hash", None),
        "manifestSha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None,
        "role": "secondary warning only; never sole rejection evidence",
        "windowing": "10-second windows with 1-second overlap; final window always included",
        "threshold": args.threshold,
        "files": len(rows),
        "warnings": sum(x["warning"] for x in rows),
        "results": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("files", "warnings", "threshold")}, indent=2))


if __name__ == "__main__":
    main()
