#!/usr/bin/env python3
"""Rank generated candidates by CLAP similarity to their authored SFX family."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np


MODEL = "laion/clap-htsat-unfused"


# The wrapper text the earlier briefs put around each family's phrase. A brief
# that wraps its prompts differently names its own in ``clapPromptTemplate``.
DEFAULT_TEMPLATE = {
    "cutAt": [" Dry, clean,"],
    "remove": ["Isolated cinematic sound effect. ", "TrackType: SFX. "],
    "negativeWord": "No",
}


def semantic_text(prompt: str, template: dict | None = None) -> str:
    # Negative clauses describe what must be absent and would pollute a positive
    # CLAP target. Keep the authored source/action/material/perspective phrase.
    template = DEFAULT_TEMPLATE if template is None else template
    text = prompt
    for marker in template.get("cutAt", []):
        text = text.split(marker)[0]
    for phrase in template.get("remove", []):
        text = text.replace(phrase, "")
    negative = template.get("negativeWord")
    if negative:
        text = re.split(r"\s+" + re.escape(negative) + r"\s+", text, maxsplit=1)[0]
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import librosa
    import torch
    from transformers import AutoProcessor, ClapModel


    manifest = json.loads(args.manifest.read_text())
    families = manifest["families"]
    family_index = {x["id"]: i for i, x in enumerate(families)}
    prompts = [semantic_text(x["prompt"], manifest.get("clapPromptTemplate")) for x in families]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if sys.platform != "linux" or device != "cuda":
        raise RuntimeError("Audio QA must run on the qualified GPU worker")
    model = ClapModel.from_pretrained(MODEL).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    with torch.inference_mode():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = model.get_text_features(**text_inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

    sr = 48_000
    window = 10 * sr
    rows = []
    for job in manifest["jobs"]:
        path = args.root / job["output"]
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if job.get("sha256") and job["sha256"] != source_sha:
            raise RuntimeError(f"Bound QA source changed: {path}")
        samples, _ = librosa.load(path, sr=sr, mono=True)
        starts = list(range(0, max(1, len(samples) - window + 1), 9 * sr))
        final_start = max(0, len(samples) - window)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        windows = []
        for start in sorted(set(starts)):
            chunk = samples[start:start + window]
            with torch.inference_mode():
                audio_inputs = processor(audios=chunk, sampling_rate=sr, return_tensors="pt")
                audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
                audio_features = model.get_audio_features(**audio_inputs)
                audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
                similarities = (audio_features @ text_features.T).squeeze(0).float().cpu().numpy()
            own = family_index[job["family"]]
            order = np.argsort(-similarities)
            rank = int(np.where(order == own)[0][0]) + 1
            windows.append({
                "startSeconds": round(start / sr, 3),
                "ownSimilarity": round(float(similarities[own]), 6),
                "ownRankAmongFamilies": rank,
                "bestFamily": families[int(order[0])]["id"],
                "bestSimilarity": round(float(similarities[order[0]]), 6),
            })
        score = float(np.mean([x["ownSimilarity"] for x in windows]))
        best_rank = min(x["ownRankAmongFamilies"] for x in windows)
        rows.append({
            "file": path.name, "job": job["id"], "family": job["family"],
            "sha256": source_sha,
            "controlExpectation": job.get("controlExpectation"),
            "semanticScore": round(score, 6), "bestOwnRank": best_rank,
            "windows": windows,
        })
        print(path.name, f"score={score:.4f}", f"best-rank={best_rank}", flush=True)

    payload = {
        "model": MODEL,
        "modelRevision": getattr(model.config, "_commit_hash", None),
        "manifestSha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "purpose": "candidate ranking within each authored family; not a standalone acceptance gate",
        "families": len(families), "files": len(rows), "results": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"model": MODEL, "families": len(families), "files": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
