#!/usr/bin/env python3
"""Self-hosted, resumable audio observations; never a media approval.

Run either cached Whisper speech transcription or Qwen2-Audio music descriptions
on one owned GPU. Inputs are local files on the configured GPU worker. Public model
weights may download from Hugging Face; private audio never goes to an API.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mode", choices=("speech", "music"), required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--speech-model", default="base")
    parser.add_argument("--observation-label", default=None)
    parser.add_argument("--question", default=None,
                        help="Explicit neutral observation prompt; no intended scene or answer")
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    plan = json.loads(args.manifest.read_text())
    jobs = [j for j in plan["jobs"] if j["mode"] == args.mode]
    if len(jobs) != args.jobs or not 0 < len(jobs) <= 160:
        raise ValueError("exact finite job count required")
    for job in jobs:
        source = (root / job["file"]).resolve()
        source.relative_to(root)
        if sha(source) != job["sha256"]:
            raise ValueError(f"source changed: {job['id']}")
    label = args.observation_label or args.mode
    if Path(label).name != label or label in ('.', '..'):
        raise ValueError('observation label must be a simple name')
    output = root / "observations" / label
    output.mkdir(parents=True, exist_ok=True)
    lock = root / ".gpu-audio-observer.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        import numpy as np
        import torch
        if args.mode == "speech":
            import whisper
            model = whisper.load_model(args.speech_model, device="cuda")
            model_info = {"model": "openai/whisper-" + args.speech_model,
                          "torch": torch.__version__}
        else:
            import transformers
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            name = "Qwen/Qwen2-Audio-7B-Instruct"
            processor = AutoProcessor.from_pretrained(name)
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                name, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda")
            model.eval()
            model_info = {"model": name, "revision": model.config._commit_hash,
                          "torch": torch.__version__, "transformers": transformers.__version__,
                          "audioSurface": "16kHz mono; no stereo-image verdict"}
        print(json.dumps({"modelLoaded": model_info}), flush=True)
        for job in jobs:
            destination = output / (job["id"] + ".json")
            if destination.exists():
                old = json.loads(destination.read_text())
                if old.get("sourceSha256") != job["sha256"] or old.get("status") != "COMPLETED":
                    raise ValueError(f"incompatible prior observation: {job['id']}")
                continue
            source = root / job["file"]
            started = time.time()
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-vn",
                "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                check=True, capture_output=True).stdout
            samples = np.frombuffer(raw, dtype=np.float32).copy()
            result = {"id": job["id"], "sourceSha256": job["sha256"], "model": model_info,
                      "durationSeconds": len(samples) / 16000, "advisoryOnly": True}
            if args.mode == "speech":
                transcription = model.transcribe(samples, language="hi", task="transcribe",
                    condition_on_previous_text=False, fp16=True, word_timestamps=True,
                    verbose=False)
                result.update(text=transcription["text"], segments=transcription["segments"])
            else:
                result["windows"] = []
                for start in range(0, len(samples), 28 * 16000):
                    clip = samples[start:start + 28 * 16000]
                    if len(clip) < 8000:
                        continue
                    question = args.question or ("Describe the music and sounds you hear. Identify instruments, "
                        "voice or wordless humming if present, rhythm and tempo character, mood, "
                        "and any change during this excerpt. Do not guess the intended scene. "
                        "Be concrete and concise; state uncertainty.")
                    conversation = [{"role": "user", "content": [
                        {"type": "audio", "audio_url": "local-excerpt.wav"},
                        {"type": "text", "text": question}]}]
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    # Retained transformers4.x uses audios;5.x uses audio.
                    # Select the installed explicit interface, never permit the
                    # processor to silently discard audio as an unknown kwarg.
                    parameters=inspect.signature(processor.__call__).parameters
                    audio_key='audios' if 'audios' in parameters else 'audio'
                    if audio_key not in parameters:
                        raise ValueError('Unqualified processor audio interface')
                    inputs = processor(text=text, **{audio_key:[clip]}, sampling_rate=16000,
                                       return_tensors="pt", padding=True).to("cuda")
                    if "input_features" not in inputs or not inputs["input_features"].numel():
                        raise ValueError('processor did not encode audio; refuse text-only observations')
                    with torch.inference_mode():
                        ids = model.generate(**inputs, max_new_tokens=220, do_sample=False)
                    description = processor.batch_decode(ids[:, inputs.input_ids.size(1):],
                        skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    result["windows"].append({"start": start / 16000,
                        "end": (start + len(clip)) / 16000, "question": question,
                        "processorAudioArgument": audio_key, "observation": description})
                    del inputs, ids
            result.update(status="COMPLETED", elapsedSeconds=time.time() - started)
            save(destination, result)
            print(json.dumps({"id": job["id"], "status": "COMPLETED",
                              "seconds": round(result["elapsedSeconds"], 2)}), flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
