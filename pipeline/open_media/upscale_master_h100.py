#!/usr/bin/env python3
"""Resumable H100 neural 720p→1080p master reconstruction.

Picture passes through official RealESRGAN_x4plus before area reduction to
1080p. There is no whole-frame resize-only fallback. Explicit reviewed regional
source repairs can be supplied at assembly and remain separately hash-bound.
Chunks make a spot-VM render resumable without reloading the model per frame.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from hd_audio_remux import remux_approved_audio
from hd_range_cache import PIXEL_CONTRACT, range_layout, load_reuse_index, layout_fingerprint


def run(cmd, **kwargs):
    return subprocess.run([str(x) for x in cmd], check=True, **kwargs)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def video_info(path: Path) -> dict:
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-threads", "8", "-count_frames", "-show_streams",
        "-of", "json", str(path),
    ], text=True))
    return next(row for row in data["streams"] if row["codec_type"] == "video")


def packet_hash(path: Path, stream: str) -> str:
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", stream,
        "-c", "copy", "-f", "hash", "-hash", "sha256", "-",
    ], text=True).strip().split("=", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--chunk-frames", type=int, default=600)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--portrait", action="store_true", help="Authored 396x704,720x1280 or756x1344 portrait to1080x1920; source geometry is cache-bound")
    ap.add_argument("--picture-only", action="store_true", help="Reconstruct a silent picture before the final audio mix exists; no audio approval claim")
    ap.add_argument("--approved-repairs", type=Path, help="Explicit hash-bound regional repair overrides; assembly only")
    ap.add_argument("--range-plan", type=Path, help="Immutable shot-aligned source-frame ranges")
    ap.add_argument("--reuse-cache", type=Path, action="append", default=[], help="Prior read-only cache; exact decoded RGB identity is mandatory")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from spandrel import ImageModelDescriptor, ModelLoader

    if args.chunk_frames < 1 or args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid shard count/index")
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lifecycle_lock = (args.work / "lifecycle.lock").open("a+")
    fcntl.flock(lifecycle_lock, fcntl.LOCK_EX if args.assemble_only else fcntl.LOCK_SH)
    shard_lock = (args.work / f"shard-{args.shard_index:02d}.lock").open("a+")
    if not args.assemble_only:
        fcntl.flock(shard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    chunks = args.work / "chunks"
    chunks.mkdir(exist_ok=True)
    state_path = args.work / f"state-shard-{args.shard_index:02d}.jsonl"
    source_hash = sha(args.input)
    source_video_hash = packet_hash(args.input, "0:v:0")
    from hd_audio_remux import source_audio_identity
    source_audio_hash = source_audio_identity(args.input, args.picture_only)
    model_hash = sha(args.model)
    repairs = {}
    if args.approved_repairs:
        if not args.assemble_only:
            raise ValueError("Regional repairs apply only at final assembly")
        manifest = json.loads(args.approved_repairs.read_text())
        if manifest.get("approved") is not True or manifest.get("sourceVideoSha256") != source_video_hash or manifest.get("modelSha256") != model_hash:
            raise ValueError("Repair approval/source/model contract mismatch")
        repairs = manifest["chunks"]
        if args.output.exists():
            raise ValueError("Repaired assembly needs a new output path; preserve earlier masters")
    applied_repairs = []
    info = video_info(args.input)
    width, height = int(info["width"]), int(info["height"])
    fps = Fraction(info["r_frame_rate"])
    frames = int(info["nb_read_frames"])
    # Retain full H3 portrait detail after an authored6px crop on either side;
    # the older396x704 route remains available for historical wide-shot crops.
    source_sizes = {(396, 704), (720, 1280), (756, 1344)} if args.portrait else {(1280, 720)}
    target_width, target_height = (1080, 1920) if args.portrait else (1920, 1080)
    if (width, height) not in source_sizes or fps != Fraction(24, 1):
        raise ValueError(f"Expected one of{sorted(source_sizes)}/24, got {width}x{height}/{fps}")
    if abs(float(info["duration"]) - frames / 24) > 0.10:
        raise ValueError("Source frame count and duration disagree beyond mux rounding")
    range_plan = json.loads(args.range_plan.read_text()) if args.range_plan else None
    if range_plan and range_plan.get("sourceVideoSha256") != source_video_hash:
        raise ValueError("HD range plan points at a different source picture")
    layout = range_layout(frames, args.chunk_frames, range_plan)
    if not args.assemble_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; resize-only fallback is forbidden")
    cache_contract = {
        "sourceVideoSha256": source_video_hash,
        "modelSha256": model_hash,
        "sourceResolution": [width, height], "sourceFps": "24/1",
        "targetResolution": [target_width, target_height], "targetFps": "24/1",
        "reconstruction": "RealESRGAN-x4plus-fp16-area-1080-sequential-v2",
    }
    configuration = {**cache_contract, "shards": args.shards, "chunkFrames": args.chunk_frames}
    if range_plan:
        configuration["rangeLayoutSha256"] = layout_fingerprint(layout)
    config_lock = (args.work / "configuration.lock").open("a+")
    fcntl.flock(config_lock, fcntl.LOCK_EX)
    config_path = args.work / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise RuntimeError("Work directory has a different shard/source/model contract; use a versioned work directory")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    fcntl.flock(config_lock, fcntl.LOCK_UN)
    reuse_index = load_reuse_index(args.reuse_cache, cache_contract)

    latest = {}
    for prior_path in sorted(args.work.glob("state*.jsonl")):
        for line in prior_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("time", 0) >= latest.get(row["chunk"], {}).get("time", 0):
                    latest[row["chunk"]] = row
            except (ValueError, KeyError):
                pass

    def record(row: dict) -> None:
        row["time"] = time.time()
        with state_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    if not args.assemble_only:
        descriptor = ModelLoader().load_from_state_dict(
            torch.load(args.model, map_location="cpu", weights_only=True)
        )
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError("Expected an image reconstruction model")
        if descriptor.scale != 4:
            raise RuntimeError(f"Expected qualified 4x reconstruction model, got scale={descriptor.scale}")
        model = descriptor.eval().to(device="cuda", dtype=torch.float16)
        model.model.to(memory_format=torch.channels_last)
        print(json.dumps({"event": "model-loaded", "gpu": torch.cuda.get_device_name(0), "scale": descriptor.scale, "shard": args.shard_index}), flush=True)

    def reconstruct(frame: np.ndarray) -> bytes:
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1)[None]
        tensor = tensor.to(device="cuda", dtype=torch.float16, memory_format=torch.channels_last) / 255
        with torch.inference_mode():
            pred = model(tensor)
            if not torch.isfinite(pred).all():
                raise FloatingPointError("Non-finite neural reconstruction")
            pred = F.interpolate(pred.float().clamp_(0, 1), size=(target_height, target_width), mode="area")
            pixels = (pred[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy().tobytes()
        del tensor, pred
        return pixels

    chunk_paths = []
    # Decode exactly once per shard. Timestamp seeking can start at frame N+1
    # on concatenated media; frame-ordinal draining cannot silently skip a cut.
    decoder = None
    frame_bytes = width * height * 3
    if not args.assemble_only:
        decoder = subprocess.Popen([
            "ffmpeg", "-v", "error", "-threads", "4", "-reinit_filter", "0", "-i", str(args.input),
            "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ], stdout=subprocess.PIPE)
        def reap_decoder():
            if decoder.poll() is None:
                decoder.kill()
                decoder.wait()
        atexit.register(reap_decoder)

    def drain(count):
        remaining = count * frame_bytes
        while remaining:
            data = decoder.stdout.read(min(remaining, 8 * 1024 * 1024))
            if not data:
                raise EOFError("Missing source frame while draining another shard/cached range")
            remaining -= len(data)

    for chunk_index, planned_range in enumerate(layout):
        start, count, chunk_id = planned_range["startFrame"], planned_range["frames"], planned_range["id"]
        output = chunks / f"{chunk_id}.mp4"
        if not args.assemble_only and chunk_index % args.shards != args.shard_index:
            drain(count)
            continue
        prior = latest.get(chunk_id, {})
        contract = {**cache_contract, "startFrame": start, "frames": count}
        contract_matches = all(prior.get(key) == value for key, value in contract.items())
        if prior.get("status") == "complete" and contract_matches and output.exists() and prior.get("sha256") == sha(output):
            chosen = output
            if chunk_id in repairs:
                repair = repairs[chunk_id]
                chosen = Path(repair["path"])
                evidence = Path(repair["auditPath"])
                if repair["originalSha256"] != prior["sha256"] or sha(chosen) != repair["sha256"] or sha(evidence) != repair["auditSha256"]:
                    raise ValueError(f"Regional repair hash mismatch: {chunk_id}")
                proof = json.loads(evidence.read_text())
                repair_range = proof.get("globalFrameRange", [])
                if (len(repair_range) != 2 or not all(isinstance(x, int) for x in repair_range)
                    or not start <= repair_range[0] < repair_range[1] <= start + count):
                    raise ValueError(f"Regional repair interval mismatch: {chunk_id}")
                ranges = [r.get("globalFrameRange",repair_range) for r in proof.get("regions", [])]
                if not ranges or any(len(r)!=2 or not all(type(x) is int for x in r) or not repair_range[0]<=r[0]<r[1]<=repair_range[1] for r in ranges):
                    raise ValueError(f"Regional mask interval mismatch: {chunk_id}")
                covered = sum(any(lo<=f<hi for lo,hi in ranges) for f in range(*repair_range))
                if proof.get("repairedFrames") != covered:
                    raise ValueError(f"Regional mask frame count mismatch: {chunk_id}")
                if (proof.get("approved") is not True or proof.get("kind") not in {"source-preserving-sky-repair", "source-preserving-background-repair", "source-preserving-regional-detail-repair"}
                    or proof.get("sourceContainerSha256") != source_hash or proof.get("originalChunkSha256") != prior["sha256"]
                    or proof.get("chunkStartFrame") != start or proof.get("outputFrames") != count
                    or proof.get("outputSha256") != repair["sha256"]):
                    raise ValueError(f"Regional repair provenance mismatch: {chunk_id}")
                repaired_info = video_info(chosen)
                if (int(repaired_info["width"]), int(repaired_info["height"]), int(repaired_info["nb_read_frames"]), Fraction(repaired_info["r_frame_rate"])) != (target_width,target_height,count,Fraction(24,1)):
                    raise ValueError(f"Regional repair frame contract mismatch: {chunk_id}")
                applied_repairs.append({"chunk": chunk_id, **repair, "recipe": proof})
            chunk_paths.append(chosen)
            if decoder is not None:
                drain(count)
            continue
        if args.assemble_only:
            raise RuntimeError(f"Cannot assemble absent, stale or unfinished HD chunk {chunk_id}")
        # A bounded picture-run buffer permits one exact-pixel lookup before
        # inference. Source chronology may move without invalidating its pixels.
        raw_chunk = decoder.stdout.read(count * frame_bytes)
        if len(raw_chunk) != count * frame_bytes:
            raise EOFError(f"Unexpected input EOF at range {chunk_id}")
        rgb_hash = hashlib.sha256(raw_chunk).hexdigest()
        cached = reuse_index.get((rgb_hash, count))
        if cached:
            partial = output.with_suffix(".reuse.partial.mp4")
            shutil.copy2(cached["path"], partial)
            if sha(partial) != cached["state"]["sha256"]:
                raise ValueError(f"Reused HD output changed: {chunk_id}")
            check = video_info(partial)
            if (int(check["width"]), int(check["height"]), int(check["nb_read_frames"]), Fraction(check["r_frame_rate"])) != (target_width,target_height,count,Fraction(24,1)):
                raise ValueError(f"Reused HD geometry/clock differs: {chunk_id}")
            partial.replace(output)
            previous = cached["state"]
            record({"chunk": chunk_id, "status": "complete", **contract,
                    "sourceContainerSha256": source_hash, "sourceRgbSha256": rgb_hash,
                    "sourcePixelContract": PIXEL_CONTRACT, "sha256": sha(output),
                    "seconds": 0, "fallback": False, "neuralInferenceSkipped": True,
                    "reusedFrom": {"path": str(cached["path"]), "chunk": previous["chunk"],
                                   "sourceVideoSha256": previous["sourceVideoSha256"],
                                   "sourceContainerSha256": previous.get("sourceContainerSha256"),
                                   "startFrame": previous["startFrame"], "frames": count,
                                   "outputSha256": previous["sha256"],
                                   "configurationSha256": cached["configurationSha256"]}})
            chunk_paths.append(output)
            del raw_chunk
            continue
        t0 = time.time()
        record({"chunk": chunk_id, "status": "running", **contract, "sourceContainerSha256": source_hash})
        partial = output.with_suffix(".partial.mp4")
        encoder = subprocess.Popen([
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{target_width}x{target_height}", "-r", "24", "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "16", "-threads", "16", "-g", "48",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
        ], stdin=subprocess.PIPE)
        try:
            for offset in range(count):
                raw = memoryview(raw_chunk)[offset * frame_bytes:(offset + 1) * frame_bytes]
                if len(raw) != width * height * 3:
                    raise EOFError(f"Unexpected input EOF at frame {start + offset}")
                frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
                encoder.stdin.write(reconstruct(frame))
                if offset % 120 == 0:
                    print(chunk_id, offset + 1, "/", count, "elapsed", round(time.time() - t0, 1), flush=True)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("Chunk decode/encode failed")
            check = video_info(partial)
            if (int(check["width"]), int(check["height"]), int(check["nb_read_frames"])) != (target_width, target_height, count):
                raise ValueError("HD chunk contract failed")
            run(["ffmpeg", "-v", "error", "-i", partial, "-f", "null", "-"])
            partial.replace(output)
            record({"chunk": chunk_id, "status": "complete", **contract, "sourceContainerSha256": source_hash,
                    "sourceRgbSha256": rgb_hash, "sourcePixelContract": PIXEL_CONTRACT,
                    "sha256": sha(output), "seconds": round(time.time() - t0, 3), "fallback": False})
            chunk_paths.append(output)
            del raw_chunk
        except BaseException:
            for process in (decoder, encoder):
                if process.poll() is None:
                    process.kill()
                    process.wait()
            raise

    if decoder is not None:
        if decoder.stdout.read(1):
            decoder.kill()
            decoder.wait()
            raise RuntimeError("Unexpected trailing decoded source frame")
        decoder.stdout.close()
        if decoder.wait():
            raise RuntimeError("Source decoder failed")
    if sha(args.input) != source_hash or sha(args.model) != model_hash:
        raise RuntimeError("Source/model changed during reconstruction")
    if args.shards > 1 and not args.assemble_only:
        print(json.dumps({"event": "shard-complete", "shard": args.shard_index, "chunks": len(chunk_paths)}), flush=True)
        return
    listing = args.work / f"{args.output.stem}.concat.txt"
    if len(applied_repairs) != len(repairs):
        raise ValueError("Unused or unverified regional repair in manifest")
    # MP4 format durations are millisecond-rounded. Repeated implicit concat
    # offsets drift across many short ranges despite retaining every frame.
    # Exact authored durations followed by sub-tick rounding restore the grid;
    # this changes timestamps only, never decoded frames or cached packets.
    listing.write_text("".join(f"file '{path.resolve()}'\nduration {r['frames']/24:.12f}\n"
                               for path,r in zip(chunk_paths,layout)))
    silent = args.work / f"{args.output.stem}.picture-only.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing,
         "-c", "copy", "-bsf:v", "setts=pts=round(PTS*TB*24)/(TB*24):dts=round(DTS*TB*24)/(TB*24)",
         "-an", silent])
    partial = args.output.with_suffix(".partial.mp4")
    if args.picture_only:
        shutil.copy2(silent, partial)
        timing_audit = {"mode": "picture-only", "audioPresent": False,
                        "note": "HD inference is independent of sound approval; final mix is a separate mux."}
    else:
        timing_audit = remux_approved_audio(silent, args.input, partial)
    decoded = run(["ffmpeg", "-v", "error", "-xerror", "-reinit_filter", "0",
                   "-i", partial, "-fps_mode", "passthrough", "-enc_time_base:v", "-1",
                   "-f", "null", "-"], capture_output=True, text=True)
    if decoded.stderr.strip():
        raise RuntimeError(f"HD full-decode emitted errors: {decoded.stderr}")
    final_info = video_info(partial)
    if (int(final_info["width"]), int(final_info["height"]), int(final_info["nb_read_frames"])) != (target_width, target_height, frames):
        raise ValueError("Final HD contract failed")
    if Fraction(final_info['avg_frame_rate']) != Fraction(24):
        raise ValueError('Final HD concat clock must be exactly24fps')
    final_audio_hash = source_audio_identity(partial, args.picture_only)
    if final_audio_hash != source_audio_hash:
        raise ValueError("Approved AAC packet payload changed during HD remux")
    partial.replace(args.output)
    audit = {
        "input": str(args.input), "inputSha256": source_hash,
        "sourceVideoPacketSha256": source_video_hash,
        "sourceAudioPacketSha256": source_audio_hash,
        "outputAudioPacketSha256": final_audio_hash,
        "approvedAudioPacketsUnchanged": None if args.picture_only else True,
        "pictureOnly": args.picture_only,
        "timing": timing_audit,
        "output": str(args.output), "outputSha256": sha(args.output),
        "model": args.model.name, "modelSha256": model_hash,
        "sourceFrames": frames, "outputFrames": frames, "fps": 24,
        "resolution": [target_width, target_height], "neuralScale": 4,
        "normalization": "GPU area reduction after x4 neural reconstruction",
        "regionalSourceRepairs": applied_repairs,
        "fallback": False, "decode": "passed",
    }
    (args.work / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    # A versioned output owns its audit even if another audio-only assembly
    # later reuses the same picture cache and replaces work/audit.json.
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
