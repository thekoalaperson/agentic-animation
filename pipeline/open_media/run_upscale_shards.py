#!/usr/bin/env python3
"""Use each existing H100 once, then assemble only after every shard succeeds."""
import argparse
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def main():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--portrait", action="store_true")
    ap.add_argument("--picture-only", action="store_true")
    ap.add_argument("--chunk-frames", type=int, default=600)
    ap.add_argument("--range-plan", type=Path)
    ap.add_argument("--reuse-cache", type=Path, action="append", default=[])
    args = ap.parse_args()
    devices = args.gpus.split(",")
    if len(set(devices)) != len(devices) or not all(x.isdigit() for x in devices):
        raise ValueError("Expected distinct numeric GPU IDs")
    args.work.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("upscale_master_h100.py")
    base = [sys.executable, "-u", str(script), str(args.input), str(args.output),
            "--model", str(args.model), "--work", str(args.work), "--shards", str(len(devices))]
    base += ["--chunk-frames",str(args.chunk_frames)]
    if args.range_plan: base += ["--range-plan", str(args.range_plan)]
    for cache in args.reuse_cache: base += ["--reuse-cache", str(cache)]
    if args.portrait: base += ["--portrait"]
    if args.picture_only: base += ["--picture-only"]
    processes, logs = [], []
    try:
        for index, device in enumerate(devices):
            log = (args.work / f"shard-{index:02d}.log").open("a")
            logs.append(log)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            processes.append(subprocess.Popen(base + ["--shard-index", str(index)], env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            print(f"GPU {device}: shard {index} PID {processes[-1].pid}", flush=True)
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"HD shard failed: {codes}; completed chunks remain resumable")
            if all(code == 0 for code in codes):
                break
            time.sleep(5)
        subprocess.run(base + ["--assemble-only"], check=True)
        print("HD_MASTER_COMPLETE", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
