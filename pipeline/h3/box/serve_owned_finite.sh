#!/usr/bin/env bash
# Qualified H3 topology, without the historical kill-all launcher or watchdog.
set -euo pipefail
runtime_root=${AGENTIC_H3_ROOT:?set the qualified H3 runtime root}
[[ "$runtime_root" == /* && -d "$runtime_root/sgl/bin" ]]
exec 8>"$runtime_root/finite-server.lock"
flock -n 8 || { echo 'Another finite server owns the lock'; exit 73; }
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$gpu_processes" ]]; then
  echo 'GPU occupied; refusing to replace an existing workload'; exit 73
fi
export HF_HOME="$runtime_root/hf" HF_HUB_OFFLINE=1 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$runtime_root/sgl/bin:$CUDA_HOME/bin:$PATH" MAX_JOBS=32
# Qualified lossless inference: no Cache-DiT approximation for corrective pilot.
export SGLANG_CACHE_DIT_ENABLED=false
cd "$runtime_root"
exec "$runtime_root/sgl/bin/sglang" serve \
  --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --num-gpus 4 --tp-size 2 --ulysses-degree 2 --encoder-parallel auto \
  --performance-mode speed --attention-backend fa \
  --warmup-resolutions 1344x768 --port 30010
