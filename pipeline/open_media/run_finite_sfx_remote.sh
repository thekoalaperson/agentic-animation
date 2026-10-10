#!/usr/bin/env bash
# Run a pre-staged finite queue. Caller owns the qualified worker lifecycle.
set -euo pipefail
job_root=${1:?absolute job directory required}
max_jobs=${2:?explicit finite job ceiling required}
audio_python=${AGENTIC_AUDIO_PYTHON:?set the qualified Stable Audio Python executable}
[[ "$job_root" == /* && -d "$job_root" && -x "$audio_python" ]]
[[ "$max_jobs" =~ ^[0-9]+$ ]] && (( max_jobs >= 1 && max_jobs <= 100 ))
exec 9>/tmp/agentic-audio-gpu.lock
flock -n 9 || { echo 'Another audio queue owns the GPU'; exit 73; }
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$gpu_processes" ]]; then
  echo 'GPU is already occupied; refusing to start'; exit 73
fi
cd "$job_root"
exec /usr/bin/timeout --signal=TERM --kill-after=60 7000 \
  "$audio_python" -u \
  "$job_root/run_stable_audio_probe.py" "$job_root" --max-jobs "$max_jobs"
