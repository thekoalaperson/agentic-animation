#!/usr/bin/env bash
# Wait for this owned server, then execute one finite, hash-bound manifest.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: run_ready_finite.sh CLIENT MANIFEST OUT MAX_JOBS' >&2
  exit 64
fi
client=$1
manifest=$2
output=$3
maximum=$4
runtime_root=${AGENTIC_H3_ROOT:?set the qualified H3 runtime root}
[[ "$runtime_root" == /* && -x "$runtime_root/sgl/bin/python" ]]
startup_deadline=$((SECONDS + 900))
until curl --fail --silent --max-time 5 http://127.0.0.1:30010/health >/dev/null; do
  if (( SECONDS >= startup_deadline )); then
    echo 'Server startup deadline reached; no generation submitted' >&2
    exit 75
  fi
  sleep 5
done
exec "$runtime_root/sgl/bin/python" "$client" --manifest "$manifest" --out "$output" --max-jobs "$maximum"
