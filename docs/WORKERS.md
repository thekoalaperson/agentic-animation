# Worker runtime contract

This repository does not provision a GPU fleet, contain model weights, or assume
a particular cloud account has running workers. Configure and qualify workers
before use. Historical instance names and prior receipts are not current inventory.

The Python lockfile covers the coordinator: manifests, HTTP orchestration, and
lightweight validation. Model runtimes remain separate:

| Lane | Runtime requirements |
|---|---|
| H3 | Compatible Linux CUDA worker, qualified SGLang/H3 server, model files, `requests` for finite client |
| HD | CUDA PyTorch, NumPy, Spandrel, selected licensed upscaler weights, ffmpeg/ffprobe |
| SFX | Qualified Stable Audio runtime, Torch, SoundFile, NumPy, appropriate model access |
| Audio diagnostics | Selected Whisper/Transformers/CLAP or AudioSet runtime; optional, separately qualified. `measure_dialogue_speech.py` uses the Whisper runtime and has no recorded worker run yet; its thresholds are Whisper's library defaults (no-speech 0.6, log-prob -1.0, compression 2.4) with beam 5 and conditioning on previous text off, not tuned on any clips and not the effects screen's values |
| Text/finishing | ffmpeg with complex libass/HarfBuzz shaping, explicit licensed fonts, GPU worker |

The finite H3 shell launchers require `AGENTIC_H3_ROOT`, whose qualified runtime
contains `sgl/bin/python` and the H3 server. The finite SFX shell launcher requires
`AGENTIC_AUDIO_PYTHON`. Set these explicitly on the selected worker; profile
examples do not provision environments or automatically export shell variables.
Cloud-copy helpers require explicit instance, project, and zone arguments.

Do not install every optional inference package into the coordinator. Record the
selected runtime's versions, model/font hashes, driver/CUDA compatibility, storage
paths, and successful qualification outputs in the private worker manifest.
Install from the relevant upstream project, honoring its actual model terms.

## Lifecycle

1. Inspect current instance/disk state and ownership. Reuse a qualified existing
   environment when it exists; otherwise report the missing dependency before
   provisioning or claiming that an old runbook can be resumed.
2. Resolve worker paths and account configuration explicitly. Check GPU, storage,
   interpreter, binary versions, model hashes, and source availability.
3. Transfer immutable inputs and verify hashes. Launch one owned finite queue
   with declared limits, heartbeat, deadline, and persisted receipts.
4. Poll and reconcile that queue. Do not kill unrelated processes, launch an
   overlapping watchdog, or infer failure solely from temporary quiet output.
5. Verify outputs and copy receipts/results back. Keep shot/chunk caches with
   their content identities so later edits invalidate only changed ranges.
6. Release ownership and stop billable compute when finished. Verify the actual
   stopped state; a submitted stop request alone does not prove shutdown.

The coordinator can be long-lived; GPU jobs remain finite. On host replacement,
transfer the queue lease deliberately and reconcile remote task IDs before any
new submission. Never resume a hash-bound job with a silently changed client.

Offline tests validate orchestration and media contracts. They do not certify a
missing model environment or prove visual/audio quality. A qualified worker
requires a bounded representative run with actual retained output evidence.
