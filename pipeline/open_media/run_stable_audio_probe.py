"""Small, resumable self-hosted Stable Audio 3 probe with untouched sources."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


if not __debug__:
    raise SystemExit("Optimized Python disables production safety gates")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# Per-model limits. A job may run to the qualified length. To go longer, first
# run a length-qualification batch: a brief with ``lengthQualificationRun: true``
# admits at most two jobs at any length, marks every output
# ``lengthQualified: false`` and writes a batch-audit.json that says it was such
# a run. A later brief names that retained audit (``lengthQualification``: its
# path inside the new job root and its SHA-256), and may then run to the longest
# length the audit measured. Steps and CFG scale are part of each job's
# identity; the defaults here are the qualified ones, and any other value is
# recorded as unqualified.
MODELS = {
    'stabilityai/stable-audio-3-small-sfx': {
        'configKey': 'small-sfx', 'qualifiedSeconds': 10, 'steps': 8, 'cfgScale': 1.0,
    },
}


def qualified_seconds(plan, root, model):
    """The longest job this batch may run, from the table or a retained qualification run."""
    limit = MODELS[model]['qualifiedSeconds']
    evidence = plan.get('lengthQualification')
    if evidence is None:
        return limit
    path = (root / evidence['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha(path) != evidence['sha256']:
        raise RuntimeError('lengthQualification must be a retained batch audit inside this job root, hash bound')
    audit = json.loads(path.read_text())
    runtime = audit.get('runtime', {})
    # Each row's seconds is measured from the written samples, and the runner
    # asserts it matches the job within one sample.
    lengths = [min(row['seconds'], row['job']['seconds']) for row in audit.get('outputs', [])
               if row.get('status') == 'complete' and row.get('job', {}).get('prompt')]
    if runtime.get('model') != model or runtime.get('lengthQualificationRun') is not True or not lengths:
        raise RuntimeError('lengthQualification is not a completed length-qualification run of this model')
    return max(limit, max(lengths))


def job_settings(job, model):
    """Steps and CFG scale for one job: its own, or the model's qualified defaults."""
    steps = job.get('steps', MODELS[model]['steps'])
    cfg = job.get('cfgScale', MODELS[model]['cfgScale'])
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 250:
        raise RuntimeError(f"{job['id']}: steps must be a whole number from 1 to 250")
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not 0 <= cfg <= 20:
        raise RuntimeError(f"{job['id']}: cfgScale must be a number from 0 to 20")
    return steps, float(cfg)


def validate_plan(plan, root, max_jobs):
    """Every refusal that needs no GPU, checked before the model is loaded."""
    assert plan['generationApiAllowed'] is False
    model = plan.get('model')
    if model not in MODELS:
        raise RuntimeError(f'No qualified local route for SFX model {model}')
    assert 1 <= len(plan['jobs']) <= max_jobs <= 100
    if len(plan['jobs']) > 2:
        assert plan.get('authorization') and plan.get('productionApproval') is False
    ids = [j['id'] for j in plan['jobs']]
    assert len(ids) == len(set(ids))
    assert len({j['output'] for j in plan['jobs']}) == len(ids)
    qualifying = plan.get('lengthQualificationRun', False)
    if not isinstance(qualifying, bool):
        raise RuntimeError('lengthQualificationRun must be true or false')
    if qualifying and 'lengthQualification' in plan:
        raise RuntimeError('A length-qualification run cannot also cite a lengthQualification')
    if qualifying and len(plan['jobs']) > 2:
        raise RuntimeError('A length-qualification run admits at most two jobs')
    limit = qualified_seconds(plan, root, model)
    for job in plan['jobs']:
        assert job['prompt'].strip()
        if qualifying:
            if isinstance(job['seconds'], bool) or not isinstance(job['seconds'], (int, float)) \
                    or not 0 < job['seconds'] < float('inf'):
                raise RuntimeError(f"{job['id']}: seconds must be a positive finite number")
        elif not 0 < job['seconds'] <= limit:
            raise RuntimeError(f"{job['id']}: {job['seconds']} s is beyond the {limit} s this model "
                               'is qualified for here; retain a qualification run at that length first '
                               '(a brief with lengthQualificationRun: true)')
        job_settings(job, model)
        assert (root / job['output']).resolve().is_relative_to(root / 'outputs')
    return model


def maximum_job_seconds(plan, root, model):
    """What the runtime record states as the batch's length bound."""
    if plan.get('lengthQualificationRun') is True:
        return max(job['seconds'] for job in plan['jobs'])
    return qualified_seconds(plan, root, model)


def output_row(plan, job, model, identity, target_sha256, target, samples, sr, steps, cfg_scale,
               inference_seconds, peak_gpu_bytes):
    """The sidecar and audit row for one completed output."""
    a = samples
    row = dict(id=job['id'], status='complete', identity=identity, sha256=target_sha256,
               path=str(target), job=job, seconds=len(a)/sr, sampleRate=sr, channels=a.shape[1],
               peak=float(np.max(np.abs(a))), rms=float(np.sqrt(np.mean(a*a))),
               overFullScaleSamples=int(np.count_nonzero(np.abs(a)>1)),
               steps=steps, cfgScale=cfg_scale,
               qualifiedSettings=(steps, cfg_scale) == (MODELS[model]['steps'], MODELS[model]['cfgScale']),
               inferenceAndSaveSeconds=inference_seconds,
               peakGpuMemoryBytes=peak_gpu_bytes,
               normalization='none; original float samples preserved', creativeVerdict='unreviewed')
    if plan.get('lengthQualificationRun') is True:
        row['lengthQualified'] = False
    return row


def write_batch_audit(root, runtime, rows, total_seconds):
    """The batch audit; a length-qualification run's audit is later evidence for longer jobs."""
    (root / 'batch-audit.json').write_text(json.dumps(dict(runtime=runtime, outputs=rows,
         completed=len(rows), totalSeconds=total_seconds, creativeVerdict='unreviewed'), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--max-jobs', type=int, default=2,
                        help='Explicit authorized batch ceiling; default is the two-source probe')
    args = parser.parse_args()
    root = args.root.resolve()
    brief = root / 'probe-brief.json'
    plan = json.loads(brief.read_text())
    model_name = validate_plan(plan, root, args.max_jobs)
    import soundfile as sf
    import torch
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.model_configs import all_models

    (root / 'outputs').mkdir(exist_ok=True)
    with (root / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert torch.cuda.is_available(), 'GPU required; no Mac/CPU fallback'
        started = time.monotonic()
        print('Resolving official Small-SFX weights using authenticated local cache', flush=True)
        config, weights = all_models[MODELS[model_name]['configKey']].resolve()
        runtime = dict(model=plan['model'], configPath=config, configSha256=sha(config),
                       weightsPath=weights, weightsSha256=sha(weights),
                       torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                       codeSha256=sha(__file__), briefSha256=sha(brief),
                       steps=MODELS[model_name]['steps'], cfgScale=MODELS[model_name]['cfgScale'],
                       maximumJobSeconds=maximum_job_seconds(plan, root, model_name), precision='float16',
                       normalization='none', sourceFormat='32-bit float WAV, native sample rate/channels')
        if plan.get('lengthQualificationRun') is True:
            runtime['lengthQualificationRun'] = True
        (root / 'runtime.json').write_text(json.dumps(runtime, indent=2))
        print('Loading one resident model', flush=True)
        model = StableAudioModel.from_pretrained(MODELS[model_name]['configKey'], device='cuda', model_half=True)
        sr = int(model.model_config['sample_rate'])
        runtime['sampleRate'] = sr
        runtime['modelLoadAndWeightResolutionSeconds'] = time.monotonic() - started
        (root / 'runtime.json').write_text(json.dumps(runtime, indent=2))
        state_path = root / 'state.jsonl'
        rows = []
        for job in plan['jobs']:
            target = root / job['output']
            sidecar = target.with_suffix('.json')
            identity = hashlib.sha256(json.dumps(dict(job=job, runtime=runtime['weightsSha256'], code=runtime['codeSha256']), sort_keys=True).encode()).hexdigest()
            if target.exists() or sidecar.exists():
                old = json.loads(sidecar.read_text())
                assert old['identity'] == identity and sha(target) == old['sha256']
                rows.append(old)
                print('REUSE ' + job['id'], flush=True)
                continue
            t0 = time.monotonic()
            with state_path.open('a') as stream:
                stream.write(json.dumps(dict(id=job['id'], status='running', identity=identity)) + '\n')
            torch.cuda.reset_peak_memory_stats()
            steps, cfg_scale = job_settings(job, model_name)
            audio = model.generate(prompt=job['prompt'], duration=job['seconds'], seed=job['seed'],
                                   steps=steps, cfg_scale=cfg_scale, batch_size=1,
                                   truncate_output_to_duration=True)
            torch.cuda.synchronize()
            a = audio.detach().float().cpu().numpy()
            if a.ndim == 3:
                assert a.shape[0] == 1
                a = a[0]
            assert a.ndim == 2 and a.shape[0] in (1, 2)
            a = a.T
            assert np.isfinite(a).all()
            assert abs(len(a) / sr - job['seconds']) <= 1 / sr
            sf.write(target, a, sr, subtype='FLOAT')
            decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(target), '-f', 'null', '-'], capture_output=True, check=True)
            assert not decoded.stderr
            row = output_row(plan, job, model_name, identity, sha(target), target, a, sr, steps, cfg_scale,
                             time.monotonic()-t0, torch.cuda.max_memory_allocated())
            sidecar.write_text(json.dumps(row, indent=2))
            with state_path.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            rows.append(row)
            print(json.dumps(row), flush=True)
        assert sha(brief) == runtime['briefSha256']
        write_batch_audit(root, runtime, rows, time.monotonic()-started)
        print(f'{len(rows)}-source review batch complete; no film mutation or automatic acceptance', flush=True)


if __name__ == '__main__':
    main()
