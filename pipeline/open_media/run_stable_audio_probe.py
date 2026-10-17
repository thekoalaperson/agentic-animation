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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--max-jobs', type=int, default=2,
                        help='Explicit authorized batch ceiling; default is the two-source probe')
    args = parser.parse_args()
    import soundfile as sf
    import torch
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.model_configs import all_models

    root = args.root.resolve()
    brief = root / 'probe-brief.json'
    plan = json.loads(brief.read_text())
    assert plan['generationApiAllowed'] is False
    assert plan['model'] == 'stabilityai/stable-audio-3-small-sfx'
    assert 1 <= len(plan['jobs']) <= args.max_jobs <= 100
    if len(plan['jobs']) > 2:
        assert plan.get('authorization') and plan.get('productionApproval') is False
    ids = [j['id'] for j in plan['jobs']]
    assert len(ids) == len(set(ids))
    assert len({j['output'] for j in plan['jobs']}) == len(ids)
    for job in plan['jobs']:
        assert job['prompt'].strip() and 0 < job['seconds'] <= 10
        assert (root / job['output']).resolve().is_relative_to(root / 'outputs')
    (root / 'outputs').mkdir(exist_ok=True)
    with (root / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert torch.cuda.is_available(), 'GPU required; no Mac/CPU fallback'
        started = time.monotonic()
        print('Resolving official Small-SFX weights using authenticated local cache', flush=True)
        config, weights = all_models['small-sfx'].resolve()
        runtime = dict(model=plan['model'], configPath=config, configSha256=sha(config),
                       weightsPath=weights, weightsSha256=sha(weights),
                       torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                       codeSha256=sha(__file__), briefSha256=sha(brief),
                       steps=8, cfgScale=1.0, precision='float16',
                       normalization='none', sourceFormat='32-bit float WAV, native sample rate/channels')
        (root / 'runtime.json').write_text(json.dumps(runtime, indent=2))
        print('Loading one resident model', flush=True)
        model = StableAudioModel.from_pretrained('small-sfx', device='cuda', model_half=True)
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
            audio = model.generate(prompt=job['prompt'], duration=job['seconds'], seed=job['seed'],
                                   steps=8, cfg_scale=1.0, batch_size=1,
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
            row = dict(id=job['id'], status='complete', identity=identity, sha256=sha(target),
                       path=str(target), job=job, seconds=len(a)/sr, sampleRate=sr, channels=a.shape[1],
                       peak=float(np.max(np.abs(a))), rms=float(np.sqrt(np.mean(a*a))),
                       overFullScaleSamples=int(np.count_nonzero(np.abs(a)>1)),
                       inferenceAndSaveSeconds=time.monotonic()-t0,
                       peakGpuMemoryBytes=torch.cuda.max_memory_allocated(),
                       normalization='none; original float samples preserved', creativeVerdict='unreviewed')
            sidecar.write_text(json.dumps(row, indent=2))
            with state_path.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            rows.append(row)
            print(json.dumps(row), flush=True)
        assert sha(brief) == runtime['briefSha256']
        (root / 'batch-audit.json').write_text(json.dumps(dict(runtime=runtime, outputs=rows,
             completed=len(rows), totalSeconds=time.monotonic()-started, creativeVerdict='unreviewed'), indent=2))
        print(f'{len(rows)}-source review batch complete; no film mutation or automatic acceptance', flush=True)


if __name__ == '__main__':
    main()
