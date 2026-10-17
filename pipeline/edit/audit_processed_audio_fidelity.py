#!/usr/bin/env python3
"""Qualify encoded audio against the exact lossless mastering transform, not dry voice.

This supplements (never rewrites) dry-voice correlation when documented nonlinear
social compression changes waveform shape. No speech or listening verdict.
"""
import argparse
import json
import platform
import socket
import subprocess
from pathlib import Path

import audit_master_voice_sync as sync


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=Path)
    p.add_argument('receipt', type=Path)
    p.add_argument('plan', type=Path)
    p.add_argument('output_dir', type=Path)
    a = p.parse_args()
    assert platform.system() == 'Linux', 'GPU worker only'
    subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], check=True)
    ep = a.episode.resolve()
    receipt_path = (ep / a.receipt).resolve()
    receipt = json.loads(receipt_path.read_text())
    master = (ep / receipt['output']).resolve()
    master.relative_to(ep)
    assert sync.sha(master) == receipt['outputSha256']
    controlled = master.with_suffix('.controlled.wav')
    assert sync.sha(controlled) == receipt['socialCompression']['controlledSha256']
    plan_path = (ep / a.plan).resolve()
    plan = json.loads(plan_path.read_text())
    out = (ep / a.output_dir).resolve()
    out.relative_to(ep / 'work')
    out.mkdir(parents=True, exist_ok=False)
    expected = out / 'expected-processed-48kf32.wav'
    filters = receipt['normalizer'] + ',' + receipt['encodedCorrections'][-1]['filter']
    if receipt.get('sampleClockFilter'):
        assert receipt['sampleClockFilter']=='aresample=48000,asetpts=N/SR/TB'
        filters+=','+receipt['sampleClockFilter']
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(controlled), '-vn',
                    '-af', filters, '-t', str(receipt['runtimeSeconds']),
                    '-ar', '48000', '-ac', '2', '-c:a', 'pcm_f32le', str(expected)], check=True)
    reference, candidate = sync.load(expected), sync.load(master)
    rows = []
    for event in plan['events']:
        if event.get('kind') == 'picture':
            continue  # wordless picture has no speech window to compare
        lo = event['start'] + event['subtitleAudioAt'] + .2
        hi = event['start'] + event['subtitleAudioAt'] + event['subtitleAudioDuration'] - .2
        start, end = round(lo * sync.SR), round(hi * sync.SR)
        begin, stop = max(0, start - 1200), min(len(candidate), end + 1200)
        pos, correlation = sync.compare(reference[start:end], candidate[begin:stop])
        lag = (begin + pos - start) / sync.SR
        rows.append(dict(event=event['id'], windowSeconds=[lo, hi],
                         lagSeconds=lag, correlation=correlation,
                         passed=abs(lag) <= .001 and correlation >= .99))
    result = dict(schemaVersion=1, kind='transform-qualified-encoded-audio-fidelity',
                  computeHost=socket.gethostname(), receiptSha256=sync.sha(receipt_path),
                  masterSha256=sync.sha(master), controlledSha256=sync.sha(controlled),
                  expectedProcessedSha256=sync.sha(expected), planSha256=sync.sha(plan_path),
                  exactTransform=filters, events=len(rows), results=rows,
                  passed=all(x['passed'] for x in rows),
                  note='Encoded fidelity and timing relative to documented processed full mix only. '
                       'Does not supersede a failed dry-voice correlation or establish speech accuracy, '
                       'naturalness, taste, hearing, or creative approval.')
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    assert result['passed'], 'Encoded result differs materially from qualified mastering transform'


if __name__ == '__main__':
    main()
