#!/usr/bin/env python3
"""Audit that a stream-copy end-card append preserves the body's A/V clock."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def packets(path, stream):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', stream,
        '-show_packets', '-show_entries', 'packet=pts_time,dts_time',
        '-of', 'json', str(path)], text=True))['packets']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('body', type=Path)
    ap.add_argument('final', type=Path)
    ap.add_argument('output', type=Path)
    a = ap.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('Run media QA on the qualified GPU host, not the Mac')
    subprocess.run(['nvidia-smi', '-L'], check=True, capture_output=True)
    record = {'body': str(a.body), 'final': str(a.final),
              'bodySha256': sha(a.body), 'finalSha256': sha(a.final), 'streams': {}}
    for stream in ['v:0', 'a:0']:
        before, after = packets(a.body, stream), packets(a.final, stream)
        if len(after) < len(before):
            raise ValueError(f'Missing body packets: {stream}')
        shifts = [float(y[key]) - float(x[key])
                  for x, y in zip(before, after)
                  for key in ['pts_time', 'dts_time']]
        deviation = max(shifts) - min(shifts)
        if deviation > .002:
            raise ValueError(f'Non-uniform body clock shift: {stream}: {deviation}')
        record['streams'][stream] = {'bodyPackets': len(before),
                                    'minShiftSeconds': min(shifts),
                                    'maxShiftSeconds': max(shifts),
                                    'shiftSpreadSeconds': deviation}
    video, audio = record['streams']['v:0'], record['streams']['a:0']
    relative = max(abs(video[v] - audio[s])
                   for v in ['minShiftSeconds', 'maxShiftSeconds']
                   for s in ['minShiftSeconds', 'maxShiftSeconds'])
    if relative > .002:
        raise ValueError(f'End-card append changes body A/V alignment: {relative}')
    record.update(maxRelativeAvClockChangeSeconds=relative, toleranceSeconds=.002,
                  note='Includes AAC priming packet; uniform container-start shift is not a speech-sync change.',
                  passed=True)
    a.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
