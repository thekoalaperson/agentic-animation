"""Recheck a retained H3 take against an explicitly corrected frame contract.

No model calls; preserve original state and output files. Linux GPU worker only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def run(source, destination, key, attempt_number, frames):
    assert platform.system() == 'Linux'
    subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], check=True, capture_output=True)
    original = source / 'state.json'
    state = json.loads(original.read_text())
    assert state.get('finishedAt'), 'Source batch must be terminal'
    repaired = copy.deepcopy(state)
    job = repaired['jobs'][key]
    attempt = job['attempts'][attempt_number - 1]
    assert attempt['number'] == attempt_number
    assert attempt['completedResponse']['status'] == 'completed'
    raw = source / f'{key}.attempt{attempt_number}.mp4'
    assert sha(raw) == attempt['downloadSha256']
    probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(raw)], check=True, capture_output=True, text=True, timeout=120).stdout)
    video = [s for s in probe['streams'] if s['codec_type'] == 'video']
    assert len(video) == 1
    v = video[0]
    assert (v['width'], v['height'], v['r_frame_rate']) == (1344, 768, '24/1')
    assert int(v['nb_read_frames']) == frames
    assert abs(float(v['duration']) - frames / 24) < .002
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(raw), '-f', 'null', '-'], check=True, capture_output=True, timeout=180)
    assert not decoded.stderr
    job.update(status='complete', output=raw.name, sha256=sha(raw), creativeVerdict='pending-independent-review')
    repaired['completeCount'] = sum(j.get('status') == 'complete' for j in repaired['jobs'].values())
    repaired['reconciliation'] = dict(originalStateSha256=sha(original), job=key, selectedAttempt=attempt_number, correctedExpectedFrames=frames, probe=probe, reason='Original expected-frame assertion was incorrect; existing downloaded take passes corrected explicit contract. Original attempts and errors preserved.')
    destination.mkdir(parents=True, exist_ok=True)
    for j in repaired['jobs'].values():
        if j.get('status') != 'complete':
            continue
        src = source / j['output']
        assert src.parent == source and sha(src) == j['sha256']
        dst = destination / src.name
        if dst.exists():
            assert sha(dst) == sha(src)
        else:
            shutil.copy2(src, dst)
    target = destination / 'state.json'
    payload = json.dumps(repaired, indent=2) + '\n'
    if target.exists():
        assert target.read_text() == payload
    else:
        with target.open('x') as out:
            out.write(payload)
    print(json.dumps(dict(state=str(target), complete=repaired['completeCount'], frames=frames)))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--attempt', type=int, required=True)
    p.add_argument('--frames', type=int, required=True)
    a = p.parse_args()
    run(a.source.resolve(), a.destination.resolve(), a.key, a.attempt, a.frames)
