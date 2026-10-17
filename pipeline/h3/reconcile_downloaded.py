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

if not __debug__:
    raise SystemExit("Optimized Python disables production safety gates")


class ReconcileError(ValueError):
    """The retained take or its recorded state does not match the stated contract."""


def require(condition, message):
    if not condition:
        raise ReconcileError(message)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def run(source, destination, key, attempt_number, frames):
    require(platform.system() == 'Linux', 'reconciliation runs on the qualified Linux GPU worker')
    subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], check=True, capture_output=True)
    original = source / 'state.json'
    state = json.loads(original.read_text())
    require(state.get('finishedAt'), f'source batch must be terminal: {original} finishedAt')
    repaired = copy.deepcopy(state)
    job = repaired['jobs'][key]
    attempt = job['attempts'][attempt_number - 1]
    require(attempt['number'] == attempt_number, f'{key}: attempt {attempt_number} is recorded as {attempt["number"]}')
    require(attempt['completedResponse']['status'] == 'completed',
            f'{key} attempt {attempt_number}: completedResponse.status is not completed')
    raw = source / f'{key}.attempt{attempt_number}.mp4'
    require(sha(raw) == attempt['downloadSha256'], f'{raw}: bytes differ from downloadSha256')
    probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(raw)], check=True, capture_output=True, text=True, timeout=120).stdout)
    video = [s for s in probe['streams'] if s['codec_type'] == 'video']
    require(len(video) == 1, f'{raw}: expected one video stream, found {len(video)}')
    v = video[0]
    require((v['width'], v['height'], v['r_frame_rate']) == (1344, 768, '24/1'),
            f'{raw}: geometry {v["width"]}x{v["height"]} at {v["r_frame_rate"]} is not 1344x768 at 24/1')
    require(int(v['nb_read_frames']) == frames, f'{raw}: {v["nb_read_frames"]} frames, expected {frames}')
    require(abs(float(v['duration']) - frames / 24) < .002, f'{raw}: duration {v["duration"]} does not match {frames} frames')
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(raw), '-f', 'null', '-'], check=True, capture_output=True, timeout=180)
    require(not decoded.stderr, f'{raw}: decode reported errors')
    job.update(status='complete', output=raw.name, sha256=sha(raw), creativeVerdict='pending-independent-review')
    repaired['completeCount'] = sum(j.get('status') == 'complete' for j in repaired['jobs'].values())
    repaired['reconciliation'] = dict(originalStateSha256=sha(original), job=key, selectedAttempt=attempt_number, correctedExpectedFrames=frames, probe=probe, reason='Original expected-frame assertion was incorrect; existing downloaded take passes corrected explicit contract. Original attempts and errors preserved.')
    destination.mkdir(parents=True, exist_ok=True)
    for j in repaired['jobs'].values():
        if j.get('status') != 'complete':
            continue
        src = source / j['output']
        require(src.parent == source and sha(src) == j['sha256'], f'{src}: output escapes the batch or its sha256 changed')
        dst = destination / src.name
        if dst.exists():
            require(sha(dst) == sha(src), f'{dst}: existing destination differs from {src}')
        else:
            shutil.copy2(src, dst)
    target = destination / 'state.json'
    payload = json.dumps(repaired, indent=2) + '\n'
    if target.exists():
        require(target.read_text() == payload, f'{target}: existing reconciled state differs')
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
    try:
        run(a.source.resolve(), a.destination.resolve(), a.key, a.attempt, a.frames)
    except ReconcileError as exc:
        raise SystemExit(f'REFUSED: {exc}') from exc
