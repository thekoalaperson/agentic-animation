"""Verify copied self-hosted sources against immutable jobs and remote audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import socket
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    if sys.platform != 'linux' or not shutil.which('nvidia-smi'):
        raise RuntimeError('Production source decoding belongs on the qualified GPU worker')
    brief_path = root / 'probe-brief.json'
    plan = json.loads(brief_path.read_text())
    audit = json.loads((root / 'batch-audit.json').read_text())
    runtime = audit['runtime']
    assert sha(brief_path) == runtime['briefSha256']
    assert plan['model'] == runtime['model'] == 'stabilityai/stable-audio-3-small-sfx'
    assert plan['generationApiAllowed'] is False
    assert plan['productionApproval'] is False
    jobs = plan['jobs']
    assert len(jobs) == plan.get('expectedJobs', len(jobs)) == audit['completed']
    rows = {r['id']: r for r in audit['outputs']}
    assert len(rows) == len(jobs) == len({j['id'] for j in jobs})
    assert sum(j['seconds'] for j in jobs) == plan.get('expectedSeconds', sum(j['seconds'] for j in jobs))

    def one(job):
        path = root / job['output']
        assert path.resolve().is_relative_to((root / 'outputs').resolve())
        row = rows[job['id']]
        sidecar = json.loads(path.with_suffix('.json').read_text())
        assert sidecar == row and row['job'] == job and row['status'] == 'complete'
        digest = sha(path)
        assert digest == row['sha256']
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)],
                               check=True, capture_output=True)
        streams = json.loads(probe.stdout)['streams']
        assert len(streams) == 1
        stream = streams[0]
        assert stream['codec_name'] == 'pcm_f32le'
        assert int(stream['sample_rate']) == row['sampleRate'] == 44100
        assert stream['channels'] == row['channels'] == 2
        assert abs(float(stream['duration'])-job['seconds']) <= 1/44100
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                                 check=True, capture_output=True)
        assert not decoded.stderr
        return dict(id=job['id'], sha256=digest, duration=job['seconds'],
                    overFullScaleSamples=row['overFullScaleSamples'], verified=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, jobs))
    result = dict(technicalPass=True, count=len(results), seconds=sum(j['seconds'] for j in jobs),
                  computeHost=socket.gethostname(), verifierSha256=sha(Path(__file__)),
                  briefSha256=sha(brief_path), remoteAuditSha256=sha(root / 'batch-audit.json'),
                  creativeVerdict='unreviewed', filmChanged=False,
                  overFullScaleFiles=[r['id'] for r in results if r['overFullScaleSamples']],
                  outputs=results)
    (root / 'local-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'outputs'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    verify(p.parse_args().root.resolve())
