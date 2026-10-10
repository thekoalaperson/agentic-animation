"""Build an offline, original-file SFX inspection page; never alters media."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote


def read(path):
    return json.loads(path.read_text())


def waveform(path, bins=360):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le',
                          '-ac', '2', '-ar', '8000', '-'], capture_output=True, check=True)
    if raw.stderr:
        raise ValueError(raw.stderr.decode())
    samples = array.array('f', raw.stdout)
    if sys.byteorder != 'little':
        samples.byteswap()
    if not samples:
        raise ValueError(f'Empty audio: {path}')
    size = math.ceil(len(samples) / bins)
    peaks = [round(min(1.0, max(abs(v) for v in samples[i:i+size])), 5)
             for i in range(0, len(samples), size)]
    return peaks, len(samples) / 16000


def build_standalone(ep, output, batches):
    """A new episode's sources, without requiring a legacy episode tree."""
    assert batches, 'At least one audited source batch is required'
    families = {}
    assets = []
    for base in batches:
        base = base.resolve()
        plan = read(base / 'probe-brief.json')
        audit = read(base / 'batch-audit.json')
        local = read(base / 'local-audit.json')
        assert local['technicalPass'] and local['count'] == len(plan['jobs'])
        assert plan['generationApiAllowed'] is False
        assert audit['runtime']['briefSha256'] == hashlib.sha256((base / 'probe-brief.json').read_bytes()).hexdigest()
        audited = {row['id']: row for row in audit['outputs']}
        for job in plan['jobs']:
            path = base / job['output']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            row = audited[job['id']]
            assert row['sha256'] == digest and row['job'] == job
            peaks, duration = waveform(path)
            assert abs(duration - job['seconds']) < .001
            family_id = job.get('family', job['id'])
            family = families.setdefault(family_id, dict(
                id=family_id, title=family_id.replace('-', ' ').title(),
                category='authored', intent=job.get('editorialIntent') or job.get('editorialNote', ''),
                shots=job.get('shots', job.get('scenes', job.get('shotAnchors', []))), placements=[], candidates=[]))
            candidate = dict(key=family_id+':'+digest, id=job['id'], hash=digest,
                family=family_id, path=str(path),
                url=quote(os.path.relpath(path, output.parent), safe='/'),
                batches=[base.name], prompt=job['prompt'], negativePrompt='',
                model=plan['model'], seed=job['seed'], steps=audit['runtime']['steps'],
                guidance=audit['runtime']['cfgScale'], duration=duration, peaks=peaks,
                selected=False, used=False, historicalReview=None, derivation=None,
                newSource=True, initialProbe=False, label='Untouched source / unreviewed',
                sampleRate=row['sampleRate'], channels=row['channels'], peak=row['peak'],
                overFullScaleSamples=row['overFullScaleSamples'],
                generationSeconds=row['inferenceAndSaveSeconds'])
            family['candidates'].append(candidate)
            assets.append(candidate)
    board_path = ep/'story/storyboard_plan.json'
    code = str(read(board_path).get('episode', ep.name)) if board_path.is_file() else ep.name
    revision = hashlib.sha256(json.dumps([x['hash'] for x in assets]).encode()).hexdigest()[:16]
    data = dict(episode=ep.name, episodeCode=code, standalone=True,
        reviewStore=code.lower()+'-sfx-debug-'+revision, downloadName=code+'_SFX_DEBUG_REVIEW.json',
        currentVerdict='unreviewed', families=list(families.values()),
        source='Self-hosted Stable Audio3 Small-SFX, text-to-audio only.',
        newCandidates=len(assets), totalCandidates=len(assets), usedFamilies=0,
        focusBatch=batches[-1].name, plannedMythicCandidates=0,
        designStatus='Scene associations authored; final sync awaits measured voice/picture and selected score.')
    template = Path(__file__).with_name('sfx_debug_template.html').read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('/*__DATA__*/null', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')))
    print(json.dumps(dict(page=str(output), sources=len(assets), seconds=sum(c['duration'] for c in assets), mediaDecoded=True)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--batch', '--new-batch', dest='batches', type=Path, action='append', required=True,
                        help='Audited Stable Audio batch; repeat for multiple palettes')
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('SFX waveform decoding belongs on the configured GPU worker')
    build_standalone(args.episode.resolve(), args.output.resolve(), args.batches)
