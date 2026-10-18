"""Build a portable, source-bound listening review; no audio processing or API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote


if not __debug__:
    raise SystemExit("Optimized Python disables production safety gates")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(report_path, draft_path, output):
    report = json.loads(report_path.read_text())
    draft = json.loads(draft_path.read_text())
    assert sha(draft_path) == report['bindings']['creativeDraftSha256']
    sources = draft_path.parent / 'sources'
    rows = []
    for cue in draft['cues']:
        takes = []
        for candidate in report['candidates']:
            if candidate['cueId'] != cue['id']:
                continue
            media = sources / Path(candidate['file']).name
            sidecar = sources / Path(candidate['sidecar']).name
            assert sha(media) == candidate['sha256']
            assert sha(sidecar) == candidate['sidecarSha256']
            takes.append({
                'id': candidate['id'], 'sha256': candidate['sha256'],
                'url': quote(os.path.relpath(media, output.parent), safe='/'),
                'duration': candidate['features']['durationSeconds'],
                'short': candidate['durationFlag'].startswith('SHORTER'),
                'requested': candidate['requestedDurationSeconds'],
                'energy': candidate['features']['trajectory']['rmsEnergy'],
            })
        # Version 1 reports predate the field and always meant two takes.
        expected = report.get('expectedCandidatesPerMovement', 2)
        assert len(takes) == expected, f"{cue['id']}: expected {expected} retrieved takes"
        rows.append({'id': cue['id'], 'shots': cue['shots'],
                     'title': cue['payload'].get('title', cue['id']),
                     'prompt': cue['payload'].get('style', ''),
                     'takes': sorted(takes, key=lambda x: x['id']),
                     'arc': cue.get('emotionalArc'),
                     'wave': report['targets'][cue['id']]['authoredSemanticWave']})
    data = {'episode': report['episode'], 'reportSha256': sha(report_path),
            'draftSha256': sha(draft_path), 'rows': rows}
    template = Path(__file__).with_name('score_candidate_review_template.html').read_text()
    assert template.count('__SCORE_DATA__') == 1
    html = template.replace('__SCORE_DATA__', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c'))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        assert output.read_text() == html, 'Choose a versioned output for changed inputs'
    else:
        with output.open('x') as stream:
            stream.write(html)
    print(json.dumps({'page': str(output), 'movements': len(rows),
                      'candidates': sum(len(r['takes']) for r in rows), 'mediaHashesVerified': True,
                      'playbackTested': False, 'pageSha256': sha(output)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('draft', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    build(a.report.resolve(), a.draft.resolve(), a.output.resolve())
