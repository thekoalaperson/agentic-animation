"""Build a local per-take review from audited footage and explicit narration provenance.

``--audit`` is the inventory ``audit_story_coverage.py`` writes. Each
``--h3-audit`` (``review_finite.py``'s ``local-audit.json``) and
``--dialogue-index`` (``prepare_dialogue_review.py``'s
``dense-review-index.json``) adds its takes to the row whose ID the job ID
names, with the frame sheet and a line saying what that check covered. The
page records notes only; it admits nothing and infers no approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_story_coverage import row_for


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add(shot, candidate):
    """Join evidence onto a take the inventory already lists, or list it."""
    for existing in shot['candidates']:
        if existing['path'] == candidate['path']:
            existing.update({k: v for k, v in candidate.items() if k in ('sheet', 'checked', 'checkedBy')})
            return
    shot['candidates'].append(candidate)


def add_evidence(data, project, h3_audits=(), dialogue_indexes=()):
    """Attach worker review evidence to the inventory's rows; return what found no row."""
    ids = {str(shot['id']) for shot in data['shots']}
    by_id = {str(shot['id']): shot for shot in data['shots']}
    unplaced = []

    def place(job_id, candidate):
        key = row_for(str(job_id), ids, by_id)
        if key is None:
            unplaced.append(str(job_id))
        else:
            _add(by_id[key], candidate)

    for audit_path in h3_audits:
        audit_path = audit_path.resolve()
        audit = json.loads(audit_path.read_text())
        for row in audit.get('outputs', []):
            silent = row.get('silent') if isinstance(row, dict) else None
            if row.get('status') != 'technical-pass' or not isinstance(silent, dict):
                continue
            recorded = Path(str(silent.get('file') or ''))
            path = recorded if recorded.is_relative_to(project) and recorded.is_file() \
                else audit_path.parent / 'silent' / recorded.name
            if not path.resolve().is_relative_to(project) or not path.is_file():
                unplaced.append(str(row.get('id')))
                continue
            sheet = Path(str(row.get('sheet') or ''))
            sheet = sheet if sheet.is_relative_to(project) and sheet.is_file() else audit_path.parent / 'review' / sheet.name
            place(row.get('id'), dict(
                path=str(path.resolve().relative_to(project)), kind='h3-base',
                hashVerified=_sha(path) == silent.get('sha256'), sha256=silent.get('sha256'),
                evidence=str(audit_path.relative_to(project)),
                **({'sheet': sheet} if sheet.name and sheet.is_file() else {}),
                checked=audit.get('frameSampling') or 'technical pass on the worker',
                checkedBy='review_finite.py', creativeApproval='not established by this check'))
    for index_path in dialogue_indexes:
        index_path = index_path.resolve()
        index = json.loads(index_path.read_text())
        for item in index.get('items', []):
            path = (index_path.parent / str(item.get('source', ''))).resolve()
            if not path.is_relative_to(project) or not path.is_file():
                unplaced.append(str(item.get('id')))
                continue
            sheet = (index_path.parent / str(item.get('sheet', ''))).resolve()
            place(item.get('id'), dict(
                path=str(path.relative_to(project)), kind='grok-dialogue',
                hashVerified=_sha(path) == item.get('sourceSha256'), sha256=item.get('sourceSha256'),
                evidence=str(index_path.relative_to(project)),
                **({'sheet': sheet} if item.get('sheet') and sheet.is_file() else {}),
                checked=item.get('reviewScope') or 'frame sheet', checkedBy='prepare_dialogue_review.py',
                creativeApproval='not established by this check'))
    return unplaced


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episode', '--project', dest='episode', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--narration-manifest', type=Path,
                        help='Optional narration manifest; paths resolve relative to that manifest')
    parser.add_argument('--h3-audit', type=Path, action='append', default=[],
                        help="review_finite.py's local-audit.json (repeatable)")
    parser.add_argument('--dialogue-index', type=Path, action='append', default=[],
                        help="prepare_dialogue_review.py's dense-review-index.json (repeatable)")
    args = parser.parse_args()
    project, output = args.episode.resolve(), args.output.resolve()
    data = json.loads(args.audit.read_text())
    data['episode'] = data.get('episode') or project.name
    unplaced = add_evidence(data, project, args.h3_audit, args.dialogue_index)
    narration = {}
    if args.narration_manifest:
        manifest_path = args.narration_manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
        for item in manifest['items']:
            path = (manifest_path.parent / item['output']).resolve()
            if not path.is_relative_to(project):
                raise ValueError('Narration output escapes the project')
            narration[item.get('shotId', item['id'].removesuffix('-hi'))] = path

    def url(path):
        path = path.resolve()
        if not path.is_relative_to(project) or not path.is_file():
            raise ValueError('Missing or escaped review asset')
        return quote(os.path.relpath(path, output.parent), safe='/')

    for shot in data['shots']:
        for candidate in shot['candidates']:
            candidate['url'] = url(project / candidate['path'])
            if isinstance(candidate.get('sheet'), Path):
                candidate['sheet'] = url(candidate['sheet'])
        voice = narration.get(shot.get('id'))
        shot['narration'] = url(voice) if voice and voice.is_file() and shot['speaker'] == 'Narrator' else None
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('story_review_template.html').read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__DATA__', payload))
    print(output)
    if unplaced:
        print(json.dumps({'evidenceWithNoRow': unplaced}))


if __name__ == '__main__':
    main()
