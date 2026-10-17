"""Inventory local story coverage without decoding media or approving creativity.

A media inventory only: which local picture and dialogue files exist for each
storyboard row, and whether their recorded hashes hold. How the rows use the
screenplay is ``edit/read_screenplay.py --units``. Rows are keyed by the
storyboard's ``still_id``; a file belongs to the row whose ID its name or job
ID equals, or the longest ID it starts with followed by ``-``, ``_`` or ``.``.
Dialogue takes are read from each ``--execution-manifest`` given, or else from
``work/grok-dialogue-*/motion`` (and the older ``work/dialogue-*/motion``).
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def row_for(name, ids, by_shot):
    """Return the row key a file or job ID belongs to, or None."""
    if ids:
        if name in ids:
            return name
        matches = [i for i in ids if name.startswith(i) and name[len(i):len(i) + 1] in ('-', '_', '.')]
        return max(matches, key=len) if matches else None
    match = re.search(r'-s(\d+)', name)  # no storyboard: fall back to the ordinal
    return str(int(match[1])) if match and str(int(match[1])) in by_shot else None


def dialogue_sidecars(root, manifests):
    if manifests:
        for manifest in manifests:
            doc = json.loads(manifest.read_text())
            for job in doc.get('jobs', []):
                stem = (root / job['outputStem']).resolve()
                if not stem.is_relative_to(root):
                    raise SystemExit(f'Execution manifest output escapes the project: {manifest}: {job.get("id")}')
                yield job.get('id'), stem.with_name(stem.name + '.vid.json')
        return
    seen = set()
    for pattern in ('work/grok-dialogue-*/motion/*.vid.json', 'work/dialogue-*/motion/*.vid.json'):
        for sidecar in sorted(root.glob(pattern)):
            if sidecar not in seen:
                seen.add(sidecar)
                yield None, sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--script', help='Pipe script; default story/script.txt, else story/production-script-v2.tsv')
    parser.add_argument('--storyboard', default='story/storyboard_plan.json')
    parser.add_argument('--execution-manifest', type=Path, action='append', default=[],
                        help='Dialogue execution manifest whose jobs to inventory (repeatable)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.episode.resolve()
    if args.script:
        script = root / args.script
    else:
        script = next((root / p for p in ('story/script.txt', 'story/production-script-v2.tsv')
                       if (root / p).is_file()), root / 'story/script.txt')
    rows = []
    for line in script.read_text().splitlines():
        if not line.strip():
            continue
        parts = [v.strip() for v in line.split('|')]  # an optional fifth column carries delivery
        number, speaker, hindi, english = parts[:4]
        delivery = parts[4] if len(parts) > 4 else ''
        rows.append(dict(shot=number, speaker=speaker, hindi=hindi, english=english,
                         **({'delivery': delivery} if delivery else {}), candidates=[]))
    board_path = root / args.storyboard
    board = json.loads(board_path.read_text()) if board_path.is_file() else {}
    board_rows = board.get('stills', [])
    for row in rows:
        index = int(row['shot']) - 1
        row['id'] = board_rows[index]['still_id'] if 0 <= index < len(board_rows) else row['shot']
    ids = {row['id'] for row in rows} if board_rows else set()
    by_shot = {row['id']: row for row in rows} if ids else {str(int(row['shot'])): row for row in rows}
    errors = []

    def admit(path, expected, kind, receipt, extra, job_id=None):
        key = row_for(job_id, ids, by_shot) if job_id else None
        key = key or row_for(path.name, ids, by_shot)
        if key is None:
            errors.append(f'Unmapped asset: {path}')
            return
        valid = path.is_file() and bool(expected) and sha(path) == expected
        if not valid:
            errors.append(f'Missing or hash mismatch: {path}')
        by_shot[key]['candidates'].append(dict(
            path=str(path.relative_to(root)), kind=kind, hashVerified=valid,
            sha256=expected, evidence=str(receipt.relative_to(root)),
            creativeApproval='not established by this audit', **extra))

    for receipt in sorted(root.glob('work/h3-*/silent/ingest-receipt.json')):
        for item in json.loads(receipt.read_text())['items']:
            if item.get('audioStreams') != 0:
                errors.append(f'H3 audio policy violation: {receipt}: {item["id"]}')
            admit(receipt.parent / item['silent'], item.get('silentSha256'),
                  'h3-continuation' if 'cont' in item['id'] or 'continuation' in str(receipt) else 'h3-base',
                  receipt, dict(frames=item.get('frames'), audioStreams=item.get('audioStreams')),
                  job_id=item.get('id'))
    for job_id, sidecar in dialogue_sidecars(root, [m.resolve() for m in args.execution_manifest]):
        if not sidecar.is_file():
            continue  # not landed yet; the runner's own state says why
        item = json.loads(sidecar.read_text())
        admit(sidecar.with_name(sidecar.name.replace('.vid.json', '.mp4')),
              item.get('sha256'), 'grok-dialogue', sidecar,
              dict(requestedSeconds=item.get('durationSeconds')), job_id=job_id)
    inventoried = {c['path'] for row in rows for c in row['candidates']}
    unreceipted = [str(p.relative_to(root))
                  for p in sorted(root.glob('work/h3-*/silent/*.mp4'))
                  if str(p.relative_to(root)) not in inventoried]
    # Presence is useful for avoiding duplicate production, but a newly computed
    # hash is not proof of a prior GPU stream audit or silent-ingest contract.
    for relative in unreceipted:
        path = root / relative
        key = row_for(path.name, ids, by_shot)
        if key is not None:
            by_shot[key]['candidates'].append(dict(
                path=relative,
                kind='h3-continuation' if 'cont' in path.stem else 'h3-base',
                hashVerified=False, observedSha256=sha(path),
                evidence=None, audioStreams=None,
                creativeApproval='not established by this audit',
                verificationNeeded='GPU stream and picture-payload comparison against retained raw source'))
    missing = []
    absent = []
    for row in rows:
        heard_over_picture = row['speaker'] in ('Narrator', '-') or row.get('delivery') in ('unsynced', 'wordless')
        lane = 'h3-base' if heard_over_picture else 'grok-dialogue'
        row['requiredLane'] = lane
        row['hasVerifiedBaseCandidate'] = any(
            c['kind'] == lane and c['hashVerified'] for c in row['candidates'])
        if not row['hasVerifiedBaseCandidate']:
            missing.append(row['shot'])
        row['hasLocalBaseCandidate'] = any(c['kind'] == lane for c in row['candidates'])
        if not row['hasLocalBaseCandidate']:
            absent.append(row['shot'])
    summary = dict(scriptShots=len(rows),
                   shotsWithVerifiedBaseCandidate=len(rows) - len(missing),
                   shotsWithoutStandardReceiptBase=missing,
                   shotsWithLocalBaseCandidate=len(rows) - len(absent),
                   shotsWithoutLocalBaseCandidate=absent,
                   unreceiptedLocalH3=unreceipted,
                   caution='Do not regenerate from this list: reconcile legacy receipts and review existing files first.',
                   errors=errors)
    report = dict(episode=board.get('episode', root.name),scope='Local bytes and receipt audit only; not timing, lip-sync or creative approval.',
                  scriptSha256=sha(script), summary=summary, shots=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
