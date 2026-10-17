"""Inventory local story coverage without decoding media or approving creativity."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--script', default='story/production-script-v2.tsv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.episode.resolve()
    script = root / args.script
    rows = []
    for line in script.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split('|')  # an optional fifth column carries delivery
        number, speaker, hindi, english = parts[:4]
        delivery = parts[4].strip() if len(parts) > 4 else ''
        rows.append(dict(shot=number, speaker=speaker, hindi=hindi, english=english,
                         **({'delivery': delivery} if delivery else {}), candidates=[]))
    board_path = root / 'story/storyboard_plan.json'
    board = json.loads(board_path.read_text()) if board_path.is_file() else {}
    for row in rows:
        index = int(row['shot']) - 1
        board_rows = board.get('stills', [])
        row['id'] = board_rows[index]['still_id'] if 0 <= index < len(board_rows) else row['shot']
    by_shot = {str(int(row['shot'])): row for row in rows}
    errors = []

    def admit(path, expected, kind, receipt, extra):
        match = re.search(r'-s(\d+)', path.name)
        if not match or str(int(match[1])) not in by_shot:
            errors.append(f'Unmapped asset: {path}')
            return
        valid = path.is_file() and bool(expected) and sha(path) == expected
        if not valid:
            errors.append(f'Missing or hash mismatch: {path}')
        by_shot[str(int(match[1]))]['candidates'].append(dict(
            path=str(path.relative_to(root)), kind=kind, hashVerified=valid,
            sha256=expected, evidence=str(receipt.relative_to(root)),
            creativeApproval='not established by this audit', **extra))

    for receipt in sorted(root.glob('work/h3-*/silent/ingest-receipt.json')):
        for item in json.loads(receipt.read_text())['items']:
            if item.get('audioStreams') != 0:
                errors.append(f'H3 audio policy violation: {receipt}: {item["id"]}')
            admit(receipt.parent / item['silent'], item.get('silentSha256'),
                  'h3-continuation' if 'cont' in item['id'] or 'continuation' in str(receipt) else 'h3-base',
                  receipt, dict(frames=item.get('frames'), audioStreams=item.get('audioStreams')))
    for sidecar in sorted(root.glob('work/dialogue-*/motion/*.vid.json')):
        item = json.loads(sidecar.read_text())
        admit(sidecar.with_name(sidecar.name.replace('.vid.json', '.mp4')),
              item.get('sha256'), 'grok-dialogue', sidecar,
              dict(requestedSeconds=item.get('durationSeconds')))
    inventoried = {c['path'] for row in rows for c in row['candidates']}
    unreceipted = [str(p.relative_to(root))
                  for p in sorted(root.glob('work/h3-*/silent/*.mp4'))
                  if str(p.relative_to(root)) not in inventoried]
    # Presence is useful for avoiding duplicate production, but a newly computed
    # hash is not proof of a prior GPU stream audit or silent-ingest contract.
    for relative in unreceipted:
        path = root / relative
        match = re.search(r'-s(\d+)', path.name)
        if match and str(int(match[1])) in by_shot:
            by_shot[str(int(match[1]))]['candidates'].append(dict(
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
