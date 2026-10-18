#!/usr/bin/env python3
"""Run one already-authorized finite narration manifest on a qualified GPU host.

Credential arrives on stdin as JSON, lives in memory, and is never written to
the job folder or command line. API generation is external; response decoding
and media verification remain on the GPU worker. No retries or overage.

Each invocation writes a new credit-preflight-attempt-NNN.json and
execution-receipt-attempt-NNN.json; execution-receipt.json lists every
attempt's before and after readings. A full execution-receipt.json or
credit-preflight.json left by the older single-file wrapper is first kept as an
attempt marked legacyRecord.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

from eleven_batch_finite import execute, load_plan, model_cautions, _atomic_json, _sha256_file
from _common import adopt_legacy_record, attempt_records, write_attempt_json


def pending_characters(plan, state_path):
    """Characters a resumed batch can still POST: items not yet past PLANNED."""
    if not state_path.exists():
        return plan['aggregateCharacters']
    items = json.loads(state_path.read_text(encoding='utf-8')).get('items', {})
    return sum(item['characters'] for item in plan['items']
               if items.get(item['id'], {}).get('status', 'PLANNED') == 'PLANNED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--max-characters', type=int, required=True)
    parser.add_argument('--approval-admission', type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('Run on the qualified GPU worker, not the Mac')
    subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], check=True)
    # load_plan verifies the authority binding, so a dry run already sees a change.
    plan = load_plan(args.manifest, args.max_characters)
    if plan['authority'] is None:
        raise SystemExit('This wrapper requires an authority and authoritySha256 binding')
    folder = args.manifest.resolve().parent
    state_path = args.manifest.resolve().with_suffix('.eleven-state.json')
    # A folder from the single-file wrapper keeps its old receipt as attempt evidence.
    for stem in ('execution-receipt', 'credit-preflight'):
        adopt_legacy_record(folder, stem)
    credential = json.load(sys.stdin)
    key = credential['elevenlabsApiKey']
    def get(endpoint):
        req = urllib.request.Request('https://api.elevenlabs.io/v1/' + endpoint,
                                      headers={'xi-api-key': key})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    sub = get('user/subscription')
    # Every voice the batch will use is fetched and recorded before any POST.
    voices = []
    for row in plan['voices']:
        found = get('voices/' + row['voiceId'])
        voices.append(dict(speaker=row['speaker'], voiceId=found['voice_id'],
                           voiceName=found['name'], modelId=row['modelId']))
    available = sub['character_limit'] - sub['character_count']
    pending = pending_characters(plan, state_path)
    if available < pending:
        raise SystemExit('Insufficient existing credits; no overage authorized')
    if sub['tier'] not in ('creator', 'pro', 'scale', 'business', 'enterprise'):
        raise SystemExit('Subscription does not support the required audio format')
    receipt = dict(tier=sub['tier'], charactersBefore=sub['character_count'],
                   characterLimit=sub['character_limit'], availableBefore=available,
                   plannedCharacters=plan['aggregateCharacters'],
                   pendingCharacters=pending,
                   manifestSha256=plan['manifestSha256'], computeHost=platform.node(),
                   noOverage=True, noAutomaticRetry=True)
    if plan['version'] == 2:
        # A cast has no single voice, so a version 2 receipt lists every voice
        # and names the models whose characters bound credits only by assumption.
        receipt['voices'] = voices
        receipt.update(model_cautions(plan))
    else:
        receipt.update(voiceId=voices[0]['voiceId'], voiceName=voices[0]['voiceName'])
    write_attempt_json(folder, 'credit-preflight', receipt)
    print(json.dumps(receipt), flush=True)
    error = None
    try:
        result = execute(args.manifest, args.max_characters, True, api_key=key,
                         approval_admission=args.approval_admission)
    except Exception as exc:  # the after reading is evidence for the next attempt
        error, result = exc, {'status': 'STOPPED', 'error': f'{type(exc).__name__}: {exc}'}
    after = get('user/subscription')
    receipt.update(charactersAfter=after['character_count'],
                   observedCharacterDelta=after['character_count'] - sub['character_count'],
                   result=result,
                   accountingNote='Shared account counter; delta is not per-job attribution if other tasks overlap')
    write_attempt_json(folder, 'execution-receipt', receipt)
    _atomic_json(folder / 'execution-receipt.json', {
        'manifestSha256': plan['manifestSha256'],
        'attempts': attempt_records(folder, 'execution-receipt'),
        'preflights': attempt_records(folder, 'credit-preflight'),
    })
    print(json.dumps(result), flush=True)
    if error is not None:
        raise error


if __name__ == '__main__':
    main()
