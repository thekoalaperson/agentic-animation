#!/usr/bin/env python3
"""Run one already-authorized finite narration manifest on a qualified GPU host.

Credential arrives on stdin as JSON, lives in memory, and is never written to
the job folder or command line. API generation is external; response decoding
and media verification remain on the GPU worker. No retries or overage.

Each invocation writes a new credit-preflight-attempt-NNN.json and
execution-receipt-attempt-NNN.json; execution-receipt.json lists every
attempt's before and after readings.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

from eleven_batch_finite import execute, load_plan, _atomic_json, _sha256_file
from _common import attempt_records, write_attempt_json


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
    credential = json.load(sys.stdin)
    key = credential['elevenlabsApiKey']
    def get(endpoint):
        req = urllib.request.Request('https://api.elevenlabs.io/v1/' + endpoint,
                                      headers={'xi-api-key': key})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    sub = get('user/subscription')
    voice = get('voices/' + plan['config']['voiceId'])
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
                   voiceId=voice['voice_id'], voiceName=voice['name'],
                   manifestSha256=plan['manifestSha256'], computeHost=platform.node(),
                   noOverage=True, noAutomaticRetry=True)
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
