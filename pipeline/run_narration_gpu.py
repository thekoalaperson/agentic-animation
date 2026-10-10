#!/usr/bin/env python3
"""Run one already-authorized finite narration manifest on a qualified GPU host.

Credential arrives on stdin as JSON, lives in memory, and is never written to
the job folder or command line. API generation is external; response decoding
and media verification remain on the GPU worker. No retries or overage.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

from eleven_batch_finite import execute, load_plan, _atomic_json, _sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--max-characters', type=int, required=True)
    parser.add_argument('--approval-admission', type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('Run on the qualified GPU worker, not the Mac')
    subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], check=True)
    plan = load_plan(args.manifest, args.max_characters)
    doc = json.loads(args.manifest.read_text())
    authority = (args.manifest.parent / doc['authority']).resolve()
    if _sha256_file(authority) != doc['authoritySha256']:
        raise SystemExit('Production authority file changed')
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
    if available < plan['aggregateCharacters']:
        raise SystemExit('Insufficient existing credits; no overage authorized')
    if sub['tier'] not in ('creator', 'pro', 'scale', 'business', 'enterprise'):
        raise SystemExit('Subscription does not support the required audio format')
    receipt = dict(tier=sub['tier'], charactersBefore=sub['character_count'],
                   characterLimit=sub['character_limit'], availableBefore=available,
                   plannedCharacters=plan['aggregateCharacters'],
                   voiceId=voice['voice_id'], voiceName=voice['name'],
                   manifestSha256=plan['manifestSha256'], computeHost=platform.node(),
                   noOverage=True, noAutomaticRetry=True)
    _atomic_json(args.manifest.parent / 'credit-preflight.json', receipt)
    print(json.dumps(receipt), flush=True)
    result = execute(args.manifest, args.max_characters, True, api_key=key, approval_admission=args.approval_admission)
    after = get('user/subscription')
    receipt.update(charactersAfter=after['character_count'],
                   observedCharacterDelta=after['character_count'] - sub['character_count'],
                   result=result,
                   accountingNote='Shared account counter; delta is not per-job attribution if other tasks overlap')
    _atomic_json(args.manifest.parent / 'execution-receipt.json', receipt)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
