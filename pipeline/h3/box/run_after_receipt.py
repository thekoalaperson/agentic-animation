"""Chain finite batches using their durable, manifest-bound completion receipt."""
import argparse
import json
import os
from pathlib import Path
import time


def ready(path, manifest_sha, expected):
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if state['manifestSha256'] != manifest_sha:
        raise RuntimeError('Previous manifest binding mismatch')
    if state.get('stopReason'):
        raise RuntimeError('Previous batch stopped; reconcile before chaining')
    if not state.get('finishedAt'):
        return False
    jobs = state['jobs']
    if len(jobs) != expected or state.get('completeCount') != expected:
        raise RuntimeError('Previous batch did not complete all expected jobs')
    if not all(j.get('status') == 'complete' for j in jobs.values()):
        raise RuntimeError('Previous batch contains incomplete jobs')
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-state', type=Path, required=True)
    p.add_argument('--previous-manifest-sha', required=True)
    p.add_argument('--previous-count', type=int, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    deadline = time.monotonic() + 3600
    while not ready(a.previous_state, a.previous_manifest_sha, a.previous_count):
        if time.monotonic() >= deadline:
            raise SystemExit('Queue deadline reached; no new work submitted')
        time.sleep(5)
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        raise SystemExit('Missing finite client command')
    print('Previous durable receipt proves completion; starting next finite batch', flush=True)
    os.execv(command[0], command)
