#!/usr/bin/env python3
"""Run source-bound technical and advisory QA for one finite Stable Audio batch.

Each model is loaded sequentially. Reports never approve a sound or admit it to
an edit; the raw and scene-context review remains an explicit human decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--speech-model', required=True, help='Qualified cached Whisper model or path')
    parser.add_argument('--speech-python', default=sys.executable)
    parser.add_argument('--classifier-python', default=sys.executable)
    args = parser.parse_args()
    root = args.root.resolve()
    deadline = time.monotonic() + 3600
    if args.wait_pid:
        status = Path(f'/proc/{args.wait_pid}/stat')
        while status.exists():
            try:
                if status.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                    break
            except FileNotFoundError:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Generation observation deadline reached; do not restart it')
            time.sleep(5)
    scripts = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(scripts / 'verify_stable_audio_review.py'), str(root)], check=True)
    brief_path = root / 'probe-brief.json'
    brief = json.loads(brief_path.read_text())
    audit = json.loads((root / 'local-audit.json').read_text())
    hashes = {row['id']: row['sha256'] for row in audit['outputs']}
    manifest = {
        'sourceBriefSha256': hashlib.sha256(brief_path.read_bytes()).hexdigest(),
        'families': [{'id': job['id'], 'prompt': job['prompt']} for job in brief['jobs']],
        'jobs': [dict(job, family=job['id'], sha256=hashes[job['id']]) for job in brief['jobs']],
    }
    manifest_path = root / 'quality-manifest.json'
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    if manifest_path.exists() and manifest_path.read_text() != payload:
        raise ValueError('Existing quality manifest changed; preserve the previous batch')
    if not manifest_path.exists():
        manifest_path.write_text(payload)
    commands = [
        [args.speech_python, scripts.parent / 'audit_sfx_speech.py', root / 'outputs',
         '--model', args.speech_model, '--device', 'cuda', '--output', root / 'speech-contamination-audit.json'],
        [args.classifier_python, scripts / 'audit_sfx_audioset.py', root,
         '--manifest', manifest_path, '--output', root / 'audioset-audit.json'],
        [args.classifier_python, scripts / 'audit_sfx_clap.py', root,
         '--manifest', manifest_path, '--output', root / 'clap-audit.json'],
    ]
    for command in commands:
        print('START ' + Path(command[1]).name, flush=True)
        subprocess.run(list(map(str, command)), check=True)
    print('QUALITY_CHECKS_COMPLETE; creative/context review remains pending', flush=True)


if __name__ == '__main__':
    main()
