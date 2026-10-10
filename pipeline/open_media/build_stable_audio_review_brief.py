"""Bind explicitly authored family prompts to the existing episode cue inventory."""
import argparse
import hashlib
import json
from pathlib import Path


def build(ep, root):
    families = {}
    sources = {}
    for directory in ['sfx-final-v1', 'sfx-audioldm2-v2', 'sfx-audioldm2-v1']:
        path = ep / 'work' / directory / 'manifest.json'
        sources[str(path.relative_to(ep))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for family in json.loads(path.read_text())['families']:
            families.setdefault(family['id'], family)
    authored_path = root / 'authored-prompts.json'
    authored = json.loads(authored_path.read_text())
    assert len(authored) == len({a['id'] for a in authored})
    assert set(families) == {a['id'] for a in authored}, 'Must cover the entire source inventory'
    jobs, review_families = [], []
    for i, a in enumerate(authored):
        f = families[a['id']]
        assert len(a['prompts']) == len(a['directions']) == 2
        review_families.append(dict(id=f['id'], title=f['title'], category=f['category'],
                                    shots=f['shots'], intent=a['intent'], directions=a['directions']))
        for take, prompt in enumerate(a['prompts'], 1):
            job_id = f"{f['id']}-sa3-t{take}"
            jobs.append(dict(id=job_id, family=f['id'], take=take, seconds=a['seconds'],
                             seed=429000+i*10+take, prompt=prompt, direction=a['directions'][take-1],
                             intendedRole=a['intent'], output=f'outputs/{job_id}.wav'))
    brief = dict(version=2, route='self-hosted-text-to-audio',
                 model='stabilityai/stable-audio-3-small-sfx', generationApiAllowed=False,
                 productionApproval=False,
                 authorization='Review-only probe set for the debug UI; no production approval, subjective listening or owner verdict claimed.',
                 sourceManifests=sources,
                 authoredPromptsSha256=hashlib.sha256(authored_path.read_bytes()).hexdigest(),
                 expectedJobs=len(jobs), expectedSeconds=sum(j['seconds'] for j in jobs),
                 families=review_families, jobs=jobs)
    target = root / 'probe-brief.json'
    encoded = json.dumps(brief, ensure_ascii=False, indent=2)+'\n'
    if target.exists():
        assert target.read_text() == encoded, 'Existing immutable brief differs; use a new batch directory'
    else:
        target.write_text(encoded)
    print(json.dumps(dict(jobs=len(jobs), seconds=brief['expectedSeconds'], families=len(families))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('episode', type=Path)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    build(a.episode.resolve(), a.root.resolve())
