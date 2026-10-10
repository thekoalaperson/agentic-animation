"""Source-bound subtitle-only edits. Never mutates recorded speech or its clock."""
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digest(hi, en):
    return hashlib.sha256((hi+'\0'+en).encode()).hexdigest()


def apply_manuscript(episode, path, expected_sha, effective):
    episode, path = Path(episode).resolve(), Path(path).resolve()
    path.relative_to(episode)
    if not expected_sha or digest(path) != expected_sha:
        raise ValueError('subtitle-manuscript hash changed')
    doc = json.loads(path.read_text())
    if doc.get('schemaVersion') != 1 or doc.get('kind') != 'subtitle-text-manuscript':
        raise ValueError('invalid subtitle-manuscript schema')
    if not doc.get('sourceBindings'):
        raise ValueError('subtitle-manuscript lacks source bindings')
    for binding in doc['sourceBindings']:
        source = (episode/binding['path']).resolve()
        source.relative_to(episode)
        if digest(source) != binding['sha256']:
            raise ValueError('subtitle-manuscript source changed: '+binding['path'])
    rows = doc.get('items', [])
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(effective):
        raise ValueError('subtitle-manuscript must cover every scene exactly once')
    output = {}
    for row in rows:
        sid = row['id']; original = effective[sid]
        if row.get('sourceTextSha256') != text_digest(original['hi'], original['en']):
            raise ValueError('subtitle-manuscript text source changed: '+sid)
        if row.get('hi') != original['hi']:
            raise ValueError('subtitle-manuscript cannot rewrite Hindi speech: '+sid)
        if not isinstance(row.get('en'), str) or not row['en'].strip():
            raise ValueError('subtitle-manuscript has empty English: '+sid)
        if row['en'] != original['en'] and not row.get('editReason'):
            raise ValueError('subtitle translation edit lacks reason: '+sid)
        result = {'hi': row['hi'], 'en': row['en']}
        pairs = row.get('cuePairs')
        if pairs is not None:
            if not isinstance(pairs, list) or not pairs:
                raise ValueError('empty authored subtitle pairs: '+sid)
            for language in ('hi','en'):
                if any(not isinstance(p.get(language), str) or not p[language].strip() for p in pairs):
                    raise ValueError('empty authored subtitle clause: '+sid)
                if ' '.join(' '.join(p[language].split()) for p in pairs) != ' '.join(result[language].split()):
                    raise ValueError('subtitle clauses changed source words: '+sid)
            weights = row.get('cueWeights', [1]*len(pairs))
            if not isinstance(weights,list) or len(weights) != len(pairs) or any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0 for w in weights):
                raise ValueError('invalid authored subtitle weights: '+sid)
            result.update(cuePairs=pairs, cueWeights=weights)
        output[sid] = result
    return output
