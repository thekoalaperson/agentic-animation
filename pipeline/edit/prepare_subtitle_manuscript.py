"""Compile reviewed translation edits without changing speech or running media tools."""
import argparse
import json
from pathlib import Path

from build_episode_timeline import _dialogue_subtitle_text, _production_overrides
from subtitle_manuscript import apply_manuscript, digest, text_digest


def prepare(episode, overrides_path, edits_path, output):
    episode = Path(episode).resolve()
    overrides_path, edits_path, output = [Path(p).resolve() for p in (overrides_path, edits_path, output)]
    for path in (overrides_path, edits_path, output):
        path.relative_to(episode)
    source = json.loads(overrides_path.read_text())
    edits = json.loads(edits_path.read_text())
    rows = _production_overrides(source)
    if set(edits) - {sid for sid, row in rows.items() if row.get('delivery') != 'wordless'}:
        raise ValueError('Unknown or wordless scene in translation edits')
    effective = {}
    items = []
    for sid, row in rows.items():
        if row.get('delivery') == 'wordless':
            continue  # picture only: no words, so nothing to translate or cue
        original = {'hi': row['narration_hi'], 'en': row['narration_en']}
        if row.get('speaker'):
            original = _dialogue_subtitle_text(original['hi'], original['en'], 'spoken-only')
        effective[sid] = original
        item = dict(id=sid, sourceTextSha256=text_digest(original['hi'], original['en']), **original)
        if sid in edits:
            edit = edits[sid]
            if set(edit) != {'en', 'editReason'} or not edit['editReason'].strip():
                raise ValueError('Translation edit requires English and reason only: '+sid)
            item.update(edit)
        items.append(item)
    doc = dict(schemaVersion=1, kind='subtitle-text-manuscript',
               reviewScope='Text translation review only; spoken accuracy, timing and rendered readability pending',
               sourceBindings=[dict(path=str(p.relative_to(episode)), sha256=digest(p))
                               for p in (overrides_path, edits_path)], items=items)
    payload = json.dumps(doc, ensure_ascii=False, indent=2)+'\n'
    if output.exists() and output.read_text() != payload:
        raise ValueError('Existing manuscript differs; choose a new version')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text(payload)
    result = apply_manuscript(episode, output, digest(output), effective)
    assert all(result[sid]['hi'] == effective[sid]['hi'] for sid in effective)
    print(json.dumps(dict(scenes=len(result), translationEdits=len(edits),
                          hindiChanged=0, manuscriptSha256=digest(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode')
    parser.add_argument('--overrides', required=True)
    parser.add_argument('--edits', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.episode, args.overrides, args.edits, args.output)
