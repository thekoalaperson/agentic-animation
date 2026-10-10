"""Compile authored paired scene text; never invent prose, timing or image approval.

Pipe-separated input: numeric scene ordinal | speaker (Narrator for VO) |
complete Hindi display text | authored English subtitle translation.
The existing board supplies only IDs/beat membership. Outputs are immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Preserve existing output; choose a new version: '+str(path))
    else:
        with path.open('x') as f:
            f.write(text)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode',type=Path)
    ap.add_argument('script',type=Path)
    ap.add_argument('output',type=Path)
    a=ap.parse_args();ep=a.episode.resolve();script=a.script.resolve();out=a.output.resolve()
    script.relative_to(ep/'story');out.relative_to(ep)
    board_path=ep/'story/storyboard_plan.json';board=json.loads(board_path.read_text())
    source=board['stills'];lines=script.read_text().splitlines()
    assert len(lines)==len(source),'Every original slot must have an explicit authored decision'
    overrides={};changes=[];hi=[];en=[];narration_chars=0
    for index,(line,old) in enumerate(zip(lines,source),1):
        parts=line.split('|');assert len(parts)==4,(index,'expected four columns')
        ordinal,speaker,hindi,english=(v.strip() for v in parts)
        assert ordinal.isdigit() and int(ordinal)==index and old['still_id'].endswith('-s'+ordinal)
        assert hindi and english, 'Both authored text columns must be non-empty'
        assert speaker and not re.search('[\t\n\r]',speaker)
        role=None if speaker=='Narrator' else speaker
        if not role:
            narration_chars+=len(hindi)
        key=old['still_id']
        overrides[key]={'speaker':role,'narration_hi':hindi,'narration_en':english,
                        'reason':'Authored production rewrite; literal supplied text retained. No inherited image approval.'}
        if role!=old.get('speaker'):changes.append({'id':key,'before':old.get('speaker'),'after':role})
        hi.append(f'{ordinal}. {hindi}');en.append(f'{ordinal}. {english}')
    payload={'episode':board['episode'],'scriptSource':str(script.relative_to(ep)),
             'scriptSha256':digest(script),'boardSha256':digest(board_path),
             'originalsPreserved':True,'items':overrides}
    save(out/'overrides.json',json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for lang,rows in [('hi',hi),('en',en)]:
        save(out/f'performed-{lang}.md','\n\n'.join(rows)+'\n')
    summary={'episode':board['episode'],'sceneCount':len(source),'dialogueCount':sum(bool(x['speaker']) for x in overrides.values()),
             'narrationCharacters':narration_chars,'roleChanges':changes,'sourceSha256':digest(script),
             'overridesSha256':digest(out/'overrides.json'),'mediaGenerationPerformed':False,'visualApprovalClaimed':False}
    save(out/'audit.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
