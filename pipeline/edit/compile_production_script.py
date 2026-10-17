"""Compile authored paired scene text; never invent prose, timing or image approval.

Pipe-separated input: numeric scene ordinal | speaker (Narrator for VO) |
complete Hindi display text | authored English subtitle translation, with an
optional fifth column ``delivery``:

* ``synced`` -- the speaker is performed on camera (default for a speaker);
* ``unsynced`` -- the speaker's voice is heard over picture (voice-over or an
  off-screen line), recorded as ``voice`` and never as a lip-sync speaker;
* ``wordless`` -- picture only; written with the reserved speaker ``-`` and
  empty text columns.

``Narrator`` stays shorthand for unsynced narrator speech. The existing board
supplies only IDs/beat membership. Outputs are immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


PICTURE_ONLY = '-'
DELIVERIES = {'synced', 'unsynced', 'wordless'}


class ScriptError(ValueError):
    """An authored script row cannot be compiled as written."""


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


def _still_ordinal(still_id):
    match = re.search(r'-s(\d+)$', still_id)
    return int(match.group(1)) if match else None


def parse_row(index, line, still_id):
    """Return (speaker, voice, delivery, hindi, english, explicit) for one row."""
    parts = line.split('|')
    if len(parts) not in (4, 5):
        raise ScriptError(f'row {index}: expected four columns, or five with delivery; found {len(parts)}')
    ordinal, speaker, hindi, english = (v.strip() for v in parts[:4])
    delivery = parts[4].strip() if len(parts) == 5 else ''
    if not ordinal.isdigit() or int(ordinal) != index:
        raise ScriptError(f'row {index}: ordinal {ordinal!r} must be the row number {index}')
    if _still_ordinal(still_id) != int(ordinal):
        raise ScriptError(f'row {index}: ordinal {ordinal} does not match storyboard still {still_id}')
    if not speaker or re.search('[\t\n\r]', speaker):
        raise ScriptError(f'row {index} ({still_id}): speaker column is empty or contains a control character')
    if delivery and delivery not in DELIVERIES:
        raise ScriptError(f'row {index} ({still_id}): unknown delivery {delivery!r}; use synced, unsynced or wordless')
    explicit = len(parts) == 5 or speaker == PICTURE_ONLY
    if speaker == PICTURE_ONLY or delivery == 'wordless':
        if speaker != PICTURE_ONLY or delivery not in ('', 'wordless'):
            raise ScriptError(f'row {index} ({still_id}): a wordless row uses the speaker "-" and no other delivery')
        if hindi or english:
            raise ScriptError(f'row {index} ({still_id}): a wordless row must leave both text columns empty')
        return None, None, 'wordless', '', '', explicit
    if not hindi or not english:
        raise ScriptError(f'row {index} ({still_id}): both authored text columns must be non-empty')
    if speaker == 'Narrator':
        if delivery == 'synced':
            raise ScriptError(f'row {index} ({still_id}): Narrator is unsynced speech; it has no on-camera take')
        return None, None, 'unsynced', hindi, english, explicit
    if delivery == 'unsynced':
        return None, speaker, 'unsynced', hindi, english, explicit
    return speaker, None, 'synced', hindi, english, explicit


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode',type=Path)
    ap.add_argument('script',type=Path)
    ap.add_argument('output',type=Path)
    a=ap.parse_args();ep=a.episode.resolve();script=a.script.resolve();out=a.output.resolve()
    script.relative_to(ep/'story');out.relative_to(ep)
    board_path=ep/'story/storyboard_plan.json';board=json.loads(board_path.read_text())
    source=board['stills'];lines=script.read_text().splitlines()
    if len(lines)!=len(source):
        raise ScriptError(f'Every original slot must have an explicit authored decision: '
                          f'{len(lines)} script rows for {len(source)} storyboard stills')
    overrides={};changes=[];hi=[];en=[];narration_chars=0
    wordless=0;voice_over=0;marked=False
    for index,(line,old) in enumerate(zip(lines,source),1):
        key=old['still_id']
        role,voice,delivery,hindi,english,explicit=parse_row(index,line,key)
        marked=marked or explicit
        if not role and delivery=='unsynced':
            narration_chars+=len(hindi)
        item={'speaker':role,'narration_hi':hindi,'narration_en':english,
              'reason':'Authored production rewrite; literal supplied text retained. No inherited image approval.'}
        # Older four-column scripts keep byte-identical outputs: only the new
        # markers add fields, and only where they say something new.
        if delivery=='wordless':
            item['delivery']='wordless';wordless+=1
        elif voice:
            item.update(delivery='unsynced',voice=voice);voice_over+=1
        overrides[key]=item
        if role!=old.get('speaker'):changes.append({'id':key,'before':old.get('speaker'),'after':role})
        if delivery!='wordless':
            ordinal=line.split('|',1)[0].strip()
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
    if marked:
        summary.update(wordlessCount=wordless,voiceOverCount=voice_over)
    save(out/'audit.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
