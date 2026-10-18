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
supplies only IDs/beat membership, plus an optional ``passage`` on a row that
continues the row before it: the two halves of one split speech, each carrying
only its own words, then count as one utterance. ``--storyboard`` selects an
alternate cut of the same project. Outputs are immutable.

Optional reports, never gates:

* ``--source-script`` (a Fountain file, plain text, or a source-units file from
  ``read_screenplay.py``) adds ``authoredBeyondSource`` to ``audit.json``: the
  performed lines not contained in the source after NFC, punctuation, danda
  and spacing normalisation, so split halves of one speech are not flagged.
* ``--speech-density`` writes ``speech-density.json``: authored characters per
  row and any requested picture seconds (an authored ``holdSeconds`` or
  ``holdFrames``, or ``--picture-seconds``). Only with an operator-stated
  ``--characters-per-second`` does it add ``estimatedSpeechSeconds``, an
  estimate that measured audio replaces; there is no default rate. With
  ``--clip-cap`` it flags on-camera rows whose estimate passes the cap. Where
  to split a line, if at all, stays the operator's choice.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from read_screenplay import contains, read_units


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


def source_text(path):
    """Return the comparable text of a source: units file, Fountain or plain text."""
    raw = path.read_text(encoding='utf-8-sig')
    if path.suffix == '.json':
        doc = json.loads(raw)
        if doc.get('kind') != 'screenplay-source-units':
            raise ScriptError(f'{path}: a JSON source must be a source-units file from read_screenplay.py')
        units = doc['units']
    elif path.suffix == '.fountain':
        units = read_units(raw)[1]
    else:
        return raw
    # Cues and parentheticals are not spoken; notes and boneyard are not the script.
    return '\n'.join(u['text'] for u in units if u['kind'] not in ('note', 'boneyard'))


def _picture_seconds(row, default):
    if row.get('holdSeconds') is not None:
        return float(row['holdSeconds'])
    if row.get('holdFrames') is not None:
        return int(row['holdFrames']) / 24
    return default


def speech_density(board, overrides, rate, cap, default_picture):
    rows = []
    for row in board['stills']:
        item = overrides[row['still_id']]
        delivery = item.get('delivery') or ('synced' if item['speaker'] else 'unsynced')
        entry = {'id': row['still_id'], 'delivery': delivery, 'characters': len(item['narration_hi'])}
        picture = _picture_seconds(row, default_picture)
        if picture is not None:
            entry['pictureSeconds'] = round(picture, 6)
        if rate is not None and entry['characters']:
            estimate = entry['characters'] / rate
            entry['estimatedSpeechSeconds'] = round(estimate, 3)
            if picture is not None and estimate > picture:
                entry['estimateExceedsPicture'] = True
            if cap is not None and delivery == 'synced' and estimate > cap:
                entry['estimateExceedsClipCap'] = True
        rows.append(entry)
    doc = {'episode': board['episode'], 'kind': 'speech-density-estimate',
           'basis': ('estimate at the operator-stated rate; measured audio replaces it'
                     if rate is not None else 'character counts only; no rate was stated'),
           'charactersPerSecond': rate, 'rows': rows}
    if cap is not None:
        doc['clipCapSeconds'] = cap
    return doc


def _positive(value):
    number = float(value)
    if not number > 0 or number != number or number == float('inf'):
        raise argparse.ArgumentTypeError('must be a positive number')
    return number


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode',type=Path)
    ap.add_argument('script',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--storyboard',type=Path,default=Path('story/storyboard_plan.json'),
                    help='Storyboard inside the episode; default story/storyboard_plan.json')
    ap.add_argument('--source-script',type=Path,
                    help='Report performed lines not contained in this source (Fountain, text or source units)')
    ap.add_argument('--speech-density',action='store_true',help='Write speech-density.json')
    ap.add_argument('--characters-per-second',type=_positive,
                    help='Operator-stated speaking rate for the estimate; there is no default')
    ap.add_argument('--clip-cap',type=_positive,help='Flag on-camera rows whose estimate passes this many seconds')
    ap.add_argument('--picture-seconds',type=_positive,
                    help='Requested picture seconds for rows with no authored hold')
    a=ap.parse_args();ep=a.episode.resolve();script=a.script.resolve();out=a.output.resolve()
    script.relative_to(ep/'story');out.relative_to(ep)
    board_path=(ep/a.storyboard).resolve();board_path.relative_to(ep)
    alternate_board=board_path!=(ep/'story/storyboard_plan.json').resolve()
    board=json.loads(board_path.read_text())
    source=board['stills'];lines=script.read_text().splitlines()
    if len(lines)!=len(source):
        raise ScriptError(f'Every original slot must have an explicit authored decision: '
                          f'{len(lines)} script rows for {len(source)} storyboard stills')
    overrides={};changes=[];hi=[];en=[];narration_chars=0
    wordless=0;voice_over=0;marked=False;passages=[]
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
        if old.get('passage') is not None:
            previous=source[index-2] if index>1 else None
            prior=overrides.get(previous['still_id']) if previous else None
            if (prior is None or old['passage']!=previous['still_id'] or delivery=='wordless'
                    or prior.get('delivery')=='wordless' or prior['speaker']!=role
                    or prior.get('voice')!=voice or prior.get('delivery')!=item.get('delivery')):
                raise ScriptError(f'row {index} ({key}): passage must name the row directly before it, '
                                  'spoken by the same speaker with the same delivery')
            item['passage']=old['passage'];passages.append([old['passage'],key])
        overrides[key]=item
        if role!=old.get('speaker'):changes.append({'id':key,'before':old.get('speaker'),'after':role})
        if delivery!='wordless':
            ordinal=line.split('|',1)[0].strip()
            hi.append(f'{ordinal}. {hindi}');en.append(f'{ordinal}. {english}')
    payload={'episode':board['episode'],'scriptSource':str(script.relative_to(ep)),
             'scriptSha256':digest(script),'boardSha256':digest(board_path),
             **({'storyboard':str(board_path.relative_to(ep))} if alternate_board else {}),
             'originalsPreserved':True,'items':overrides}
    save(out/'overrides.json',json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for lang,rows in [('hi',hi),('en',en)]:
        save(out/f'performed-{lang}.md','\n\n'.join(rows)+'\n')
    # The second half of a split passage continues an utterance; it is not a new one.
    summary={'episode':board['episode'],'sceneCount':len(source),
             'dialogueCount':sum(bool(x['speaker']) and not x.get('passage') for x in overrides.values()),
             'narrationCharacters':narration_chars,'roleChanges':changes,'sourceSha256':digest(script),
             'overridesSha256':digest(out/'overrides.json'),'mediaGenerationPerformed':False,'visualApprovalClaimed':False}
    if marked:
        summary.update(wordlessCount=wordless,voiceOverCount=voice_over)
    if passages:
        summary['passages']=passages
    if a.source_script:
        source_path=a.source_script.resolve()
        text=source_text(source_path)
        try:
            shown=str(source_path.relative_to(ep))
        except ValueError:
            shown=str(source_path)
        summary['authoredBeyondSource']={
            'source':shown,'sourceSha256':digest(source_path),
            'comparison':'containment after NFC, punctuation, danda and spacing normalisation; a report, not a gate',
            'rows':[{'id':key,'text':item['narration_hi']} for key,item in overrides.items()
                    if item['narration_hi'] and not contains(text,item['narration_hi'])]}
    save(out/'audit.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if a.speech_density or a.characters_per_second or a.clip_cap or a.picture_seconds:
        density=speech_density(board,overrides,a.characters_per_second,a.clip_cap,a.picture_seconds)
        save(out/'speech-density.json',json.dumps(density,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
