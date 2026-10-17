"""Read a Fountain screenplay into source units; never choose shots.

Read mode writes an immutable ``story/source-units-vN.json`` and a readable
``source-units-vN.md``. Each unit (heading, action, dialogue block with its
cue, extensions and parentheticals, transition, section, synopsis, lyric, note,
boneyard) gets an ID made of its scene ordinal and element ordinal, for example
``s002.e003``; elements before the first heading belong to scene 0. The text
hash is kept beside the ID, and the screenplay's SHA-256 is recorded. Dual
dialogue (a cue ending in ``^``) is kept as a pair. Notes and bracketed lines
such as ``[TIME ...]`` are flagged, never interpreted. Lines that look like a
character cue but are not one under Fountain's rules (a cue is Latin upper
case or forced with ``@``) are listed, never guessed.

Reading a revised screenplay writes the next version plus a diff (renumbered,
moved, changed, added, removed) against the latest earlier version. Nothing is
rewritten in place. The reader emits no shots, stills or durations: rows cite
units through an optional storyboard ``source_units`` list, in whatever
segmentation the director chooses.

``--units`` reports how the storyboard covers the units: per unit, whether its
words are spoken on camera, heard over picture, pictured without words, or not
used, whether the text is verbatim or changed, and whether it was split,
merged with another unit or reordered. It also lists spoken text found in no
unit, rows citing no unit, subtitles that carry words their frozen take does
not speak, and the exact line of every frozen dialogue job and prompt against
its unit. Comparison is after Unicode NFC with punctuation, danda and spacing
normalised; nukta and ZWJ/ZWNJ differences are reported on their own, never as
changes. The report holds lists, not scores, and nothing gates on it.

``--episode-range FIRST LAST`` limits the report to one episode's span of units
and records the span with the units file's hash.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata


class ScreenplayError(ValueError):
    """The screenplay or a cited unit cannot be read as written."""


HEADING = re.compile(r'^(INT|EXT|EST|INT\./EXT|INT/EXT|I/E)[. ]', re.IGNORECASE)
SCENE_NUMBER = re.compile(r'\s+#([A-Za-z0-9.\-]+)#\s*$')
TITLE_KEY = re.compile(r'^([A-Za-z][A-Za-z ]*):(.*)$')
EXTENSION = re.compile(r'\s*\(([^()]*)\)\s*$')
NOTE = re.compile(r'\[\[(.*?)\]\]', re.DOTALL)
BONEYARD = re.compile(r'/\*(.*?)\*/', re.DOTALL)
BRACKET_LINE = re.compile(r'^\[(?!\[).*\](?<!\]\])$')
SENTENCE_END = ('.', '!', '?', ',', ';', ':', '।', '॥')
NUKTA = '़'
JOINERS = ('‌', '‍')


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def text_sha(text):
    return sha256_bytes(text.encode('utf-8'))


def normalize(text, loose=False):
    """Compare text, not typography.

    NFC first; Fountain notes are dropped; every punctuation character
    (danda included) becomes a space; Latin case folds; spacing collapses.
    ``loose`` also drops nukta and ZWJ/ZWNJ, so a caller can tell a spelling
    variant from a change.
    """
    text = NOTE.sub(' ', unicodedata.normalize('NFC', text or '')).replace('﻿', '')
    if loose:
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if c != NUKTA and c not in JOINERS)
        text = unicodedata.normalize('NFC', text)
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return ' '.join(text.casefold().split())


def compare(expected, actual):
    """Return 'verbatim', 'nukta-or-joiner' or 'changed'."""
    if normalize(expected) == normalize(actual):
        return 'verbatim'
    if normalize(expected, loose=True) == normalize(actual, loose=True):
        return 'nukta-or-joiner'
    return 'changed'


def contains(haystack, needle, loose=False):
    needle = normalize(needle, loose)
    return bool(needle) and needle in normalize(haystack, loose)


# --------------------------------------------------------------------------
# Reading


def _strip_spans(lines, pattern, kind, elements):
    """Remove boneyard or note spans; record each as its own element.

    A line that held only the span is dropped, so a note inside a dialogue
    block does not split it. Inline notes stay in the text for the reader and
    are flagged on the element that contains them.
    """
    text = '\n'.join(line for _, line in lines)
    starts = []
    offset = 0
    for number, line in lines:
        starts.append((offset, number))
        offset += len(line) + 1

    def line_at(position):
        number = lines[0][0] if lines else 1
        for start, n in starts:
            if start > position:
                break
            number = n
        return number

    keep = []
    removed = set()
    for match in pattern.finditer(text):
        first, last = line_at(match.start()), line_at(match.end() - 1)
        before = text[text.rfind('\n', 0, match.start()) + 1:match.start()]
        after_end = text.find('\n', match.end())
        after = text[match.end():after_end if after_end != -1 else len(text)]
        whole = not before.strip() and not after.strip()
        if kind == 'note' and not whole:
            continue  # inline note: stays in the text and is flagged later
        elements.append(dict(kind=kind, line=first, endLine=last, text=match.group(1).strip()))
        if whole:
            removed.update(range(first, last + 1))
        else:
            keep.append((match.start(), match.end()))
    if keep:
        pieces, cursor = [], 0
        for start, end in keep:
            pieces.append(text[cursor:start])
            pieces.append('\n' * text.count('\n', start, end))
            cursor = end
        pieces.append(text[cursor:])
        text = ''.join(pieces)
    rebuilt = text.split('\n')
    return [(number, rebuilt[i]) for i, (number, _) in enumerate(lines) if number not in removed]


def _title_page(lines):
    """Split a leading title page (key: value lines) from the body."""
    first = next((i for i, (_, line) in enumerate(lines) if line.strip()), None)
    if first is None or not TITLE_KEY.match(lines[first][1].strip()):
        return {}, lines
    page, key, end = {}, None, first
    for i in range(first, len(lines)):
        raw = lines[i][1]
        if not raw.strip():
            end = i
            break
        match = TITLE_KEY.match(raw.strip())
        if match and not raw[:1].isspace():
            key = match.group(1).strip()
            page[key] = match.group(2).strip()
        elif key:
            page[key] = (page[key] + '\n' + raw.strip()).strip()
        end = i + 1
    return page, lines[end:]


def _blocks(lines):
    block = []
    for number, raw in lines:
        line = raw.rstrip('\r')
        if line.strip() or (line == '  ' and block):
            block.append((number, line))
        elif block:
            yield block
            block = []
    if block:
        yield block


def _is_cue(line):
    """Return (name, extensions, dual, forced) when a line is a character cue."""
    text = line.strip()
    forced = text.startswith('@')
    if forced:
        text = text[1:].strip()
    dual = text.endswith('^')
    if dual:
        text = text[:-1].rstrip()
    extensions = []
    while True:
        match = EXTENSION.search(text)
        if not match or match.start() == 0:
            break
        extensions.insert(0, match.group(1).strip())
        text = text[:match.start()].rstrip()
    if not text:
        return None
    if not forced:
        has_case = any(c.isalpha() and c.upper() != c.lower() for c in text)
        if not has_case or text != text.upper() or not text[0].isalpha() or text.endswith('TO:'):
            return None
    return text, extensions, dual, forced


def _transition(line):
    text = line.strip()
    if text.startswith('>') and not text.endswith('<'):
        return text[1:].strip()
    if text.endswith('TO:') and text == text.upper() and any(c.isalpha() for c in text):
        return text
    return None


def _heading(line):
    text = line.strip()
    if text.startswith('.') and not text.startswith('..'):
        text = text[1:].strip()
    elif not HEADING.match(text):
        return None
    number = SCENE_NUMBER.search(text)
    return (text[:number.start()].rstrip(), number.group(1)) if number else (text, None)


def _block_elements(block, elements, uncertain):
    first_number, first = block[0]
    stripped = first.strip()
    if re.fullmatch(r'={3,}', stripped):
        return  # page break
    if stripped.startswith('#') or (stripped.startswith('=') and not stripped.startswith('==')):
        for number, line in block:
            text = line.strip()
            if text.startswith('#'):
                level = len(text) - len(text.lstrip('#'))
                elements.append(dict(kind='section', line=number, endLine=number,
                                     level=level, text=text[level:].strip()))
            elif text.startswith('='):
                elements.append(dict(kind='synopsis', line=number, endLine=number, text=text[1:].strip()))
            else:
                elements.append(dict(kind='action', line=number, endLine=number, text=text))
        return
    last_number = block[-1][0]
    if len(block) > 1 and _heading(first):
        # A heading run straight into its action: read the heading, then the rest.
        _block_elements(block[:1], elements, uncertain)
        _block_elements(block[1:], elements, uncertain)
        return
    if len(block) == 1:
        heading = _heading(first)
        if heading:
            unit = dict(kind='heading', line=first_number, endLine=first_number, text=heading[0])
            if heading[1]:
                unit['sceneNumber'] = heading[1]
            elements.append(unit)
            return
        transition = _transition(first)
        if transition:
            elements.append(dict(kind='transition', line=first_number, endLine=first_number, text=transition))
            return
    if all(line.strip().startswith('~') for _, line in block):
        elements.append(dict(kind='lyric', line=first_number, endLine=last_number,
                             text='\n'.join(line.strip()[1:].strip() for _, line in block)))
        return
    cue = _is_cue(first) if len(block) > 1 and not stripped.startswith('!') else None
    if cue:
        name, extensions, dual, forced = cue
        lines = []
        for number, line in block[1:]:
            text = line.strip()
            if text.startswith('(') and text.endswith(')'):
                lines.append(dict(kind='parenthetical', line=number, text=text))
            elif text.startswith('~'):
                lines.append(dict(kind='lyric', line=number, text=text[1:].strip()))
            else:
                lines.append(dict(kind='dialogue', line=number, text=text))
        unit = dict(kind='dialogue', line=first_number, endLine=last_number, character=name,
                    text='\n'.join(row['text'] for row in lines if row['kind'] != 'parenthetical'),
                    lines=lines)
        if extensions:
            unit['extensions'] = extensions
        if forced:
            unit['forcedCue'] = True
        if dual:
            unit['dualCue'] = True
        elements.append(unit)
        return
    text_lines = []
    for number, line in block:
        text = line.strip()
        if number == first_number and text.startswith('!'):
            text = text[1:].strip()
        text_lines.append(text)
    unit = dict(kind='action', line=first_number, endLine=last_number, text='\n'.join(text_lines))
    if stripped.startswith('>') and stripped.endswith('<'):
        unit.update(text=stripped[1:-1].strip(), centered=True)
    brackets = [number for number, line in block if BRACKET_LINE.match(line.strip())]
    if brackets:
        unit['bracketLines'] = brackets
    if (len(block) > 1 and not stripped.startswith('!') and len(stripped) <= 40
            and not stripped.endswith(SENTENCE_END) and not BRACKET_LINE.match(stripped)):
        uncertain.append(dict(line=first_number, text=stripped,
                              reason='followed by text like a cue, but not upper-case Latin or @-forced'))
    elements.append(unit)


def read_units(raw_text):
    """Parse Fountain text; return (title page, units, flags)."""
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    lines = list(enumerate(text.split('\n'), 1))
    elements = []
    lines = _strip_spans(lines, BONEYARD, 'boneyard', elements)
    lines = _strip_spans(lines, NOTE, 'note', elements)
    title, lines = _title_page(lines)
    uncertain = []
    for block in _blocks(lines):
        _block_elements(block, elements, uncertain)
    elements.sort(key=lambda unit: unit['line'])
    scene, element, units = 0, 0, []
    for unit in elements:
        if unit['kind'] == 'heading':
            scene, element = scene + 1, 0
        element += 1
        unit_id = f's{scene:03d}.e{element:03d}'
        units.append(dict(id=unit_id, scene=scene, element=element, **unit, textSha256=text_sha(unit['text'])))
    for index, unit in enumerate(units):
        if unit.pop('dualCue', False):
            previous = units[index - 1] if index else None
            if not previous or previous['kind'] != 'dialogue' or 'dual' in previous:
                uncertain.append(dict(line=unit['line'], text=unit['character'],
                                      reason='dual-dialogue cue (^) with no dialogue block directly before it'))
                continue
            previous['dual'] = dict(side='left', pair=unit['id'])
            unit['dual'] = dict(side='right', pair=previous['id'])
        if unit['kind'] not in ('note', 'boneyard') and NOTE.search(unit['text']):
            unit['inlineNotes'] = [m.group(1).strip() for m in NOTE.finditer(unit['text'])]
    flags = dict(
        notes=[u['id'] for u in units if u['kind'] == 'note' or 'inlineNotes' in u],
        bracketLines=[u['id'] for u in units if 'bracketLines' in u],
        boneyard=[u['id'] for u in units if u['kind'] == 'boneyard'],
        uncertainCues=uncertain)
    return title, units, flags


def diff_units(old, new):
    """Describe a revision: renumbered, moved, changed, added and removed units.

    Unchanged runs are found by kind and text hash; a unit whose text reappears
    elsewhere has moved; within an edited stretch, units of the same kind pair
    up in order as changed; whatever is left was added or removed.
    """
    key = lambda u: (u['kind'], u['textSha256'])
    matcher = difflib.SequenceMatcher(None, [key(u) for u in old], [key(u) for u in new], autojunk=False)
    opcodes = matcher.get_opcodes()
    renumbered, matched_old, matched_new = [], set(), set()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                matched_old.add(i)
                matched_new.add(j)
                if old[i]['id'] != new[j]['id']:
                    renumbered.append(dict(old=old[i]['id'], new=new[j]['id']))
    moved = []
    for j, unit in enumerate(new):
        if j in matched_new:
            continue
        for i, previous in enumerate(old):
            if i not in matched_old and key(previous) == key(unit):
                matched_old.add(i)
                matched_new.add(j)
                moved.append(dict(old=previous['id'], new=unit['id'], kind=unit['kind']))
                break
    changed = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'replace':
            continue
        left = [i for i in range(i1, i2) if i not in matched_old]
        for j in (j for j in range(j1, j2) if j not in matched_new):
            pair = next((i for i in left if old[i]['kind'] == new[j]['kind']), None)
            if pair is None:
                continue
            left.remove(pair)
            matched_old.add(pair)
            matched_new.add(j)
            changed.append(dict(old=old[pair]['id'], new=new[j]['id'], kind=new[j]['kind'],
                                before=old[pair]['text'], after=new[j]['text']))
    added = [dict(id=u['id'], kind=u['kind'], text=u['text']) for j, u in enumerate(new) if j not in matched_new]
    removed = [dict(id=u['id'], kind=u['kind'], text=u['text']) for i, u in enumerate(old) if i not in matched_old]
    return dict(renumbered=renumbered, moved=moved, changed=changed, added=added, removed=removed)


def _relative(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _versions(directory):
    found = []
    for path in directory.glob('source-units-v*.json'):
        match = re.fullmatch(r'source-units-v(\d+)\.json', path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def _write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def render_markdown(doc):
    out = [f"# Source units v{doc['version']}", '',
           f"Read from `{doc['source']}` (SHA-256 `{doc['sourceSha256']}`). "
           'Units only: no shots, stills or durations.', '']
    for unit in doc['units']:
        if unit['kind'] == 'heading':
            out += ['', f"## {unit['id']} {unit['text']}", '']
            continue
        label = unit['kind']
        if unit['kind'] == 'dialogue':
            label = unit['character'] + ''.join(f' ({e})' for e in unit.get('extensions', []))
            if 'dual' in unit:
                label += f" [dual {unit['dual']['side']} with {unit['dual']['pair']}]"
            body = '\n'.join(('  ' + line['text']) for line in unit['lines'])
            out.append(f"- **{unit['id']}** {label}, line {unit['line']}\n{body}")
            continue
        flags = ' [bracketed line flagged]' if 'bracketLines' in unit else ''
        flags += ' [inline note flagged]' if 'inlineNotes' in unit else ''
        text = unit['text'].replace('\n', ' / ')
        out.append(f"- **{unit['id']}** {label}, line {unit['line']}{flags}: {text}")
    if doc['flags']['uncertainCues']:
        out += ['', '## Lines that look like cues but are not read as cues', '']
        out += [f"- line {u['line']}: {u['text']} ({u['reason']})" for u in doc['flags']['uncertainCues']]
    return '\n'.join(out) + '\n'


def read_mode(project, screenplay, output):
    raw = screenplay.read_bytes()
    title, units, flags = read_units(raw.decode('utf-8'))
    directory = project / 'story'
    versions = _versions(directory)
    previous = versions[-1][1] if versions else None
    if previous is not None:
        old = json.loads(previous.read_text(encoding='utf-8'))
        reader_sha = sha256_bytes(Path(__file__).read_bytes())
        if (old.get('sourceSha256') == sha256_bytes(raw) and old.get('readerSha256') == reader_sha
                and output is None):
            print(json.dumps(dict(unchanged=str(previous), sourceSha256=old['sourceSha256'])))
            return previous
    version = versions[-1][0] + 1 if versions else 1
    target = output or directory / f'source-units-v{version}.json'
    target = target.resolve()
    if not target.is_relative_to(project):
        raise ScreenplayError('source units must be written inside the project')
    if target.exists():
        raise ScreenplayError('Preserve existing source units; choose a new version: ' + str(target))
    doc = dict(kind='screenplay-source-units', schemaVersion=1, version=version,
               source=_relative(screenplay, project), sourceSha256=sha256_bytes(raw),
               reader='pipeline/edit/read_screenplay.py',
               readerSha256=sha256_bytes(Path(__file__).read_bytes()),
               scope='An inventory of the source; it chooses no shots, stills or durations.',
               previous=(dict(path=_relative(previous, project), sha256=sha256_bytes(previous.read_bytes()))
                         if previous else None),
               **({'titlePage': title} if title else {}), flags=flags, units=units)
    text = json.dumps(doc, ensure_ascii=False, indent=2) + '\n'
    markdown = target.with_suffix('.md')
    diff_path = target.with_name(target.stem + '.diff.json')
    for path in (markdown, diff_path):
        if path.exists():
            raise ScreenplayError('Preserve existing output; choose a new version: ' + str(path))
    _write_new(target, text)
    _write_new(markdown, render_markdown(doc))
    if previous is not None:
        diff = dict(kind='screenplay-source-units-diff', schemaVersion=1,
                    previous=doc['previous'], current=dict(path=_relative(target, project), sha256=text_sha(text)),
                    **diff_units(old['units'], units))
        _write_new(diff_path, json.dumps(diff, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(units=str(target), count=len(units), version=version)))
    return target


# --------------------------------------------------------------------------
# The --units report

SPOKEN_KINDS = ('dialogue', 'lyric')


def _row_kind(row, override):
    delivery = row.get('delivery') or override.get('delivery')
    if delivery == 'wordless':
        return 'picture'
    if row.get('speaker'):
        return 'dialogue'
    return 'narration'


def _cited(row):
    raw = row.get('source_units')
    if raw is None:
        return None
    if not isinstance(raw, list) or any(not isinstance(u, str) or not u for u in raw):
        raise ScreenplayError(f"{row.get('still_id')}: source_units must be a list of unit IDs")
    return raw


def _overrides(doc):
    raw = doc.get('items', doc.get('overrides', {}))
    if isinstance(raw, list):
        return {str(item['id']): item for item in raw}
    return dict(raw)


def _dialogue_jobs(project, manifests):
    jobs = {}
    for path in manifests:
        doc = json.loads(path.read_text(encoding='utf-8'))
        for job in doc.get('jobs', []):
            prompt = (project / job['prompt']) if job.get('prompt') else None  # project-relative, as the runner reads it
            if prompt is not None and not prompt.resolve().is_relative_to(project):
                raise ScreenplayError('dialogue prompt escapes the project: ' + str(prompt))
            jobs.setdefault(job['id'], []).append(dict(
                manifest=_relative(path, project), manifestSha256=sha256_bytes(path.read_bytes()),
                hindi=job.get('hindi', ''),
                prompt=prompt.read_text(encoding='utf-8') if prompt is not None and prompt.is_file() else None))
    return jobs


DEVANAGARI_RUN = re.compile(r'[ऀ-ॿ][ऀ-ॿ‌‍\s.,!?;:\'"‘’“”-]*')


def units_report(project, units_path, board_path, overrides_path=None, manifests=(), reasons=None,
                 episode_range=None):
    units_raw = units_path.read_bytes()
    units_doc = json.loads(units_raw)
    units = units_doc['units']
    index = {u['id']: n for n, u in enumerate(units)}
    by_id = {u['id']: u for u in units}
    board = json.loads(board_path.read_text(encoding='utf-8'))
    rows = board.get('stills', [])
    overrides = _overrides(json.loads(overrides_path.read_text(encoding='utf-8'))) if overrides_path else {}
    reasons = reasons or {}
    unknown_reasons = sorted(set(reasons) - set(by_id))
    if unknown_reasons:
        raise ScreenplayError('reasons name unknown units: ' + ', '.join(unknown_reasons))
    span = None
    if episode_range:
        first, last = episode_range
        if first not in index or last not in index or index[first] > index[last]:
            raise ScreenplayError('episode range must name two units in reading order')
        span = range(index[first], index[last] + 1)
    in_span = (lambda uid: index[uid] in span) if span else (lambda uid: True)

    effective = []
    unknown = []
    for position, row in enumerate(rows):
        still = str(row.get('still_id'))
        override = overrides.get(still, {})
        kind = _row_kind(row, override)
        text = '' if kind == 'picture' else str(override.get('narration_hi') or row.get('narration_hi') or '')
        english = '' if kind == 'picture' else str(override.get('narration_en') or row.get('narration_en') or '')
        cited = _cited(row)
        for uid in cited or []:
            if uid not in by_id:
                unknown.append(dict(still=still, unit=uid))
        effective.append(dict(position=position, still=still, kind=kind, text=text, english=english,
                              cited=[u for u in (cited or []) if u in by_id], citesNothing=cited is None,
                              speaker=row.get('speaker') or override.get('voice') or row.get('voice')))
    if unknown:
        raise ScreenplayError('storyboard cites unknown units: ' + json.dumps(unknown, ensure_ascii=False))

    def speaks(row, unit):
        if not row['text']:
            return False
        if unit['kind'] in SPOKEN_KINDS:
            return True
        return contains(unit['text'], row['text']) or contains(row['text'], unit['text'])

    report_units = []
    last_unit_position = -1
    reordered = set()
    for row in effective:
        for uid in row['cited']:
            if index[uid] < last_unit_position:
                reordered.add(uid)
            last_unit_position = max(last_unit_position, index[uid])
    for unit in units:
        if unit['kind'] in ('heading', 'section', 'synopsis', 'note', 'boneyard', 'transition'):
            continue
        if not in_span(unit['id']):
            continue
        citing = [r for r in effective if unit['id'] in r['cited']]
        speaking = [r for r in citing if speaks(r, unit)]
        entry = dict(id=unit['id'], kind=unit['kind'], rows=[r['still'] for r in citing])
        if unit['kind'] == 'dialogue':
            entry['character'] = unit['character']
        if not citing:
            entry['use'] = 'not used'
        elif not speaking:
            entry['use'] = 'pictured without words'
        else:
            kinds = {r['kind'] for r in speaking}
            entry['use'] = ('spoken on camera' if kinds == {'dialogue'} else
                            'heard over picture' if kinds == {'narration'} else 'spoken on camera and over picture')
            merged = [r for r in speaking if len([u for u in r['cited'] if speaks(r, by_id[u])]) > 1]
            if merged:
                entry['mergedWith'] = sorted({u for r in merged for u in r['cited'] if u != unit['id']})
                match = ('verbatim' if all(contains(r['text'], unit['text']) for r in merged) else
                         'nukta-or-joiner' if all(contains(r['text'], unit['text'], True) for r in merged)
                         else 'changed')
            else:
                match = compare(unit['text'], ' '.join(r['text'] for r in speaking))
            entry['text'] = match
            if len(speaking) > 1:
                entry['splitAcross'] = [r['still'] for r in speaking]
            if match != 'verbatim':
                entry['spoken'] = [dict(still=r['still'], text=r['text']) for r in speaking]
                entry['source'] = unit['text']
        if unit['id'] in reordered:
            entry['reordered'] = True
        if unit['id'] in reasons:
            entry['reason'] = reasons[unit['id']]
        report_units.append(entry)

    every_unit = ' \n '.join(u['text'] for u in units if u['kind'] not in ('note', 'boneyard'))
    beyond = [dict(still=r['still'], text=r['text']) for r in effective
              if r['text'] and not contains(every_unit, r['text'], loose=True)
              and not contains(' \n '.join(by_id[u]['text'] for u in r['cited']), r['text'], loose=True)]
    uncited = [r['still'] for r in effective if r['citesNothing']]
    outside = [dict(still=r['still'], units=[u for u in r['cited'] if not in_span(u)])
               for r in effective if any(not in_span(u) for u in r['cited'])]
    jobs = _dialogue_jobs(project, manifests)
    job_lines, subtitle_extra = [], []
    for row in effective:
        for job in jobs.get(row['still'], []):
            unit_text = ' '.join(by_id[u]['text'] for u in row['cited'])
            line = dict(still=row['still'], manifest=job['manifest'], manifestSha256=job['manifestSha256'],
                        hindi=job['hindi'], units=row['cited'])
            line['jobAgainstUnits'] = ('no unit cited' if not row['cited'] else
                                       'verbatim' if contains(unit_text, job['hindi']) else
                                       'nukta-or-joiner' if contains(unit_text, job['hindi'], True) else 'changed')
            if job['prompt'] is not None:
                runs = [m.group(0).strip() for m in DEVANAGARI_RUN.finditer(job['prompt'])]
                line['promptLinesNotInUnits'] = [r for r in runs if normalize(r)
                                                 and not contains(unit_text, r, loose=True)]
            else:
                line['prompt'] = 'not read (missing file)'
            job_lines.append(line)
            if (row['kind'] == 'dialogue' and row['text'] and job['hindi']
                    and normalize(job['hindi']) != normalize(row['text'])
                    and contains(row['text'], job['hindi'])):
                subtitle_extra.append(dict(still=row['still'], subtitle=row['text'], spoken=job['hindi']))
    split_english = []
    for entry in report_units:
        stills = entry.get('splitAcross') or []
        texts = [r['english'] for r in effective if r['still'] in stills]
        if len(texts) > 1 and len({normalize(t) for t in texts}) < len(texts):
            split_english.append(dict(unit=entry['id'], rows=stills, english=texts))
    return dict(
        kind='screenplay-unit-coverage', schemaVersion=1,
        scope='How this storyboard uses the source. A departure is a choice to see, not a fault; nothing gates on it.',
        units=dict(path=_relative(units_path, project), sha256=sha256_bytes(units_raw)),
        storyboard=dict(path=_relative(board_path, project), sha256=sha256_bytes(board_path.read_bytes())),
        **({'overrides': dict(path=_relative(overrides_path, project),
                              sha256=sha256_bytes(overrides_path.read_bytes()))} if overrides_path else {}),
        **({'episodeRange': dict(first=episode_range[0], last=episode_range[1],
                                 unitsSha256=sha256_bytes(units_raw))} if episode_range else {}),
        comparison='NFC; punctuation, danda and spacing normalised; nukta and ZWJ/ZWNJ reported separately',
        sourceUnits=report_units,
        spokenTextInNoUnit=beyond,
        rowsCitingNoUnit=uncited,
        **({'rowsCitingUnitsOutsideRange': outside} if episode_range else {}),
        subtitleCarriesUnspokenWords=subtitle_extra,
        splitRowsWithIdenticalEnglish=split_english,
        dialogueJobLines=job_lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--screenplay', type=Path, help='Fountain file to read into source units')
    mode.add_argument('--units', type=Path, help='Source units file to report coverage against')
    parser.add_argument('--output', type=Path,
                        help='Read mode: default story/source-units-vN.json; --units: the report path (required)')
    parser.add_argument('--storyboard', type=Path, default=Path('story/storyboard_plan.json'))
    parser.add_argument('--overrides', type=Path, help='Compiled overrides.json, for the effective row text')
    parser.add_argument('--dialogue-manifest', type=Path, action='append', default=[],
                        help='Frozen dialogue execution manifest to compare against the units (repeatable)')
    parser.add_argument('--reasons', type=Path,
                        help='Optional JSON object of unit ID to free-text reason, shown in the report')
    parser.add_argument('--episode-range', nargs=2, metavar=('FIRST', 'LAST'),
                        help='Report only the units from FIRST to LAST, recorded with the units hash')
    args = parser.parse_args(argv)
    project = args.project.resolve()
    if not project.is_dir():
        parser.error('project must be an existing directory')

    def inside(path):
        path = (project / path).resolve()
        if not path.is_relative_to(project):
            parser.error(f'{path} must be inside the project')
        return path

    try:
        if args.screenplay:
            output = inside(args.output) if args.output else None
            read_mode(project, args.screenplay.resolve(), output)
            return 0
        if not args.output:
            parser.error('--units needs --output')
        reasons = json.loads(inside(args.reasons).read_text(encoding='utf-8')) if args.reasons else None
        if reasons is not None and (not isinstance(reasons, dict)
                                    or any(not isinstance(v, str) for v in reasons.values())):
            parser.error('--reasons must be a JSON object of unit ID to text')
        report = units_report(project, inside(args.units), inside(args.storyboard),
                              inside(args.overrides) if args.overrides else None,
                              [inside(m) for m in args.dialogue_manifest], reasons, args.episode_range)
        output = inside(args.output)
        if output.exists():
            raise ScreenplayError('Preserve existing report; choose a new path: ' + str(output))
        _write_new(output, json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(str(output))
        return 0
    except ScreenplayError as exc:
        print(f'read_screenplay: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
