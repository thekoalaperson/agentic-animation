#!/usr/bin/env python3
"""Inventory a still-board episode, safely ingest silent H3 candidates, prepare VO.

No generation, network or creative approval. Source lineage and missing recorded
input hashes remain explicit. Review sheets are diagnostics, not film graphics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common
from _common import profile_value

WORKER_FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
PREVIEW_SHEETS = 'sheets-preview-not-final'


def sha(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('Existing input differs; choose a new work version: ' + str(path))
    if not path.exists():
        path.write_text(text)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
        '-show_streams', '-show_format', '-of', 'json', str(path)]))


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True)


def ingest(job):
    source, target = job
    sidecar = target.with_suffix('.ingest.json')
    source_hash = sha(source)
    if target.exists():
        if not sidecar.exists():
            raise ValueError('Untracked silent output: ' + str(target))
        old = json.loads(sidecar.read_text())
        if old['sourceSha256'] != source_hash or old['sha256'] != sha(target):
            raise ValueError('Source/output hash changed: ' + str(target))
        return old
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.partial.mp4')
    run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-map', '0:v:0',
         '-c:v', 'copy', '-an', '-movflags', '+faststart', str(tmp)])
    media = probe(tmp)
    assert len(media['streams']) == 1 and media['streams'][0]['codec_type'] == 'video'
    run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(tmp), '-f', 'null', '-'])
    tmp.replace(target)
    v = media['streams'][0]
    record = dict(source=str(source), sourceSha256=source_hash, file=str(target),
                  sha256=sha(target), duration=float(v['duration']),
                  width=v['width'], height=v['height'], fps=v['r_frame_rate'],
                  frames=int(v['nb_frames']), audioStreams=0,
                  technique='video stream copy; all H3 audio discarded',
                  creativeApproval='pending-independent-internal-review')
    save(sidecar, record)
    return record


STAGING_KEYS = ('screen_side', 'gaze', 'attitude', 'reacting', 'line')


def staging_summary(staging, limit=60):
    """One line from a generation-manifest row's optional free-text staging notes.

    The notes are the operator's own blocking reminders; nothing checks them
    against any rule. Known keys come first, any others follow in their order.
    """
    if not isinstance(staging, dict):
        return ''
    keys = [k for k in STAGING_KEYS if k in staging] + [k for k in staging if k not in STAGING_KEYS]
    parts = []
    for key in keys:
        value = staging[key]
        if isinstance(value, dict):
            value = ' '.join(str(v) for v in value.values() if v not in (None, ''))
        if value not in (None, ''):
            parts.append(f"{key.replace('_', ' ')}: {value}")
    text = ' / '.join(parts)
    return text if len(text) <= limit else text[:limit - 1] + '\u2026'


def sheet_caption(row, staging=None):
    """Still ID and speaker on the first line, staging summary on the second."""
    first = row['id'] + (f" | {row['speaker']}" if row['speaker'] else '')
    return first, staging_summary(staging)


def sample_times(duration, count=6):
    """Sheet sample times spread over a clip's measured duration.

    The first sample is frame 0 and the last is the clip's final frame, so a
    3-second clip and a 12-second clip are both seen whole.
    """
    if not duration > 0 or count < 2:
        raise ValueError('A motion sheet needs a measured positive duration')
    last = max(0.0, duration - 1 / 24)
    return [round(last * i / (count - 1), 3) for i in range(count)]


def cut_pairs(rows):
    """Consecutive story-order rows, as (outgoing, incoming) pairs."""
    return list(zip(rows, rows[1:]))


def clip_seconds(path):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                   '-of', 'json', str(path)])
    return float(json.loads(out)['format']['duration'])


def frame_at(source, seconds=None, last=False, size=(256, 146)):
    """One decoded frame as a PIL image: at a time, or the clip's final frame."""
    from PIL import Image
    scale = f'scale={size[0]}:{size[1]}'
    if not last:
        raw = run(['ffmpeg', '-v', 'error', '-ss', str(seconds), '-i', str(source),
                   '-frames:v', '1', '-vf', scale, '-f', 'image2pipe', '-vcodec', 'png', '-']).stdout
        return Image.open(io.BytesIO(raw)).convert('RGB')
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / 'last.png'
        run(['ffmpeg', '-v', 'error', '-sseof', '-1', '-i', str(source), '-vf', scale,
             '-update', '1', '-f', 'image2', str(target)])
        return Image.open(target).convert('RGB')


def sheet_fonts(font_path=None, preview=False):
    """Review fonts: an explicit or profile font, else the worker's DejaVuSans.

    A preview off the worker falls back to Pillow's own font and says so.
    """
    from PIL import ImageFont
    path = font_path or (WORKER_FONT if WORKER_FONT.is_file() else None)
    if path is None:
        if not preview:
            raise ValueError('GPU worker requires its retained DejaVuSans review font')
        return ImageFont.load_default(20), ImageFont.load_default(15), None
    return ImageFont.truetype(str(path), 20), ImageFont.truetype(str(path), 15), Path(path)


def make_sheets(rows, work, clips, staging=None, font_path=None, preview=False):
    from PIL import Image, ImageDraw
    staging = staging or {}
    sheets = work / (PREVIEW_SHEETS if preview else 'sheets')
    sheets.mkdir(parents=True, exist_ok=True)
    font, small, used_font = sheet_fonts(font_path, preview)
    written = []
    for start in range(0, len(rows), 12):
        page = Image.new('RGB', (1440, 4 * 320), '#111722')
        draw = ImageDraw.Draw(page)
        for i, row in enumerate(rows[start:start+12]):
            im = Image.open(row['still']).convert('RGB')
            im.thumbnail((476, 268))
            x, y = (i % 3)*480, (i // 3)*320
            page.paste(im, (x, y))
            first, second = sheet_caption(row, staging.get(row['id']))
            draw.text((x+8,y+270), first, font=font, fill='white')
            if second:
                draw.text((x+8,y+296), second, font=small, fill='#c8d0dc')
        name = sheets / f'stills-{start+1:03d}-{min(start+12,len(rows)):03d}.jpg'
        page.save(name, quality=93)
        written.append(name)
    if clips is not None:
        moving = [row for row in rows if (clips / (row['id'] + '.mp4')).is_file()]
        for start in range(0, len(moving), 6):
            page = Image.new('RGB', (6*256, 6*166), '#111722')
            draw = ImageDraw.Draw(page)
            for i, row in enumerate(moving[start:start+6]):
                source = clips / (row['id'] + '.mp4')
                times = sample_times(clip_seconds(source))
                for j, t in enumerate(times):
                    page.paste(frame_at(source, t), (j*256, i*166))
                draw.text((4,i*166+146),row['id']+' | '+', '.join(f'{t:g}' for t in times)+' s',font=font,fill='white')
            name = sheets / f'motion-{start+1:03d}-{min(start+6,len(moving)):03d}.jpg'
            page.save(name, quality=94)
            written.append(name)
        # Cut pairs: the last frame of each clip beside the first frame of the
        # next one in story order, captioned with speaker and staging notes.
        pairs = [(a, b) for a, b in cut_pairs(rows)
                 if (clips / (a['id'] + '.mp4')).is_file() and (clips / (b['id'] + '.mp4')).is_file()]
        for start in range(0, len(pairs), 4):
            page = Image.new('RGB', (2*400, 4*270), '#111722')
            draw = ImageDraw.Draw(page)
            for i, (outgoing, incoming) in enumerate(pairs[start:start+4]):
                y = i * 270
                for x, row, last in ((0, outgoing, True), (400, incoming, False)):
                    source = clips / (row['id'] + '.mp4')
                    page.paste(frame_at(source, 0, last=last, size=(384, 216)), (x+8, y+4))
                    first, second = sheet_caption(row, staging.get(row['id']))
                    draw.text((x+8, y+222), ('last | ' if last else 'first | ') + first, font=small, fill='white')
                    if second:
                        draw.text((x+8, y+242), second, font=small, fill='#c8d0dc')
            name = sheets / f'cuts-{start+1:03d}-{min(start+4,len(pairs)):03d}.jpg'
            page.save(name, quality=94)
            written.append(name)
    return written, used_font


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode', type=Path)
    ap.add_argument('--h3-sources', type=Path,
                    help='Existing H3 candidates, if any; omit for a still-only episode')
    ap.add_argument('--work',default='work/production-intake-v1')
    ap.add_argument('--voice-id', help='Optional narration voice; otherwise profile voice.id')
    ap.add_argument('--source-root',type=Path,help='Explicit relocated root for immutable imported still manifests; default project')
    ap.add_argument('--sheets',action='store_true')
    ap.add_argument('--preview', action='store_true',
                    help='With --sheets: draw review sheets anywhere ffmpeg runs, into '
                         f'{PREVIEW_SHEETS}/; no ingest and no inventory documents are written')
    ap.add_argument('--font', type=Path, help='Review sheet font; otherwise the profile fonts.review, then the worker DejaVuSans')
    ap.add_argument('--inventory-only', action='store_true',
                    help='Metadata/hash pass only; no media decode, processing or sheets')
    ap.add_argument('--overrides', type=Path,
                    help='Versioned episode-owned text/speaker corrections; originals survive')
    args=ap.parse_args(argv)
    if args.inventory_only and args.sheets:
        ap.error('--inventory-only cannot render sheets')
    if args.preview and not args.sheets:
        ap.error('--preview draws review sheets; pass --sheets')
    if not args.inventory_only and not args.preview:
        if platform.system() != 'Linux':
            ap.error('Media processing must run on a qualified GPU worker; use --inventory-only on the Mac')
        run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    ep=args.episode.resolve()
    repo=args.source_root.resolve() if args.source_root else ep
    work=ep/args.work
    board_path=ep/'story/storyboard_plan.json'
    manifest_path=ep/'story/generation_manifest.json'
    board=json.loads(board_path.read_text())
    manifest=json.loads(manifest_path.read_text())
    override_doc = json.loads(args.overrides.read_text()) if args.overrides else {}
    overrides = override_doc.get('items', {})
    known_ids = {r['still_id'] for r in board['stills']}
    if set(overrides) - known_ids:
        raise ValueError('Overrides contain unknown scene IDs')
    records={r['still_id']:r for r in manifest['stills']}
    rows=[]
    narration=[]
    dialogues=[]
    for row in board['stills']:
        key=row['still_id']
        change = overrides.get(key, {})
        if set(change) - {'speaker', 'narration_hi', 'narration_en', 'reason'}:
            raise ValueError('Unsupported override fields: ' + key)
        row = {**row, **change}
        rec=records[key]
        still=(repo/rec['target_output_path']).resolve()
        provenance=(repo/rec['provenance_sidecar']).resolve()
        if not still.is_relative_to(repo) or not provenance.is_relative_to(repo):
            raise ValueError('Still or provenance path escapes the explicit source root')
        assert still.is_file() and provenance.is_file()
        recorded_still = json.loads(provenance.read_text()).get('target_sha256')
        if not recorded_still or recorded_still != sha(still):
            raise ValueError('Still bytes do not match registered sidecar: ' + key)
        speaker=row.get('speaker') or ''
        display=row['narration_hi']
        text=display
        # The speaker column identifies the performer. The authored text is
        # already the complete utterance; quote marks are not a speech parser.
        r=dict(id=key, speaker=speaker, still=str(still),stillSha256=sha(still),
               stillSidecar=str(provenance),stillSidecarSha256=sha(provenance),
               displayText=display,text=text,english=row['narration_en'],
               beat=row['beat'],historicalH3InputByteBinding='unavailable in original sidecar',
               status='candidate; no human motion verdict found')
        rows.append(r)
        if speaker:
            dialogues.append(r)
        else:
            narration.append(dict(id=key+'-hi',shotId=key,language='hi',
                displayText=text,text=text,role='narration',
                output=f'audio/narration-v1/{key}-hi.mp3'))
    sources=args.h3_sources.resolve() if args.h3_sources else None
    if sources is not None and not sources.is_dir():
        raise ValueError('H3 source directory does not exist: ' + str(sources))
    available = {r['id']: sources/(r['id']+'.mp4') for r in rows
                 if sources is not None and (sources/(r['id']+'.mp4')).is_file()}
    ingested = {}
    if not args.inventory_only and not args.preview:
        with ThreadPoolExecutor(max_workers=4) as pool:
            result=list(pool.map(ingest,[(raw,work/'silent-h3'/(key+'.mp4'))
                                        for key,raw in available.items()]))
            ingested=dict(zip(available,result))
    for row in rows:
        raw = available.get(row['id'])
        if raw is None:
            row['rawH3'] = None
            row['historicalH3InputByteBinding'] = 'not applicable; no existing candidate found'
            row['status'] = 'still inventory only; motion generation pending'
            row['eligibleForProductionPicture'] = False
            continue
        row['rawH3'] = dict(file=str(raw), sha256=sha(raw), bytes=raw.stat().st_size,
                           editorialAudio='forbidden; remove on GPU worker before assembly')
        if row['id'] in ingested:
            row['silentH3']=ingested[row['id']]
        row['eligibleForProductionPicture']=not bool(row['speaker'])
    if args.preview:
        font = _common.review_font(args.font)
        written, used = make_sheets(rows, work, sources, {key: rec.get('staging') for key, rec in records.items()},
                                    font_path=font, preview=True)
        audit = dict(kind='review-sheets-preview-not-final', **_common.preview_host_record(used),
                     boardSha256=sha(board_path), manifestSha256=sha(manifest_path),
                     typography='unqualified off the worker; captions are drawn without complex shaping',
                     sheets=[dict(path=p.name, sha256=sha(p)) for p in written])
        _common.atomic_json(work / PREVIEW_SHEETS / 'sheet-audit.json', audit)
        print(json.dumps(dict(stills=len(rows), sheets=len(written), preview=True)))
        return
    save(work/'inventory.json',dict(episode=manifest['episode'],
         boardSha256=sha(board_path),manifestSha256=sha(manifest_path),
         count=len(rows),narrationCount=len(narration),dialogueCount=len(dialogues),
         historicalH3Count=len(available),missingH3Count=len(rows)-len(available),
         inventoryOnly=args.inventory_only,
         overridesPath=str(args.overrides.resolve()) if args.overrides else None,
         overridesSha256=sha(args.overrides) if args.overrides else None,items=rows))
    save(work/'narration-plan.json',dict(episode=manifest['episode'],
         model='eleven_multilingual_v2',voice=args.voice_id or profile_value('voice','id'),
         financialApproval=False,existingCreditsOnly=True,
         characters=sum(len(r['text']) for r in narration),items=narration))
    save(work/'dialogue-draft.json',dict(episode=manifest['episode'],
         route='grok15 / 720p / native Hindi',financialApproval=False,
         promptsRequireSourceInspection=True,items=dialogues))
    if args.sheets:
        make_sheets(rows,work,sources,{key: rec.get('staging') for key, rec in records.items()},
                    font_path=_common.review_font(args.font))
    print(json.dumps(dict(stills=len(rows),silentH3=len(ingested),
        narration=len(narration),dialogue=len(dialogues),characters=sum(len(r['text']) for r in narration))))


if __name__=='__main__':
    main()
