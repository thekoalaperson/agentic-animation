#!/usr/bin/env python3
"""Timed still animatic; scratch dialogue is explicitly silent.

This is planning media, never final motion, performance or spend approval.
Episode manifests own shot durations, source holds and complete spoken text.
Only already-generated narration or bound temporary audio is used; no network
or generation occurs.

Rows are `narration` (measured narration that must fit whole),
`scratch-dialogue` (silent, authored timing) or `hold` (authored frames with
optional bound temporary audio and optional text). A plan may bind one
temporary score stem (`tempScore`) that is mixed under the voices.

Without --preview the render runs on the qualified GPU worker, as before.
--preview runs anywhere ffmpeg runs. Its outputs are named
`*-preview-not-final`, carry finalApproved false and record the real host,
ffmpeg build and font hash. Off the worker typography is unqualified; card
text in a script that needs complex shaping is refused when no shaper is
available, and --no-cards renders the stills with no text at all.

--from-storyboard writes a plan instead of rendering: measured narration
where the file exists, an explicit estimatedSeconds where authored, and a
refusal naming the row otherwise.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common

MODES = ('narration', 'scratch-dialogue', 'hold')
PREVIEW = '-preview-not-final'


def sha(path):
    with Path(path).open('rb') as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def run(command):
    try:
        return subprocess.run([str(s) for s in command], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.decode('utf-8', errors='replace')[-6000:]) from exc


def save(path, value):
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def labelled(name, preview):
    """A preview output's name carries -preview-not-final before its suffix."""
    if not preview:
        return name
    path = Path(name)
    return path.stem + PREVIEW + path.suffix


def bound(root, spec):
    path = (root / spec['path']).resolve()
    path.relative_to(root)
    if sha(path) != spec['sha256']:
        raise ValueError('Bound source changed: ' + str(path))
    return path


def validate(plan):
    if plan.get('kind') != 'timed-still-animatic-not-final' or plan.get('finalApproved') is not False:
        raise ValueError('An animatic cannot masquerade as a final film')
    if plan.get('fps') != 24 or not 1 <= len(plan['scenes']) <= 300:
        raise ValueError('Finite 24fps scene plan required')
    seen = set()
    cursor = 0
    for row in plan['scenes']:
        if not re.fullmatch(r'[a-z0-9-]+', row['id']) or row['id'] in seen:
            raise ValueError('Duplicate or invalid ID')
        seen.add(row['id'])
        if not isinstance(row['frames'], int) or not 12 <= row['frames'] <= 1440:
            raise ValueError('Invalid scene frame count')
        if row.get('startFrame', cursor) != cursor:
            raise ValueError('Scene clock is not contiguous')
        cursor += row['frames']
        if row['mode'] not in MODES:
            raise ValueError('Unknown audio mode')
        if row['mode'] == 'scratch-dialogue' and row.get('audio'):
            raise ValueError('Scratch dialogue cannot invent a performance')
        if row['mode'] == 'narration' or (row['mode'] == 'hold' and row.get('audio')):
            duration = row['audio']['durationSeconds']
            lead = row.get('audioLeadFrames', 0) / 24
            if not math.isfinite(duration) or duration <= 0 or lead < 0 or duration + lead > row['frames'] / 24 + 1e-6:
                raise ValueError('Whole narration must fit without retiming')
        if row['mode'] == 'hold':
            if not isinstance(row.get('text', ''), str) or not row.get('sourceStatus'):
                raise ValueError('A hold needs an honest source status: ' + row['id'])
        elif not row.get('text') or not row.get('sourceStatus'):
            raise ValueError('Spoken text and honest source status required')
    if plan.get('frames', cursor) != cursor:
        raise ValueError('Declared total frames mismatch')
    score = plan.get('tempScore')
    if score is not None:
        gain = score.get('gainDb', 0)
        if (not isinstance(score.get('path'), str) or not re.fullmatch(r'[0-9a-f]{64}', str(score.get('sha256')))
                or isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain)
                or not -40 <= gain <= 0):
            raise ValueError('tempScore needs a bound path, sha256 and gainDb between -40 and 0')


def _font_families(fonts, font_dir):
    families = {}
    for role, source in fonts.items():
        source = Path(source)
        target = font_dir / (sha(source) + source.suffix)
        if not target.exists():
            shutil.copyfile(source, target)
        if sha(target) != sha(source):
            raise ValueError('Changed font cache')
        families[role] = run(['fc-scan', '--format=%{family}\n', target]).stdout.decode().splitlines()[0].split(',')[0]
        if not families[role] or any(c in families[role] for c in '\n\r'):
            raise ValueError('Invalid font family')
    return families


def card_label(row):
    return row['id'] + ' | ' + row['mode'].upper() + ' | ' + row['sourceStatus']


def _ass_raster(row, destination, fonts, raster):
    import numpy as np
    from native_text_gpu import literal
    font_dir = destination.parent / 'type-fonts'
    font_dir.mkdir(exist_ok=True)
    families = _font_families(fonts, font_dir)
    ass = destination.with_suffix('.ass')
    for path in (ass, font_dir):
        if any(c in str(path) for c in "':,[]\\"):
            raise ValueError('Unsafe typography filter path')
    label = card_label(row)
    text = row.get('text') or ''
    attempts = []
    for size in (96, 88, 80, 72):
        script = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1280
ScaledBorderAndShadow: yes
WrapStyle: 0
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
'''
        script += f"Style: label,{families['latin']},34,&H008AD1FF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,24,24,10,1\n"
        script += f"Style: caption,{families['hi']},{size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,34,34,1084,1\n"
        script += '[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n'
        script += 'Dialogue: 0,0:00:00.00,0:00:01.00,label,,0,0,0,,' + literal(label) + '\n'
        script += 'Dialogue: 0,0:00:00.00,0:00:01.00,caption,,0,0,0,,' + literal(text) + '\n'
        ass.write_text(script, encoding='utf8')
        result = run(['ffmpeg', '-y', '-v', 'info', '-f', 'lavfi', '-i',
            'color=c=black@0:s=1920x1280:r=1,format=rgba', '-vf',
            f'ass=filename={ass}:fontsdir={font_dir}:alpha=1:shaping=complex',
            '-frames:v', '1', '-threads', '2', raster])
        logs = result.stderr.decode()
        destination.with_suffix('.type.log').write_text(logs)
        if 'HarfBuzz' not in logs or '(COMPLEX)' not in logs or 'Glyph 0x' in logs or 'fontselect: failed' in logs:
            raise ValueError('Complex shaping/font coverage not verified: ' + row['id'])
        raw = run(['ffmpeg', '-v', 'error', '-i', raster, '-frames:v', '1', '-pix_fmt', 'rgba', '-f', 'rawvideo', '-']).stdout
        pixels = np.frombuffer(raw, np.uint8).reshape(1280, 1920, 4)
        ys, xs = np.where(pixels[1080:, :, 3] > 4)
        bounds = [int(xs.min()), int(ys.min()+1080), int(xs.max()+1), int(ys.max()+1081)] if len(xs) else None
        safe = bool(bounds and bounds[0] >= 24 and bounds[2] <= 1896 and bounds[1] >= 1090 and bounds[3] <= 1260)
        if not text and bounds is None:
            safe = True
        attempts.append({'fontSize2x': size, 'captionBounds2x': bounds, 'passed': safe})
        if safe:
            break
    else:
        raise ValueError('Caption exceeds animatic card: ' + row['id'])
    return {'shaper': 'libass complex/HarfBuzz',
            'fonts': {k: {'path': v, 'sha256': sha(v), 'family': families[k]} for k, v in fonts.items()},
            'attempts': attempts}


def _plain_raster(row, fonts, raster):
    """Card text drawn glyph by glyph with Pillow; refused for complex scripts."""
    from PIL import Image, ImageDraw, ImageFont
    label = card_label(row)
    text = row.get('text') or ''
    if _common.needs_complex_shaping(label + text):
        raise ValueError('Card text for ' + row['id'] + ' needs complex shaping, and this ffmpeg has '
                         'no libass; pass --no-cards, or render on a host whose ffmpeg has libass with HarfBuzz')
    font_path = (fonts or {}).get('hi') or (fonts or {}).get('latin')

    def face(size):
        return ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size)
    page = Image.new('RGBA', (1920, 1280), (0, 0, 0, 0))
    draw = ImageDraw.Draw(page)
    draw.text((24, 10), label, font=face(34), fill=(255, 209, 138, 255))
    attempts = []
    for size in (96, 88, 80, 72):
        box = draw.textbbox((0, 0), text, font=face(size)) if text else None
        width = (box[2] - box[0]) if box else 0
        safe = width <= 1872
        attempts.append({'fontSize2x': size, 'captionWidth2x': width, 'passed': safe})
        if safe:
            if text:
                draw.text(((1920 - width) // 2 - box[0], 1094), text, font=face(size), fill=(255, 255, 255, 255))
            break
    else:
        raise ValueError('Caption exceeds animatic card: ' + row['id'])
    page.save(raster)
    return {'shaper': 'Pillow basic layout; no complex shaping', 'typography': 'unqualified',
            'fonts': ({k: {'path': str(v), 'sha256': sha(v)} for k, v in fonts.items()} if fonts
                      else 'Pillow built-in default'),
            'attempts': attempts}


def card(still, row, destination, fonts, preview=False, no_cards=False):
    if not preview and platform.system() != 'Linux':
        raise RuntimeError('Animatic typography belongs on the qualified GPU host')
    destination = Path(destination)
    still_graph = 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1280:(ow-iw)/2:0:color=0x0b111b'
    if no_cards:
        run(['ffmpeg', '-v', 'error', '-y', '-i', still, '-vf', still_graph + ',scale=960:640:flags=lanczos',
             '-frames:v', '1', '-threads', '2', destination])
        save(destination.with_suffix('.type.json'), {'shaper': None, 'typography': 'none; --no-cards',
             'outputSha256': sha(destination), 'visualApproval': False})
        return
    raster = destination.with_suffix('.type.png')
    if not preview:
        record = _ass_raster(row, destination, fonts, raster)
    elif (_common.needs_complex_shaping(card_label(row) + (row.get('text') or ''))
            and fonts and 'ass' in _common.ffmpeg_capabilities()['filters']):
        # Complex scripts off the worker still go through libass and its
        # HarfBuzz verification; anything else uses the plain path below.
        record = _ass_raster(row, destination, fonts, raster)
        record['typography'] = 'unqualified; libass on a preview host'
    else:
        record = _plain_raster(row, fonts, raster)
    graph = f'[0:v]{still_graph},drawbox=x=0:y=0:w=1920:h=60:color=0x0b111b:t=fill[bg];[bg][1:v]overlay=0:0,scale=960:640:flags=lanczos[out]'
    run(['ffmpeg', '-v', 'error', '-y', '-i', still, '-i', raster, '-filter_complex', graph,
         '-map', '[out]', '-frames:v', '1', '-threads', '2', destination])
    save(destination.with_suffix('.type.json'), {**record, 'outputSha256': sha(destination), 'visualApproval': False})


def media_seconds(path):
    probe = json.loads(run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'json', path]).stdout)
    seconds = float(probe['format']['duration'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Unmeasurable audio: ' + str(path))
    return seconds


def plan_from_storyboard(episode, board, manifest, measure=media_seconds, narration_dir='audio/narration-v1',
                         estimates=None, fonts=None, temp_score=None):
    """Build an animatic plan from the storyboard and the registered stills.

    A narration row uses its measured file where one exists. Any other row,
    or narration not yet made, needs an explicit estimatedSeconds on the
    storyboard row or in the estimates mapping; unmade narration becomes a
    silent hold that still shows its words. Rows with the reserved speaker
    '-' or no text become holds. Nothing is guessed.
    """
    episode = Path(episode).resolve()
    estimates = estimates or {}
    records = {r['still_id']: r for r in manifest['stills']}
    known = {row['still_id'] for row in board['stills']}
    if set(estimates) - known:
        raise ValueError('Estimates name stills not on the storyboard: ' + ', '.join(sorted(set(estimates) - known)))
    scenes, missing, cursor = [], [], 0
    for row in board['stills']:
        key = row['still_id']
        rec = records.get(key)
        if rec is None or not rec.get('provenance_sidecar'):
            missing.append(f'{key}: still is not registered in the generation manifest')
            continue
        still = (episode / rec['target_output_path']).resolve()
        sidecar = (episode / rec['provenance_sidecar']).resolve()
        if not still.is_relative_to(episode) or not sidecar.is_relative_to(episode):
            raise ValueError('Still or sidecar escapes the episode: ' + key)
        if (not still.is_file() or not sidecar.is_file()
                or json.loads(sidecar.read_text()).get('target_sha256') != sha(still)):
            missing.append(f'{key}: still bytes do not match the registered sidecar')
            continue
        authored = row.get('estimatedSeconds')
        if key in estimates:
            if authored is not None and authored != estimates[key]:
                raise ValueError(f'{key}: storyboard and estimates file disagree')
            authored = estimates[key]
        if authored is not None and (isinstance(authored, bool) or not isinstance(authored, (int, float))
                                     or not math.isfinite(authored) or authored <= 0):
            raise ValueError(f'{key}: estimatedSeconds must be a positive number')
        speaker = row.get('speaker') or ''
        text = row.get('narration_hi') or ''
        if speaker == '-' or not text:
            mode = 'hold'
        elif speaker:
            mode = 'scratch-dialogue'
        else:
            mode = 'narration'
        scene = {'id': key, 'mode': mode, 'text': text,
                 'still': {'path': str(still.relative_to(episode)), 'sha256': sha(still)}}
        audio = (episode / narration_dir / f'{key}-hi.mp3').resolve() if mode == 'narration' else None
        if audio is not None and audio.is_file():
            if not audio.is_relative_to(episode):
                raise ValueError('Narration escapes the episode: ' + key)
            seconds = measure(audio)
            scene['audio'] = {'path': str(audio.relative_to(episode)), 'sha256': sha(audio),
                              'durationSeconds': seconds}
            frames = math.ceil(seconds * 24 - 1e-9)
            basis = 'measured narration'
            if authored is not None and round(authored * 24) > frames:
                frames, basis = round(authored * 24), 'estimatedSeconds, longer than the measured narration'
        elif authored is not None:
            frames, basis = round(authored * 24), 'estimatedSeconds'
            if mode == 'narration':
                # Unmade narration plays as a silent, worded hold on its estimate.
                scene['mode'], basis = 'hold', 'narration not yet made; estimatedSeconds'
        else:
            what = 'no measured narration file and ' if mode == 'narration' else ''
            missing.append(f'{key}: {what}no estimatedSeconds for this {mode} row')
            continue
        scene.update(frames=frames, startFrame=cursor,
                     sourceStatus=f'registered still; {basis}', timingBasis=basis)
        cursor += frames
        scenes.append(scene)
    if missing:
        raise ValueError('Cannot time the animatic:\n  ' + '\n  '.join(missing))
    plan = {'kind': 'timed-still-animatic-not-final', 'finalApproved': False, 'fps': 24,
            'frames': cursor, 'scenes': scenes}
    if fonts:
        plan['fonts'] = fonts
    if temp_score:
        plan['tempScore'] = temp_score
    validate(plan)
    return plan


def write_plan(args):
    ep = args.episode.resolve()
    target = args.manifest.resolve()
    target.relative_to(ep / 'work')
    if target.exists():
        raise ValueError('Animatic plan exists; choose a new version: ' + str(target))
    board_path = (args.storyboard or ep / 'story/storyboard_plan.json').resolve()
    manifest_path = (args.stills_manifest or ep / 'story/generation_manifest.json').resolve()
    estimates = json.loads(args.estimates.read_text()) if args.estimates else None
    font = _common.review_font(args.font)
    fonts = {'latin': str(font), 'hi': str(font)} if font else None
    temp_score = None
    if args.temp_score:
        score = args.temp_score.resolve()
        score.relative_to(ep)
        temp_score = {'path': str(score.relative_to(ep)), 'sha256': sha(score), 'gainDb': args.temp_score_gain_db}
    plan = plan_from_storyboard(ep, json.loads(board_path.read_text()), json.loads(manifest_path.read_text()),
                                estimates=estimates, fonts=fonts, temp_score=temp_score)
    plan['writtenFrom'] = {'storyboard': str(board_path), 'storyboardSha256': sha(board_path),
                           'stillsManifest': str(manifest_path), 'stillsManifestSha256': sha(manifest_path),
                           'estimatesSha256': sha(args.estimates) if args.estimates else None,
                           'toolSha256': sha(__file__)}
    _common.atomic_new_bytes(target, (json.dumps(plan, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'plan': str(target), 'scenes': len(plan['scenes']), 'frames': plan['frames'],
                      'sha256': sha(target)}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode', type=Path)
    ap.add_argument('manifest', type=Path, help='The animatic plan under work/ (written by --from-storyboard)')
    ap.add_argument('--jobs', type=int, help='Exact scene count; required to render')
    ap.add_argument('--reuse-picture-tool', help='Exact prior renderer SHA for cache-only audio/clock repair; missing cache is an error')
    ap.add_argument('--preview', action='store_true', help='Render anywhere ffmpeg runs; outputs are labelled preview-not-final')
    ap.add_argument('--no-cards', action='store_true', help='Preview only: stills without card text')
    ap.add_argument('--font', type=Path, help='Preview card font, or the font a written plan names; else the profile fonts.review')
    ap.add_argument('--from-storyboard', action='store_true', help='Write the plan instead of rendering')
    ap.add_argument('--storyboard', type=Path, help='With --from-storyboard: default story/storyboard_plan.json')
    ap.add_argument('--stills-manifest', type=Path, help='With --from-storyboard: default story/generation_manifest.json')
    ap.add_argument('--estimates', type=Path, help='With --from-storyboard: JSON object of still ID to estimatedSeconds')
    ap.add_argument('--temp-score', type=Path, help='With --from-storyboard: a temporary score stem to bind')
    ap.add_argument('--temp-score-gain-db', type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.from_storyboard:
        return write_plan(args)
    if args.jobs is None:
        ap.error('--jobs is required to render')
    if args.no_cards and not args.preview:
        ap.error('--no-cards is a preview option')
    preview = args.preview
    ep = args.episode.resolve()
    manifest = args.manifest.resolve()
    manifest.relative_to(ep / 'work')
    plan = json.loads(manifest.read_text())
    validate(plan)
    if len(plan['scenes']) != args.jobs:
        raise ValueError('Exact finite count required')
    if args.reuse_picture_tool and not re.fullmatch(r'[0-9a-f]{64}', args.reuse_picture_tool):
        raise ValueError('Exact prior renderer SHA required')
    if preview:
        if args.reuse_picture_tool:
            raise ValueError('A preview never reuses a worker picture cache')
        host = _common.preview_host_record(None)
    else:
        if platform.system() != 'Linux':
            raise RuntimeError('All media work belongs on the qualified GPU host, not the Mac')
        run(['nvidia-smi', '-L'])
    plan_hash = sha(manifest)
    fonts = {}
    if not args.no_cards:
        override = _common.review_font(args.font) if preview else None
        fonts = {'latin': str(override), 'hi': str(override)} if override else plan.get('fonts', {})
        if not preview and not fonts:
            raise ValueError('Plan fonts are required for qualified cards')
        if fonts and set(fonts) != {'latin', 'hi'}:
            raise ValueError('Plan fonts need latin and hi roles')
        for font in fonts.values():
            if not Path(font).is_file():
                raise ValueError('Missing retained font: ' + font
                                 + ('; pass --font or --no-cards for a preview' if preview else ''))
    if preview:
        host['font'] = {k: {'path': v, 'sha256': sha(v)} for k, v in fonts.items()} or None
    inputs = []
    for row in plan['scenes']:
        inputs.append((bound(ep, row['still']), bound(ep, row['audio']) if row.get('audio') else None))
    score_path = bound(ep, plan['tempScore']) if plan.get('tempScore') else None
    root = manifest.parent
    cache = root / labelled('segments', preview)
    cache.mkdir(exist_ok=True)
    lock = root / '.render.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        outputs = []
        for row, (still, audio) in zip(plan['scenes'], inputs):
            identity = {'scene': row, 'tool': args.reuse_picture_tool or sha(__file__), 'fonts': {k: sha(v) for k,v in fonts.items()}}
            if preview:
                identity.update(preview=True, noCards=args.no_cards, ffmpeg=host['ffmpeg'])
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            output = cache / (row['id'] + '-' + key + '.mov')
            receipt = output.with_suffix('.json')
            if output.exists():
                if not receipt.is_file() or json.loads(receipt.read_text())['sha256'] != sha(output):
                    raise ValueError('Untracked/changed cached segment')
            else:
                if args.reuse_picture_tool:
                    raise ValueError('Required immutable prior picture cache missing: ' + str(output))
                source_card = output.with_suffix('.png')
                card(still, row, source_card, fonts, preview=preview, no_cards=args.no_cards)
                samples = row['frames'] * 2000
                cmd = ['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-framerate', '24', '-i', source_card]
                cmd += ['-i', audio] if audio else ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo']
                lead = row.get('audioLeadFrames', 0) * 2000
                graph = f'[1:a]aresample=48000,adelay={lead}S:all=1,apad=whole_len={samples},atrim=end_sample={samples},asetpts=N/SR/TB[a]'
                temp = output.with_suffix('.partial.mov')
                # The retained L4 has compute drivers but no NVENC user library.
                # Encode this low-resolution planning artifact with libx264; a
                # preview does the same wherever it runs and says so.
                cmd += ['-filter_complex', graph, '-map', '0:v:0', '-map', '[a]', '-frames:v', row['frames'], '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-threads', '4', '-pix_fmt', 'yuv420p', '-video_track_timescale', '24000', '-c:a', 'pcm_s16le', '-ar', '48000', '-ac', '2', '-t', f"{row['frames']/24:.9f}", temp]
                run(cmd)
                temp.replace(output)
                record = {'id': row['id'], 'sha256': sha(output), 'frames': row['frames'], 'plannedAudioSamples': samples, 'sourceCardSha256': sha(source_card), 'finalApproved': False}
                if preview:
                    record['preview'] = True
                save(receipt, record)
            outputs.append(output)
            print(json.dumps({'id': row['id'], 'status': 'COMPLETED'}), flush=True)
        listing = root / labelled('segments.ffconcat', preview)
        if any("'" in str(p) or '\n' in str(p) for p in outputs):
            raise ValueError('Unsafe concat path')
        listing.write_text('ffconcat version 1.0\n' + ''.join(f"file '{p}'\n" for p in outputs))
        final = root / labelled('timed-animatic-v1.mp4', preview)
        temp = final.with_suffix('.partial.mp4')
        frames = sum(s['frames'] for s in plan['scenes'])
        # Segment audio can end early when a video frame limit ends FFmpeg.
        # Rebuild from whole original recordings on the integer scene clock;
        # never use cached segment sound or pad over missing spoken material.
        import numpy as np
        from assemble_longform import audio as decode_audio, write_audio
        stem = np.zeros((frames * 2000, 2), dtype=np.float32)
        speech = []
        cursor = 0
        for row, (_, source) in zip(plan['scenes'], inputs):
            if source:
                samples = decode_audio(source)
                offset = row.get('audioLeadFrames', 0) * 2000
                if offset + len(samples) > row['frames'] * 2000:
                    raise ValueError('Decoded whole voice exceeds scene: ' + row['id'])
                start = cursor * 2000 + offset
                stem[start:start+len(samples)] = samples
                event = {'id': row['id'], 'sourceSha256': row['audio']['sha256'], 'startSample': start, 'completeDecodedSamples': len(samples), 'retimed': False, 'trimmed': False}
                if row['mode'] == 'hold':
                    event['role'] = 'temporary audio under a hold'
                speech.append(event)
            cursor += row['frames']
        voice = root / labelled('complete-narration-clock.wav', preview)
        mixed = score_record = None
        if score_path is not None:
            score = decode_audio(score_path)[:frames * 2000]
            gain_db = plan['tempScore'].get('gainDb', 0)
            mixed = stem.copy()
            mixed[:len(score)] += score * (10 ** (gain_db / 20))
            peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
            if peak >= 1:
                raise ValueError('Voices plus temporary score clip; lower tempScore.gainDb')
            score_record = {'sourceSha256': plan['tempScore']['sha256'], 'gainDb': gain_db,
                            'decodedSamplesUsed': len(score), 'endsBeforeFilm': len(score) < frames * 2000,
                            'peakOfMix': peak, 'role': 'temporary score; never a production stem'}
        write_audio(voice, stem)
        del stem
        voice_audit = root / labelled('voice-clock-audit.json', preview)
        save(voice_audit, {'samples': frames*2000, 'sampleRate': 48000, 'sha256': sha(voice), 'events': speech, 'scratchDialogue': 'silent', 'source': 'whole original narration, not segment audio'})
        mux = voice
        if mixed is not None:
            mux = root / labelled('animatic-mix-with-temp-score.wav', preview)
            write_audio(mux, mixed)
            del mixed
            score_record['mixSha256'] = sha(mux)
        # Reset picture PTS from decoded frame number; fractional container
        # duration rounding must not accumulate at the150 clip boundaries.
        run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', listing, '-i', mux, '-map', '0:v:0', '-map', '1:a:0', '-vf', 'setpts=N/(24*TB)', '-r', '24', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-threads', '4', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-t', f'{frames/24:.9f}', '-movflags', '+faststart', temp])
        run(['ffmpeg', '-v', 'error', '-xerror', '-i', temp, '-f', 'null', '-'])
        probe = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', temp]).stdout)
        v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        a = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        if int(v['nb_frames']) != frames or abs(float(v['duration']) - frames/24) > .045 or abs(float(a['duration'])-frames/24) > .065:
            raise ValueError('Animatic clock mismatch')
        if sha(manifest) != plan_hash:
            raise ValueError('Manifest changed during rendering')
        temp.replace(final)
        if preview:
            encoder = f"libx264 via {host['ffmpeg']} on {host['computeHost']} ({host['platform']}); preview, not the qualified worker"
        else:
            encoder = 'libx264 on qualified GPU worker; no Mac processing'
        audit = {'kind': plan['kind'], 'finalApproved': False, 'manifestSha256': plan_hash, 'toolSha256': sha(__file__), 'host': platform.node(), 'encoder': encoder, 'output': str(final), 'sha256': sha(final), 'scenes': len(outputs), 'frames': frames, 'durationSeconds': frames/24, 'fullDecode': 'passed', 'narrationCount': sum(s['mode']=='narration' for s in plan['scenes']), 'scratchDialogueCount': sum(s['mode']=='scratch-dialogue' for s in plan['scenes']), 'limitations': 'Silent dialogue slots use authored temporary timings. Still holds and known source defects are labeled. This is not final motion or a hearing/temporal approval.'}
        audit.update(reusedPictureToolSha256=args.reuse_picture_tool, voiceClockAuditSha256=sha(voice_audit))
        if any(s['mode'] == 'hold' for s in plan['scenes']):
            audit['holdCount'] = sum(s['mode'] == 'hold' for s in plan['scenes'])
        if score_record is not None:
            audit['tempScore'] = score_record
        if preview:
            audit.update(preview=True, outputLabel='preview-not-final', hostRecord=host,
                         typography='none; --no-cards' if args.no_cards else 'unqualified off the worker')
        save(root / labelled('render-audit.json', preview), audit)
        print(json.dumps(audit), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
