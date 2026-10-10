#!/usr/bin/env python3
"""GPU-host-only still/VO animatic; scratch dialogue is explicitly silent.

This is planning media, never final motion, performance or spend approval.
Episode manifests own shot durations, source holds and complete spoken text.
Only already-generated narration is used; no network or generation occurs.
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
        if row['mode'] not in ('narration', 'scratch-dialogue'):
            raise ValueError('Unknown audio mode')
        if row['mode'] == 'scratch-dialogue' and row.get('audio'):
            raise ValueError('Scratch dialogue cannot invent a performance')
        if row['mode'] == 'narration':
            duration = row['audio']['durationSeconds']
            lead = row.get('audioLeadFrames', 0) / 24
            if not math.isfinite(duration) or duration <= 0 or lead < 0 or duration + lead > row['frames'] / 24 + 1e-6:
                raise ValueError('Whole narration must fit without retiming')
        if not row.get('text') or not row.get('sourceStatus'):
            raise ValueError('Spoken text and honest source status required')
    if plan.get('frames', cursor) != cursor:
        raise ValueError('Declared total frames mismatch')


def card(still, row, destination, fonts):
    import numpy as np
    from native_text_gpu import literal
    if platform.system() != 'Linux':
        raise RuntimeError('Animatic typography belongs on the qualified GPU host')
    destination = Path(destination)
    font_dir = destination.parent / 'type-fonts'
    font_dir.mkdir(exist_ok=True)
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
    ass = destination.with_suffix('.ass')
    raster = destination.with_suffix('.type.png')
    for path in (ass, font_dir):
        if any(c in str(path) for c in "':,[]\\"):
            raise ValueError('Unsafe typography filter path')
    label = row['id'] + ' | ' + row['mode'].upper() + ' | ' + row['sourceStatus']
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
        script += 'Dialogue: 0,0:00:00.00,0:00:01.00,caption,,0,0,0,,' + literal(row['text']) + '\n'
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
        attempts.append({'fontSize2x': size, 'captionBounds2x': bounds, 'passed': safe})
        if safe:
            break
    else:
        raise ValueError('Caption exceeds animatic card: ' + row['id'])
    graph = '[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1280:(ow-iw)/2:0:color=0x0b111b,drawbox=x=0:y=0:w=1920:h=60:color=0x0b111b:t=fill[bg];[bg][1:v]overlay=0:0,scale=960:640:flags=lanczos[out]'
    run(['ffmpeg', '-v', 'error', '-y', '-i', still, '-i', raster, '-filter_complex', graph,
         '-map', '[out]', '-frames:v', '1', '-threads', '2', destination])
    save(destination.with_suffix('.type.json'), {'shaper': 'libass complex/HarfBuzz',
         'fonts': {k: {'path': v, 'sha256': sha(v), 'family': families[k]} for k,v in fonts.items()},
         'attempts': attempts, 'outputSha256': sha(destination), 'visualApproval': False})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode', type=Path)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--jobs', type=int, required=True)
    ap.add_argument('--reuse-picture-tool', help='Exact prior renderer SHA for cache-only audio/clock repair; missing cache is an error')
    args = ap.parse_args()
    ep = args.episode.resolve()
    manifest = args.manifest.resolve()
    manifest.relative_to(ep / 'work')
    plan = json.loads(manifest.read_text())
    validate(plan)
    if len(plan['scenes']) != args.jobs:
        raise ValueError('Exact finite count required')
    if args.reuse_picture_tool and not re.fullmatch(r'[0-9a-f]{64}', args.reuse_picture_tool):
        raise ValueError('Exact prior renderer SHA required')
    if platform.system() != 'Linux':
        raise RuntimeError('All media work belongs on the qualified GPU host, not the Mac')
    run(['nvidia-smi', '-L'])
    plan_hash = sha(manifest)
    fonts = plan['fonts']
    for font in fonts.values():
        if not Path(font).is_file():
            raise ValueError('Missing retained font: ' + font)
    inputs = []
    for row in plan['scenes']:
        inputs.append((bound(ep, row['still']), bound(ep, row['audio']) if row.get('audio') else None))
    root = manifest.parent
    cache = root / 'segments'
    cache.mkdir(exist_ok=True)
    lock = root / '.render.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        outputs = []
        for row, (still, audio) in zip(plan['scenes'], inputs):
            key = hashlib.sha256(json.dumps({'scene': row, 'tool': args.reuse_picture_tool or sha(__file__), 'fonts': {k: sha(v) for k,v in fonts.items()}}, sort_keys=True).encode()).hexdigest()[:20]
            output = cache / (row['id'] + '-' + key + '.mov')
            receipt = output.with_suffix('.json')
            if output.exists():
                if not receipt.is_file() or json.loads(receipt.read_text())['sha256'] != sha(output):
                    raise ValueError('Untracked/changed cached segment')
            else:
                if args.reuse_picture_tool:
                    raise ValueError('Required immutable prior picture cache missing: ' + str(output))
                source_card = output.with_suffix('.png')
                card(still, row, source_card, fonts)
                samples = row['frames'] * 2000
                cmd = ['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-framerate', '24', '-i', source_card]
                cmd += ['-i', audio] if audio else ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo']
                lead = row.get('audioLeadFrames', 0) * 2000
                graph = f'[1:a]aresample=48000,adelay={lead}S:all=1,apad=whole_len={samples},atrim=end_sample={samples},asetpts=N/SR/TB[a]'
                temp = output.with_suffix('.partial.mov')
                # The retained L4 has compute drivers but no NVENC user library.
                # Encode this low-resolution planning artifact on that worker;
                # do not reinstall drivers or move media processing to the Mac.
                cmd += ['-filter_complex', graph, '-map', '0:v:0', '-map', '[a]', '-frames:v', row['frames'], '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-threads', '4', '-pix_fmt', 'yuv420p', '-video_track_timescale', '24000', '-c:a', 'pcm_s16le', '-ar', '48000', '-ac', '2', '-t', f"{row['frames']/24:.9f}", temp]
                run(cmd)
                temp.replace(output)
                save(receipt, {'id': row['id'], 'sha256': sha(output), 'frames': row['frames'], 'plannedAudioSamples': samples, 'sourceCardSha256': sha(source_card), 'finalApproved': False})
            outputs.append(output)
            print(json.dumps({'id': row['id'], 'status': 'COMPLETED'}), flush=True)
        listing = root / 'segments.ffconcat'
        if any("'" in str(p) or '\n' in str(p) for p in outputs):
            raise ValueError('Unsafe concat path')
        listing.write_text('ffconcat version 1.0\n' + ''.join(f"file '{p}'\n" for p in outputs))
        final = root / 'timed-animatic-v1.mp4'
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
                speech.append({'id': row['id'], 'sourceSha256': row['audio']['sha256'], 'startSample': start, 'completeDecodedSamples': len(samples), 'retimed': False, 'trimmed': False})
            cursor += row['frames']
        voice = root / 'complete-narration-clock.wav'
        write_audio(voice, stem)
        del stem
        save(root / 'voice-clock-audit.json', {'samples': frames*2000, 'sampleRate': 48000, 'sha256': sha(voice), 'events': speech, 'scratchDialogue': 'silent', 'source': 'whole original narration, not segment audio'})
        # Reset picture PTS from decoded frame number; fractional container
        # duration rounding must not accumulate at the150 clip boundaries.
        run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', listing, '-i', voice, '-map', '0:v:0', '-map', '1:a:0', '-vf', 'setpts=N/(24*TB)', '-r', '24', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-threads', '4', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-t', f'{frames/24:.9f}', '-movflags', '+faststart', temp])
        run(['ffmpeg', '-v', 'error', '-xerror', '-i', temp, '-f', 'null', '-'])
        probe = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', temp]).stdout)
        v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        a = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        if int(v['nb_frames']) != frames or abs(float(v['duration']) - frames/24) > .045 or abs(float(a['duration'])-frames/24) > .065:
            raise ValueError('Animatic clock mismatch')
        if sha(manifest) != plan_hash:
            raise ValueError('Manifest changed during rendering')
        temp.replace(final)
        audit = {'kind': plan['kind'], 'finalApproved': False, 'manifestSha256': plan_hash, 'toolSha256': sha(__file__), 'host': platform.node(), 'encoder': 'libx264 on qualified GPU worker; no Mac processing', 'output': str(final), 'sha256': sha(final), 'scenes': len(outputs), 'frames': frames, 'durationSeconds': frames/24, 'fullDecode': 'passed', 'narrationCount': sum(s['mode']=='narration' for s in plan['scenes']), 'scratchDialogueCount': sum(s['mode']=='scratch-dialogue' for s in plan['scenes']), 'limitations': 'Silent dialogue slots use authored temporary timings. Still holds and known source defects are labeled. This is not final motion or a hearing/temporal approval.'}
        audit.update(reusedPictureToolSha256=args.reuse_picture_tool, voiceClockAuditSha256=sha(root / 'voice-clock-audit.json'))
        save(root / 'render-audit.json', audit)
        print(json.dumps(audit), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
