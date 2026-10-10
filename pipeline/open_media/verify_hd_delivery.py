#!/usr/bin/env python3
"""Read-only local verification of assembled long masters and still-led Shorts.

Does not promote, publish, approve pictures, restart jobs, or modify media.
Run after copy-back. Missing assets remain failures, never inferred successes.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time

if __package__:
    from .audit_hd_sources import sha
else:
    from audit_hd_sources import sha


def output(*cmd):
    return subprocess.check_output([str(x) for x in cmd], text=True)


def stream_hash(path, stream):
    return output('ffmpeg', '-v', 'error', '-i', path, '-map', stream,
                  '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-').strip()


def verify_media(path, expected_hash, size, frames, fps, duration):
    if sha(path) != expected_hash:
        raise ValueError('Local file differs from remote audit hash')
    streams = json.loads(output('ffprobe', '-v', 'error', '-threads', '2',
                               '-count_frames', '-show_streams', '-of', 'json', path))['streams']
    video = [s for s in streams if s['codec_type'] == 'video']
    audio = [s for s in streams if s['codec_type'] == 'audio']
    if len(video) != 1 or len(audio) != 1 or len(streams) != 2:
        raise ValueError('Expected exactly one picture and one audio stream')
    v, a = video[0], audio[0]
    if [v['width'], v['height']] != size or Fraction(v['avg_frame_rate']) != fps:
        raise ValueError('Resolution or cadence differs from delivery contract')
    if int(v['nb_read_frames']) != frames or abs(float(v['duration'])-duration) > .001:
        raise ValueError('Frame count or picture duration differs from approved timeline')
    if a['codec_name'] != 'aac' or int(a['sample_rate']) != 48000 or a['channels'] != 2:
        raise ValueError('Expected 48kHz stereo AAC')
    if abs(float(a['duration'])-duration) > .1:
        raise ValueError('Audio/picture duration mismatch')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '2', '-i', str(path),
                    '-f', 'null', '-'], check=True)
    return {'file':str(path), 'sha256':expected_hash, 'size':size, 'frames':frames,
            'fps':fps, 'duration':duration, 'decode':'passed', 'audio':'48kHz stereo AAC'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('batch', type=Path)
    args = ap.parse_args()
    root = args.batch.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    results, failures = {}, {}
    for lang, timeline in manifest['timelines'].items():
        try:
            audit = json.loads((root/f'renders/audit-{lang}.json').read_text())
            path = (root/audit['output']).resolve()
            if not path.is_relative_to(root/'renders'):
                raise ValueError('Master path outside render directory')
            if audit.get('fallback') is not False:
                raise ValueError('Neural/no-fallback evidence missing')
            frames = sum(s['frames']*2 for s in timeline['shots'])
            item = verify_media(path, audit['sha256'], [1920,1080], frames, 48, timeline['duration'])
            original = root/manifest['masters'][lang]['source']
            if sha(original) != manifest['masters'][lang]['sha256']:
                raise ValueError('Approved source master changed')
            original_audio = stream_hash(original, '0:a:0')
            if stream_hash(path, '0:a:0') != original_audio:
                raise ValueError('Approved AAC soundtrack changed')
            item['audioBitstreamHash'] = original_audio
            results['long-'+lang] = item
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            failures['long-'+lang] = str(exc)
    try:
        plan_path = root/'renders/shorts/plan.json'
        plan = json.loads(plan_path.read_text())
        candidates=[root/'shorts-stills-v3/plan.json',root/'shorts-stills-v2/plan.next.json',
                    root/'shorts-stills-v2/plan.json']
        if sha(plan_path) not in {sha(p) for p in candidates if p.exists()}:
            raise ValueError('Returned Short plan differs from staged selection')
        for lang, timeline in plan['timelines'].items():
            audit = json.loads((root/f'renders/shorts/audit-{lang}.json').read_text())
            if audit['planSha256'] != sha(plan_path) or audit['shots'] != [1,2,3,5,6,7,8]:
                raise ValueError('Short source plan mismatch')
            if len(audit['reports']) != 2:
                raise ValueError('Expected YouTube and Reels mixes')
            for report in audit['reports']:
                path = (root/'renders/shorts'/report['file']).resolve()
                if path.parent != root/'renders/shorts':
                    raise ValueError('Short path outside render directory')
                key = 'short-'+lang+('-social' if path.stem.endswith('_social') else '')
                item = verify_media(path, report['sha256'], [1080,1920],
                                    sum(s['frames'] for s in timeline['shots']),48,timeline['duration'])
                target = -10 if path.stem.endswith('_social') else -14
                proc = subprocess.run(['ffmpeg','-v','info','-i',str(path),'-vn','-af',
                    f'loudnorm=I={target}:TP=-1.2:LRA=11:print_format=json','-f','null','-'],
                    check=True,capture_output=True,text=True)
                measured = json.loads(proc.stderr[proc.stderr.rfind('{'):proc.stderr.rfind('}')+1])
                if abs(float(measured['input_i'])-target)>.5 or float(measured['input_tp'])>-.8:
                    raise ValueError('Short loudness/peak outside contract')
                item['loudness'] = measured
                results[key] = item
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        failures['shorts'] = str(exc)
    report = {'time':time.time(), 'manifestSha256':sha(root/'manifest.json'),
              'technicalPassed':len(results)==6 and not failures, 'media':results,
              'failures':failures, 'scope':'Six motion masters only. Does not certify creative/temporal approval, captions, artwork, metadata, mirrors or VM shutdown.'}
    (root/'local-final-media-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('technicalPassed','failures')},indent=2))
    if not report['technicalPassed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
