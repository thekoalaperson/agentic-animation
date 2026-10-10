#!/usr/bin/env python3
"""GPU-host-only, source-bound sound auditions. Never mutates final masters."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

SR = 48000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, capture_output=True, **kwargs)


def read(path):
    return json.loads(Path(path).read_text())


def decode(path, start=0, duration=None, filters=None):
    args = ['ffmpeg', '-v', 'error', '-ss', str(start), '-i', path]
    if duration is not None:
        args += ['-t', str(duration)]
    if filters:
        args += ['-af', filters]
    args += ['-ar', str(SR), '-ac', '2', '-f', 'f32le', '-']
    a = np.frombuffer(run(args).stdout, dtype='<f4').reshape(-1, 2).copy()
    assert len(a) and np.isfinite(a).all(), path
    return a


def write_wav(path, data):
    assert len(data) and np.isfinite(data).all()
    run(['ffmpeg', '-y', '-v', 'error', '-f', 'f32le', '-ar', SR, '-ac', 2,
         '-i', 'pipe:0', '-c:a', 'pcm_f32le', path], input=data.astype('<f4').tobytes())


def block_rms(data, block=4800):
    count = len(data) // block
    assert count > 0
    return np.sqrt(np.mean(data[:count*block].reshape(count, block, 2).astype(float)**2, axis=(1, 2)))


def active_level(data):
    return max(float(np.percentile(block_rms(data), 80)), 1e-7)


def fit_length(data, n, tolerance=48):
    if len(data) < n:
        assert n-len(data) <= tolerance, ('source shortage', len(data), n)
        return np.pad(data, ((0, n-len(data)), (0, 0)))
    return data[:n]


def peaks(data, bins=480):
    step = math.ceil(len(data)/bins)
    return [round(float(np.max(np.abs(data[i:i+step]))), 5) for i in range(0, len(data), step)]


def packet_hash(path):
    return run(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:v:0', '-c', 'copy',
                '-f', 'hash', '-hash', 'sha256', '-']).stdout.decode().strip()


def build(ep, root, plan_path):
    if platform.system() != 'Linux':
        raise RuntimeError('Media processing requires the qualified GPU worker, not the Mac')
    run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    artifacts = {'plan': plan_path, 'runner': Path(__file__).resolve(),
        'template': Path(__file__).with_name('sound_revision_review_template.html'),
        'musicManifest': root/'music-pilot/execution-manifest.json',
        'musicState': root/'music-pilot/execution-manifest.state.json',
        'sfxAudit': root/'sfx-pilot/batch-audit.json',
        'sfxSpeech': root/'sfx-pilot/speech-audit-base.json',
        'sfxBrief': root/'sfx-pilot/probe-brief.json'}
    frozen = {k: sha(p) for k, p in artifacts.items()}
    used_sources = {}
    plan = read(plan_path)
    assert plan['finalApproved'] is False and plan['filmMutationAllowed'] is False
    inputs = {k: (ep / v['path']).resolve() for k, v in plan['inputs'].items()}
    for k, p in inputs.items():
        assert p.is_relative_to(ep) and sha(p) == plan['inputs'][k]['sha256'], k
    clock = read(inputs['clock'])
    events = {e['shot']: e for e in clock['events']}
    manifest = root / 'music-pilot/execution-manifest.json'
    state = read(manifest.with_suffix('.state.json'))
    assert state['executionManifestSha256'] == frozen['musicManifest']
    assert all(r['status'] == 'DOWNLOADED' for r in state['items'].values())
    music_cues = {c['id']: c for c in read(manifest)['cues']}
    sfx_root = root / 'sfx-pilot'
    sfx_audit = read(sfx_root / 'batch-audit.json')
    sfx_rows = {r['job']['family']: r for r in sfx_audit['outputs']}
    assert sfx_audit['runtime']['briefSha256'] == sha(sfx_root/'probe-brief.json')
    speech = {r['file']: r for r in read(sfx_root/'speech-audit-base.json')['results']}
    out = root / plan['outputDirectory']
    out.mkdir(parents=True, exist_ok=False)
    scenes = []
    raw_sfx = []
    for family, row in sfx_rows.items():
        p = sfx_root/row['job']['output']
        assert sha(p) == row['sha256']
        assert speech[p.name]['sha256'] == row['sha256'], 'Stale SFX speech screen'
        used_sources[p] = row['sha256']
        raw_sfx.append(dict(id=family, title=family.replace('-', ' ').title(),
            url=os.path.relpath(p, out), sha256=sha(p), prompt=row['job']['prompt'],
            duration=row['seconds'], peaks=peaks(decode(p)),
            warning='Tonal/music-like classifier warning; excluded from scene mixes.' if family in plan['excludedSfx'] else '',
            speechScreen='flagged' if speech[p.name]['quarantine'] else 'no credible speech detected; not a listening verdict'))
    for scene in plan['scenes']:
        lo, hi = scene['shots']
        start, end = events[lo]['start'], events[hi]['end']
        frames = round((end-start)*24)
        seconds, n = frames/24, frames*2000
        v = fit_length(decode(inputs['voice'], start, seconds), n)
        old = fit_length(decode(inputs['oldPremaster'], start, seconds), n)
        fx = np.zeros_like(v)
        effect_records = []
        for effect in scene['effects']:
            family = effect['family']
            assert family not in plan['excludedSfx']
            source = sfx_rows[family]
            path = sfx_root/source['job']['output']
            assert not speech[path.name]['quarantine']
            ratio = 2**(effect['semitones']/12)
            filters = f"aresample=48000,asetrate={round(SR*ratio)},aresample=48000,atempo={effect['speed']/ratio},highpass=f=35"
            a = decode(path, filters=filters)
            # Find the first substantial 10ms onset, preserving a 10ms handle.
            rms = block_rms(a, 480)
            active = np.flatnonzero(rms > max(np.max(rms)*.07, 1e-6))
            trim = max(0, int(active[0])*480-480) if len(active) else 0
            a = a[trim:]
            gain = min(10**(effect['targetRmsDb']/20)/active_level(a),
                       10**(effect['peakCapDb']/20)/max(float(np.max(np.abs(a))), 1e-7))
            a *= gain
            fade = min(2400, len(a)//4)
            a[:240] *= np.linspace(0, 1, 240)[:, None]
            a[-fade:] *= np.linspace(1, 0, fade)[:, None]
            at = round((events[effect['shot']]['start']-start+effect['offset'])*SR)
            assert 0 <= at < n
            count = min(n-at, len(a))
            fx[at:at+count] += a[:count]
            effect_records.append({**effect, 'sourceSha256': source['sha256'],
                'sourceLeadingTrimSeconds': trim/SR, 'atSeconds': at/SR,
                'durationSeconds': count/SR, 'filter': filters, 'gain': gain,
                'syncStatus': 'authored scene-relative audition anchor; not final frame-approved sync'})
        cue = music_cues[scene['cue']]
        candidates = state['items'][scene['cue']]['candidates']
        assert len(candidates) == 2
        versions = [('old', 'Old / rejected', old, None)]
        fx_rms = block_rms(fx)
        effect_env = np.interp(np.arange(n)/SR, np.arange(len(fx_rms))*.1+.05,
                               np.where(fx_rms > .015, .66, 1.0))
        vr = block_rms(v)
        for take, record in enumerate(candidates, 1):
            path = Path(record['file'])
            assert sha(path) == record['sha256']
            used_sources[path] = record['sha256']
            data = decode(path, filters='highpass=f=30,equalizer=f=2400:t=q:w=0.7:g=-3')
            gain = 10**(scene['musicActiveRmsDb']/20)/active_level(data)
            data *= gain
            source_start = round(scene['musicSourceStart']*SR)
            assert source_start+n <= len(data), 'Music source cannot fill this preview; author a different window'
            m = data[source_start:source_start+n].copy()
            fade = min(24000, n//4)
            m[:fade] *= np.linspace(0, 1, fade)[:, None]
            m[-fade:] *= np.linspace(1, 0, fade)[:, None]
            mr = block_rms(m)
            protection = np.where(vr > .018, np.minimum(1, vr*.65/np.maximum(mr, 1e-7)), 1)
            protection = np.interp(np.arange(n)/SR, np.arange(len(protection))*.1+.05, protection)
            m *= (protection*effect_env)[:, None]
            versions.append((str(take), 'New '+('A' if take == 1 else 'B'), v+m+fx,
                dict(url=os.path.relpath(path, out), sha256=record['sha256'],
                     prompt=cue['payload']['style'], sourceStart=source_start/SR,
                     sourceEnd=(source_start+n)/SR, gain=gain,
                     mixTreatment='Whole-source active level, 2.4kHz -3dB space, gentle speech-window protection and authored effect handoffs; no voice retime')))
        common_gain = min(1, .84/max(float(np.max(np.abs(x[2]))) for x in versions))
        proxy = out/(scene['id']+'-picture.mp4')
        run(['ffmpeg', '-y', '-v', 'error', '-ss', start, '-i', inputs['picture'],
             '-an', '-vf', 'setpts=PTS-STARTPTS', '-frames:v', frames,
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', proxy])
        picture_hash = packet_hash(proxy)
        variant_rows = []
        for key, label, mixed, music in versions:
            wav = out/(scene['id']+'-'+key+'.wav')
            mp4 = out/(scene['id']+'-'+key+'.mp4')
            mixed = mixed*common_gain
            write_wav(wav, mixed)
            run(['ffmpeg', '-y', '-v', 'error', '-i', proxy, '-i', wav,
                 '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
                 '-b:a', '256k', '-ar', SR, '-t', seconds, '-movflags', '+faststart', mp4])
            run(['ffmpeg', '-v', 'error', '-i', mp4, '-f', 'null', '-'])
            assert packet_hash(mp4) == picture_hash
            variant_rows.append(dict(key=key, label=label, url=mp4.name,
                sha256=sha(mp4), wave=peaks(decode(mp4)), music=music,
                samplePeak=float(np.max(np.abs(mixed))), commonGain=common_gain))
        scenes.append(dict(id=scene['id'], title=scene['title'], intent=scene['intent'],
            start=start, end=end, duration=seconds, shots=[lo, hi],
            text='\n'.join(events[s]['text'] for s in range(lo, hi+1)),
            lines=[dict(shot=s, start=events[s]['start']-start,
                        end=events[s]['end']-start, text=events[s]['text'])
                   for s in range(lo, hi+1)],
            effects=effect_records, variants=variant_rows, picturePacketSha256=picture_hash))
        print(json.dumps({'scene':scene['id'], 'seconds':seconds, 'versions':3}), flush=True)
    for k, p in artifacts.items():
        assert sha(p) == frozen[k], ('Execution artifact changed during render', k)
    for p, expected in used_sources.items():
        assert sha(p) == expected, ('Source changed during render', str(p))
    output = dict(episode=plan['episode'], version=2, scenes=scenes, sounds=raw_sfx,
        scope='Listening prototypes, not final film. Old arrangement comes from its exact premaster. Identical voice gain and identical picture packets across alternatives; common peak protection, not loudness-normalized preference scoring.',
        inputHashes={k:sha(p) for k,p in inputs.items()}, planSha256=frozen['plan'],
        runnerSha256=frozen['runner'], musicManifestSha256=frozen['musicManifest'],
        executionArtifactHashes=frozen,
        filmChanged=False, creativeApproval=False)
    (out/'review-data.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    template = Path(__file__).with_name('sound_revision_review_template.html').read_text()
    (out/'SOUND_REVISION_REVIEW.html').write_text(template.replace('/*__DATA__*/null',
        json.dumps(output, ensure_ascii=False).replace('<', '\\u003c')))
    for k, p in inputs.items():
        assert sha(p) == plan['inputs'][k]['sha256']
    for k, p in artifacts.items():
        assert sha(p) == frozen[k], ('Execution artifact changed before closeout', k)
    print(json.dumps({'page':str(out/'SOUND_REVISION_REVIEW.html'),
                      'scenes':len(scenes),'sounds':len(raw_sfx),'filmChanged':False}),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode', type=Path)
    ap.add_argument('root', type=Path)
    ap.add_argument('plan', type=Path)
    a = ap.parse_args()
    build(a.episode.resolve(), a.root.resolve(), a.plan.resolve())
