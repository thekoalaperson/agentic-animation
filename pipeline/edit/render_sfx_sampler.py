"""Render a labelled, source-bound sound-only sampler on the qualified GPU host."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    return subprocess.run([str(x) for x in args], check=True, capture_output=True).stdout


def render_layered(ep, manifest, plan):
    """Audition authored composites alone and against identical voice/score controls."""
    import math
    import re
    import numpy as np
    from render_designed_sfx import decode, save

    digest = sha(manifest)
    out = (ep / plan['outputDirectory']).resolve()
    out.relative_to(ep / 'work')
    out.mkdir(parents=True, exist_ok=True)
    final = out / plan['videoFile']
    assert final.parent == out and not final.exists()
    assert plan['finalApproved'] is False and len(plan['items']) == plan['expectedItems']
    sr = 48000
    font = Path(plan['font'])
    assert font.is_file()
    bindings, sections, records = {}, [], []

    def source(spec):
        p = (ep / spec['source']).resolve()
        p.relative_to(ep)
        assert sha(p) == spec['sha256']
        bindings[str(p)] = spec['sha256']
        return p

    def fitted(a, n):
        return np.pad(a[:n], ((0, max(0, n-len(a))), (0, 0)))

    def fades(a, fi, fo):
        a = a.copy()
        i, o = min(round(fi*sr), len(a)//2), min(round(fo*sr), len(a)//2)
        if i: a[:i] *= np.linspace(0, 1, i)[:, None]
        if o: a[-o:] *= np.linspace(1, 0, o)[:, None]
        return a

    def level(a, db):
        usable = len(a)//4800*4800
        rms = float(np.percentile(np.sqrt(np.mean(a[:usable].reshape(-1, 4800, 2).astype(float)**2, axis=(1, 2))), 90))
        assert rms > 1e-7
        gain = float(np.clip(db - 20*math.log10(rms), -24, 12))
        return a * 10**(gain/20), gain

    for index, item in enumerate(plan['items'], 1):
        assert re.fullmatch(r'[a-z0-9-]+', item['id'])
        frames = round(item['seconds']*24)
        n = frames*2000
        stem = np.zeros((n, 2), np.float32)
        layers = []
        for layer in item['layers']:
            p = source(layer)
            start, end = layer['sourceIn'], layer['sourceOut']
            assert 0 <= start < end and 0.5 <= layer['speed'] <= 2
            ratio = 2**(layer['semitones']/12)
            filters = [f'aresample={sr}', f'atrim=start={start}:end={end}', 'asetpts=PTS-STARTPTS',
                       f'asetrate={round(sr*ratio)}', f'aresample={sr}']
            tempo = layer['speed']/ratio
            while tempo < .5: filters.append('atempo=0.5'); tempo /= .5
            while tempo > 2: filters.append('atempo=2'); tempo /= 2
            filters += [f'atempo={tempo:.12f}', f"highpass=f={layer.get('highpass',35)}",
                        f"lowpass=f={layer.get('lowpass',18000)}"]
            a = decode(p, ','.join(filters))
            expected = round((end-start)/layer['speed']*sr)
            assert abs(len(a)-expected) < sr*.2
            a = fitted(a, expected)
            a, gain = level(a, layer['activeRmsDb'])
            at = round(layer['at']*sr)
            assert 0 <= at < n
            a = fades(a[:n-at], layer.get('fadeIn',.015), layer.get('fadeOut',.4))
            stem[at:at+len(a)] += a
            layers.append({**layer, 'actualStart':at/sr, 'actualEnd':(at+len(a))/sr,
                           'levelGainDb':gain, 'filters':filters})
        raw = out / (item['id']+'-sfx.wav')
        save(raw, stem)
        records.append({'id':item['id'], 'stemSha256':sha(raw), 'layers':layers})
        sections.append(dict(item=item, mode='DESIGNED SFX ONLY', audio=stem,
                             frames=frames, layers=layers, picture=None))
        if item.get('context'):
            ctx = item['context']
            picture = source(ctx['picture'])
            probe = json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',picture]))
            assert len(probe['streams']) == 1 and probe['streams'][0]['codec_type']=='video'
            assert float(probe['streams'][0]['duration']) >= frames/24
            voice = decode(source(ctx['voice']))
            assert len(voice) <= n, 'Never truncate a Hindi performance for a sampler'
            voice = fitted(voice, n)
            score = decode(source(ctx['score']), f"atrim=start={ctx['scoreIn']}:duration={frames/24},asetpts=PTS-STARTPTS")
            score, score_gain = level(fitted(score,n), ctx['scoreActiveRmsDb'])
            score = fades(score, .5, .7)
            control = voice+score
            added = stem*10**(ctx['sfxGainDb']/20)
            # Gentle speech protection on effects only; no voice processing or score pumping.
            block = 480
            envelope = np.sqrt(np.mean(fitted(voice, math.ceil(n/block)*block).reshape(-1,block,2).astype(float)**2,axis=(1,2)))
            protect = np.clip(envelope/.045,0,1)
            protect = np.convolve(protect,np.ones(21)/21,mode='same')
            added *= np.repeat(10**(-ctx.get('speechSfxDipDb',3)*protect/20),block)[:n,None]
            for mode, a in [('VOICE + SCORE / CONTROL',control),('VOICE + SCORE + DESIGNED SFX',control+added)]:
                sections.append(dict(item=item, mode=mode, audio=a, frames=frames,
                                     layers=layers if 'DESIGNED' in mode else [], picture=picture))
            records[-1]['context']={**ctx,'scoreGainDb':score_gain,'voiceProcessing':'None except shared final playback gain; complete unretimed performance'}
    peak = max(float(np.max(np.abs(s['audio']))) for s in sections)
    gain = min(1,10**(-2/20)/max(peak,1e-9))
    clips, all_audio, cursor, segments = [], [], 0, []
    for i, s in enumerate(sections,1):
        item = s['item']; audio_path = out/f'section-{i:02}-audio.wav'
        save(audio_path,s['audio']*gain)
        texts=[f"{i:02}/{len(sections):02}  |  {item['label']}",s['mode'],item['description']]
        files=[]
        for j,t in enumerate(texts):
            p=out/f'section-{i:02}-label-{j}.txt';p.write_text(t);files.append(p)
        command=['ffmpeg','-v','error','-y','-i',audio_path]
        if s['picture']:
            command+=['-i',s['picture']]
            graph='[1:v]scale=1600:900:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:130:color=0x111823,setsar=1[bg];'
        else:
            command+=['-f','lavfi','-i',f"color=c=0x111823:s=1920x1080:r=24:d={s['frames']/24}"]
            graph='[0:a]showwaves=s=1680x360:mode=cline:rate=24:colors=0x68CBC3:scale=sqrt[w];[1:v][w]overlay=120:360[bg];'
        filters=[]
        for p,size,y,color in [(files[0],42,28,'0xF0D3A0'),(files[1],26,88,'white'),(files[2],24,1040,'0xC0CDD9')]:
            filters.append(f'drawtext=fontfile={font}:textfile={p}:expansion=none:fontsize={size}:fontcolor={color}:x=(w-text_w)/2:y={y}')
        for j,layer in enumerate(s['layers']):
            p=out/f'section-{i:02}-layer-{j}.txt';p.write_text(layer['label'])
            filters.append(f"drawtext=fontfile={font}:textfile={p}:expansion=none:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65:x=30:y={180+j*37}:enable='between(t,{layer['actualStart']},{layer['actualEnd']})'")
        graph+='[bg]'+','.join(filters)+'[v]'
        clip=out/f'section-{i:02}.mp4'
        run(command+['-filter_complex_threads','2','-filter_complex',graph,'-map','[v]','-an','-frames:v',s['frames'],
                     '-r','24','-c:v','libx264','-preset','veryfast','-crf','21','-threads','4','-pix_fmt','yuv420p',clip])
        clips.append(clip);all_audio.append(s['audio']*gain)
        segments.append(dict(index=i,id=item['id'],mode=s['mode'],startSeconds=cursor/24,seconds=s['frames']/24))
        cursor+=s['frames'];print(json.dumps(segments[-1]),flush=True)
    listing=out/'sampler.ffconcat';listing.write_text('ffconcat version 1.0\n'+''.join(f"file '{p}'\n" for p in clips))
    sound=out/'complete-sampler-audio.wav';save(sound,np.concatenate(all_audio))
    run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',listing,'-i',sound,'-map','0:v:0','-map','1:a:0',
         '-c:v','copy','-c:a','aac','-b:a','256k','-ar','48000','-movflags','+faststart',final])
    run(['ffmpeg','-v','error','-xerror','-i',final,'-f','null','-'])
    probe=json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',final]))
    v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
    assert int(v['nb_read_frames'])==cursor and (v['width'],v['height'])==(1920,1080)
    assert abs(float(a['duration'])-cursor/24)<.04
    assert sha(manifest)==digest and all(sha(Path(p))==h for p,h in bindings.items())
    run(['ffmpeg','-v','error','-y','-ss','12','-i',final,'-frames:v','1',out/'preview.jpg'])
    audit=dict(manifestSha256=digest,rendererSha256=sha(__file__),outputSha256=sha(final),seconds=cursor/24,frames=cursor,
               fullDecode='passed',commonPlaybackGainDb=20*math.log10(gain),preGainPeak=peak,sections=segments,items=records,
               finalApproved=False,listeningVerdict='Pending user; assistant audio perception unavailable; no listening claim',sources=bindings)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'video':str(final),'seconds':cursor/24,'sha256':sha(final)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode', type=Path)
    ap.add_argument('manifest', type=Path)
    args = ap.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('Sampler media processing belongs on the qualified GPU host')
    run(['nvidia-smi', '-L'])
    ep = args.episode.resolve()
    manifest = args.manifest.resolve()
    manifest.relative_to(ep)
    plan = json.loads(manifest.read_text())
    if plan.get('mode') == 'layered-context':
        return render_layered(ep, manifest, plan)
    digest = sha(manifest)
    out = (ep / plan['outputDirectory']).resolve()
    out.relative_to(ep / 'work')
    out.mkdir(parents=True, exist_ok=True)
    filename = plan.get('videoFile', 'SFX_SAMPLER_V1.mp4')
    if Path(filename).name != filename or not filename.endswith('.mp4'):
        raise ValueError('Sampler filename must be a plain MP4 basename')
    final = out / filename
    if final.exists():
        raise RuntimeError('Preserve earlier sampler; choose a new output directory')
    font = Path(plan['font']).resolve()
    assert font.is_file() and len(plan['items']) == plan['expectedItems']
    gain = float(plan['commonGainDb'])
    assert -6 <= gain <= 0
    sources = []
    for row in plan['items']:
        source = (ep / row['source']).resolve()
        source.relative_to(ep)
        assert sha(source) == row['sha256']
        sources.append(source)
    clips, records = [], []
    cursor = 0
    for index, (row, source) in enumerate(zip(plan['items'], sources), 1):
        probe = json.loads(run(['ffprobe', '-v', 'error', '-show_format', '-of', 'json', source]))
        seconds = float(probe['format']['duration'])
        assert abs(seconds-row['seconds']) < 1/44100
        frames = round((seconds+1)*24)
        samples = frames*2000
        texts = [plan['title'], f'{index:02d} / {len(sources):02d}', row['label'], row['description'],
                 'RAW GENERATIONS  |  SAME PLAYBACK GAIN  |  SOUND ONLY']
        files = []
        for j, text in enumerate(texts):
            p = out / f'label-{index:02d}-{j}.txt'
            p.write_text(text, encoding='utf-8')
            files.append(p)
        for p in [font, *files]:
            assert not any(c in str(p) for c in "':,[]\\")
        graph = (f'[0:a]aresample=48000,volume={gain}dB,adelay=12000S:all=1,'
                 f'apad=whole_len={samples},atrim=end_sample={samples},asetpts=N/SR/TB,asplit=2[a][av];'
                 '[av]showwaves=s=1680x220:mode=cline:rate=24:colors=0x68CBC3:scale=sqrt[viz];'
                 '[1:v][viz]overlay=120:640:shortest=1[base];')
        specs = [(30,'0x96A6B7','100','78'),(30,'0x96A6B7','w-text_w-100','78'),
                 (80,'0xF0D3A0','(w-text_w)/2','280'),(34,'0xD1DBE6','(w-text_w)/2','412'),
                 (24,'0x96A6B7','(w-text_w)/2','970')]
        filters = []
        for p,(size,color,x,y) in zip(files,specs):
            filters.append(f'drawtext=fontfile={font}:textfile={p}:expansion=none:fontsize={size}:fontcolor={color}:x={x}:y={y}')
        graph += '[base]'+','.join(filters)+'[v]'
        clip = out/f'segment-{index:02d}.mov'
        run(['ffmpeg','-v','error','-y','-i',source,'-f','lavfi','-i',
             f'color=c=0x111823:s=1920x1080:r=24:d={frames/24}',
             '-filter_complex',graph,'-map','[v]','-map','[a]','-frames:v',frames,
             '-c:v','libx264','-preset','veryfast','-crf','21','-threads','4','-pix_fmt','yuv420p',
             '-video_track_timescale','24000','-c:a','pcm_s24le','-ar','48000','-ac','2',
             '-t',f'{frames/24:.9f}',clip])
        clips.append(clip)
        records.append({**row,'index':index,'startSeconds':cursor/24,'soundStartSeconds':cursor/24+.25,
                        'frames':frames,'commonGainDb':gain,'preSilenceSeconds':.25,
                        'postSilenceSeconds':.75,'segmentSha256':sha(clip)})
        cursor += frames
    listing = out/'sampler.ffconcat'
    listing.write_text('ffconcat version 1.0\n'+''.join(f"file '{p}'\n" for p in clips))
    # Build sound independently: a segment's video-frame limit can stop its
    # audio encoder early. Complete source audio owns the sampler clock.
    sound = out/'complete-sampler-audio.wav'
    command = ['ffmpeg','-v','error','-y']
    chains = []
    for i,(source,row) in enumerate(zip(sources,records)):
        command += ['-i',source]
        chains.append(f'[{i}:a]aresample=48000,volume={gain}dB,adelay=12000S:all=1,'
                      f'apad=whole_len={row["frames"]*2000},atrim=end_sample={row["frames"]*2000},'
                      f'asetpts=N/SR/TB[a{i}]')
    graph = ';'.join(chains)+';'+''.join(f'[a{i}]' for i in range(len(sources)))
    graph += f'concat=n={len(sources)}:v=0:a=1[all]'
    run(command+['-filter_complex',graph,'-map','[all]','-ar','48000','-ac','2','-c:a','pcm_s24le',sound])
    run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',listing,'-i',sound,'-map','0:v:0','-map','1:a:0',
         '-c:v','copy','-c:a','aac','-b:a','256k','-ar','48000','-movflags','+faststart',final])
    run(['ffmpeg','-v','error','-xerror','-i',final,'-f','null','-'])
    probe = json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',final]))
    video = next(s for s in probe['streams'] if s['codec_type']=='video')
    audio = next(s for s in probe['streams'] if s['codec_type']=='audio')
    assert int(video['nb_frames'])==cursor and (video['width'],video['height'])==(1920,1080)
    assert abs(float(audio['duration'])-cursor/24)<.06
    assert sha(manifest)==digest and all(sha(p)==r['sha256'] for p,r in zip(sources,plan['items']))
    run(['ffmpeg','-v','error','-y','-ss','2','-i',final,'-frames:v','1',out/'preview.jpg'])
    report={'manifestSha256':digest,'rendererSha256':sha(__file__),'fontSha256':sha(font),
            'outputSha256':sha(final),'frames':cursor,'seconds':cursor/24,'resolution':[1920,1080],
            'fullDecode':'passed','items':records,'normalization':'none; one uniform headroom gain',
            'sourceEditing':'none; complete originals at native speed with silence between samples',
            'listeningVerdict':'pending user sample test; no model listening claimed'}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'video':str(final),'seconds':cursor/24,'items':len(records),'sha256':sha(final)}))


if __name__ == '__main__':
    main()
