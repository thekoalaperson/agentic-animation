#!/usr/bin/env python3
"""Render source-bound complete-line motion Shorts; native type is a separate step."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
try:
    from .short_audio_contract import question_text, cta_filter, prepared_audio, gap_frames
except ImportError:
    from short_audio_contract import question_text, cta_filter, prepared_audio, gap_frames


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def run(cmd, **kw):
    return subprocess.run([str(x) for x in cmd],check=True,capture_output=True,**kw)


def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode',type=Path);ap.add_argument('spec',type=Path)
    ap.add_argument('--gpu-type-only',action='store_true')
    ap.add_argument('--fonts',type=Path,help='Explicit font directory on the GPU worker')
    a=ap.parse_args()
    ep=a.episode.resolve();sp=(ep/a.spec).resolve();sp.relative_to(ep/'work')
    spec=json.loads(sp.read_text()); spec_sha=sha(sp);root=sp.parent
    assert spec['kind']=='authored-motion-short' and spec['fps']==24
    if spec['language'] != 'hi':
        raise ValueError('This authored Short preset is qualified for Hindi')
    if (not isinstance(spec.get('channelHandle'), str) or not spec['channelHandle'].startswith('@')
            or not isinstance(spec.get('channelName'), str) or not spec['channelName'].strip()):
        raise ValueError('Explicit channelHandle and channelName are required')
    assert sum(b['frames']+gap_frames(b) for b in spec['beats'])==spec['bodyFrames']
    if any(gap_frames(b) for b in spec['beats']):
        assert spec.get('audioMode')=='prepared-whole-short', 'Authored gaps need source-bound prepared audio'
    assert sum(p['frames'] for p in spec['picture'])==spec['bodyFrames']
    assert spec['bodyFrames']+spec['ctaFrames']==spec['totalFrames']
    raster_dir=root/'native-type';raster_dir.mkdir(exist_ok=True)
    if a.gpu_type_only:
        try:
            from .native_text_gpu import raster
        except ImportError:
            from native_text_gpu import raster
        rasterizer='GPU host libass/HarfBuzz complex shaping; 2x; per-raster geometry/font audit and actual visual review required'
        type_options = dict(fonts_dir=a.fonts,channel_name=spec['channelName'],
            channel_handle=spec['channelHandle'],latin_font=spec.get('latinFont','Noto Serif'),
            devanagari_font=spec.get('devanagariFont','Noto Serif Devanagari'))
        assert not (raster_dir/'manifest.json').exists(), 'Preserve completed native rasters'
        rows=[]
        for beat in spec['beats']:
            parts=beat['captionParts'];assert ' '.join(parts)==beat['spokenTextHi']
            weights=[len(p.replace(' ','')) for p in parts];total=sum(weights)
            t=beat['startFrame']/24
            for part,w in zip(parts,weights):
                end=t+beat['audioDuration']*w/total
                assert end-t>=1, 'Do not flash isolated words'
                path=raster_dir/f'caption-{len(rows):02d}-2x.png'
                raster('caption-safe',path,2160,3840,part,**type_options)
                rows.append(dict(text=part,start=t,end=end,path=str(path.relative_to(root)),sha256=sha(path)))
                t=end
        fixed=[]
        fixed_specs=[('channel-cta',spec['ctaText'],'cta')]
        if question_text(spec) is not None:
            fixed_specs.insert(0,('short-question',question_text(spec),'question'))
        for mode,text,name in fixed_specs:
            path=raster_dir/f'{name}-2x.png';raster(mode,path,2160,3840,text,**type_options)
            fixed.append(dict(name=name,path=str(path.relative_to(root)),sha256=sha(path)))
        write_json(raster_dir/'manifest.json',dict(specSha256=spec_sha,captions=rows,fixed=fixed,
            captionTiming='Full measured speech windows; authored complete clauses proportionally subdivided, not word alignment',
            rasterizer=rasterizer))
        print(json.dumps(dict(captions=len(rows),nativeTypeOnly=True)));return

    assert platform.system()=='Linux', 'All media work requires the configured Linux GPU worker'
    run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'])
    import numpy as np
    try:
        from . import mix_absolute_sfx_longform as mix
        from .shorts_motion_crop import build_clip_filter
    except ImportError:
        import mix_absolute_sfx_longform as mix
        from shorts_motion_crop import build_clip_filter
    manifest_path=raster_dir/'manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['specSha256']==spec_sha
    inputs={}
    for key,binding in spec['inputs'].items():
        path=(ep/binding['path']).resolve();path.relative_to(ep);assert sha(path)==binding['sha256']
        inputs[key]=path
    for p in spec['picture']:
        path=(ep/p['path']).resolve();path.relative_to(ep);assert sha(path)==p['sha256']
    output=root/'render';output.mkdir(exist_ok=True)
    assert not (output/'render-audit.json').exists(), 'Preserve completed render'
    small={}
    for row in manifest['captions']+manifest['fixed']:
        source=root/row['path'];assert sha(source)==row['sha256']
        target=output/source.name.replace('-2x','')
        run(['ffmpeg','-y','-v','error','-i',source,'-vf','scale=1080:1920:flags=lanczos','-frames:v','1',target])
        small[row['path']]=target
    segments=[];runs=[];cursor=0
    for index,p in enumerate(spec['picture']):
        source=ep/p['path'];out=output/f'segment-{index:02d}.mp4'
        n=p['sourceFrames'];start=p['firstSourceFrame'];frames=p['frames'];speed=n/frames
        vf=(f'trim=start_frame={start}:end_frame={start+n},setpts=(PTS-STARTPTS)/{speed:.12f},fps=24,'
            +build_clip_filter(frames/24,p['layout'],1080,1920))
        run(['ffmpeg','-y','-v','error','-threads','3','-reinit_filter','0','-i',source,
             '-vf',vf,'-an','-frames:v',frames,'-c:v','libx264','-crf','17','-preset','fast',
             '-threads','4','-pix_fmt','yuv420p','-video_track_timescale','12288',out])
        probe=mix.probe_media(out)['video'][0];assert probe['frames']==frames
        segments.append((out,frames));runs.append(dict(id=p['id'],frames=frames,timelineStart=cursor/24,
            sourceSha256=p['sha256'],memberSegmentIds=[p['scene']],role='authored full-source moving panel'))
        cursor+=frames
    listing=output/'body-concat.txt'
    listing.write_text(''.join(f"file '{p}'\nduration {n/24:.12f}\n" for p,n in segments))
    body=output/'body-panels.mp4'
    run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',listing,'-map','0:v:0',
        '-c','copy','-bsf:v','setts=pts=round(PTS/512)*512:dts=round(DTS/512)*512',body])
    ins=['-reinit_filter','0','-i',body];filters=[];last='0:v'
    question=next((x for x in manifest['fixed'] if x['name']=='question'),None)
    assert bool(question)==(question_text(spec) is not None)
    overlays=([dict(question,start=0,end=spec['bodyFrames']/24)] if question else [])+manifest['captions']
    for index,c in enumerate(overlays):
        ins+=['-i',small[c['path']]]
        filters.append(f"[{last}][{index+1}:v]overlay=0:0:enable='gte(t,{c['start']:.9f})*lt(t,{c['end']:.9f})'[v{index}]")
        last=f'v{index}'
    capped=output/'body-captioned.mp4'
    run(['ffmpeg','-y','-v','error',*ins,'-filter_complex_threads','1','-filter_complex',';'.join(filters),
        '-map',f'[{last}]','-an','-frames:v',spec['bodyFrames'],'-c:v','libx264','-crf','16',
        '-preset','fast','-threads','4','-pix_fmt','yuv420p','-video_track_timescale','12288',capped])
    card=small[next(x for x in manifest['fixed'] if x['name']=='cta')['path']];cta=output/'cta.mp4'
    card_filter=cta_filter(spec)
    run(['ffmpeg','-y','-v','error','-loop','1','-framerate','24','-i',card,
        *(['-vf',card_filter] if card_filter else []),'-frames:v',spec['ctaFrames'],
        '-an','-c:v','libx264','-crf','16','-preset','fast','-threads','4','-pix_fmt','yuv420p',
        '-video_track_timescale','12288',cta])
    full_list=output/'full-concat.txt';full_list.write_text(
        f"file '{capped}'\nduration {spec['bodyFrames']/24:.12f}\nfile '{cta}'\nduration {spec['ctaFrames']/24:.12f}\n")
    picture=output/'picture-1080x1920.mp4'
    run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',full_list,'-c','copy',
        '-bsf:v','setts=pts=round(PTS/512)*512:dts=round(DTS/512)*512','-movflags','+faststart',picture])
    probe=mix.probe_media(picture)['video'][0]
    assert (probe['width'],probe['height'],probe['frames'],probe['frameRate'])==(1080,1920,spec['totalFrames'],'24/1')

    # Legacy film slicing and direct Short re-conform are separate contracts.
    prepared = spec.get('audioMode', 'legacy-film-slices') == 'prepared-whole-short'
    if spec.get('audioMode', 'legacy-film-slices') not in ('legacy-film-slices','prepared-whole-short'):
        raise ValueError('Unknown Short audio mode')
    if prepared:
        result,voice_short,audio_rows,whole_voice_error=prepared_audio(spec,inputs,mix.decode_samples)
    else:
        result,voice_short,audio_rows=legacy_audio(spec,inputs,mix,np)
        whole_voice_error=None
    assert np.isfinite(result).all()
    wav=output/'short-premaster-48kf32.wav';vwav=output/'short-voice-48kf32.wav'
    for path,samples in [(wav,result),(vwav,voice_short)]:
        run(['ffmpeg','-y','-v','error','-f','f32le','-ar','48000','-ac','2','-i','-',
             '-c:a','pcm_f32le',path],input=samples.astype('<f4').tobytes())
    runs.append(dict(id='channel-cta',frames=spec['ctaFrames'],timelineStart=spec['bodyFrames']/24,
        sourceSha256=sha(card),memberSegmentIds=['channel-cta'],role='explicit full-video channel end screen'))
    write_json(output/'picture-render-runs.json',runs)
    for p in spec['picture']:assert sha(ep/p['path'])==p['sha256']
    for k,b in spec['inputs'].items():assert sha(inputs[k])==b['sha256']
    assert sha(sp)==spec_sha
    receipt=dict(kind='authored-complete-line-motion-short',specSha256=spec_sha,toolSha256=sha(__file__),
        computeHost=socket.gethostname(),frames=spec['totalFrames'],bodyFrames=spec['bodyFrames'],
        ctaFrames=spec['ctaFrames'],resolution=[1080,1920],fps=24,
        picture=dict(path=str(picture.relative_to(ep)),sha256=sha(picture)),
        premaster=dict(path=str(wav.relative_to(ep)),sha256=sha(wav)),
        voice=dict(path=str(vwav.relative_to(ep)),sha256=sha(vwav)),audioSlices=audio_rows,
        nativeTypeManifestSha256=sha(manifest_path),captionCount=len(manifest['captions']),
        voiceOrderAndWholeWindowsPreserved=True,wholeVoicePcmMaxError=whole_voice_error,
        audioMode=spec.get('audioMode','legacy-film-slices'),ctaVisibility=spec.get('ctaVisibility','legacy-fades'),
        ctaAudio='Source-reconformed musical resolution; no speech' if prepared else 'silent after a background-only resolution; no extra source speech',
        finalApproved=False,continuousPlaybackPerformed=False,creativeReview='pending actual rendered output inspection')
    write_json(output/'render-audit.json',receipt);print(json.dumps(receipt))


def legacy_audio(spec,inputs,mix,np):
    # The film mixer is linear: voice + ducked music + ducked SFX. Subtraction
    # isolates that exact background, not speech restoration or new generation.
    audit=json.loads(inputs['mixAudit'].read_text())
    assert '[voice][mduck][sduck]amix=inputs=3:normalize=0:dropout_transition=0[out]' in audit['filterGraph']
    voice=mix.decode_samples(inputs['voice']);premix=mix.decode_samples(inputs['premix'])
    assert voice.shape==premix.shape
    background=premix.astype(np.float64)-voice.astype(np.float64)
    result=np.zeros((spec['totalFrames']*2000,2),np.float64);voice_short=result.copy();audio_rows=[]
    for i,b in enumerate(spec['beats']):
        source=b['sourceStartFrame']*2000;dest=b['startFrame']*2000;count=b['frames']*2000
        v=voice[source:source+count].astype(np.float64);bg=background[source:source+count].copy()
        assert len(v)==len(bg)==count
        previous=spec['beats'][i-1] if i else None;following=spec['beats'][i+1] if i+1<len(spec['beats']) else None
        fade_in=i==0 or previous['sourceStartFrame']+previous['frames']!=b['sourceStartFrame']
        fade_out=following is None or b['sourceStartFrame']+b['frames']!=following['sourceStartFrame']
        n=round(.18*48000)
        if fade_in:bg[:n]*=np.linspace(0,1,n)[:,None]
        if fade_out:
            tail=round((.6 if following is None else .18)*48000)
            bg[-tail:]*=np.linspace(1,0,tail)[:,None]
        voice_short[dest:dest+count]=v;result[dest:dest+count]=v+bg
        audio_rows.append(dict(id=b['id'],sourceStartSample=source,outputStartSample=dest,samples=count,
            voiceSamplesUnchanged=True,backgroundFadeIn=fade_in,backgroundFadeOut=fade_out))
    return result,voice_short,audio_rows


if __name__=='__main__':main()
