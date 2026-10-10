"""Render finite, unchanged-picture music-balance auditions on a GPU worker.

Current is a reconstruction of the delivered PREMASTER balance, not a claim of
identical mastered AAC. Every alternative shares exact voice/SFX and one common
linear headroom trim. No per-option loudness normalization, limiter or new audio
generation. Source-aware score re-conform restores excessive stacked attenuation.
"""
import argparse
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
from build_sound_revision_review import SR,sha,read,run as _run,decode,write_wav,block_rms,active_level,fit_length,peaks,packet_hash
from build_score_sfx_revision import ramp,apply_envelope


def run(args, **kwargs):
    try:return _run(args, **kwargs)
    except subprocess.CalledProcessError as exc:
        print((exc.stderr or b'').decode(errors='replace')[-6000:],flush=True)
        raise


def bound(root,spec):
    p=(root/spec['path']).resolve()
    assert p.is_relative_to(root) and sha(p)==spec['sha256'],str(p)
    return p


def excerpt(path,start,n):
    return fit_length(decode(path,start,n/SR),n,tolerance=1)


def levels(a):
    return {'rmsDb':round(20*np.log10(max(float(np.sqrt(np.mean(a.astype(float)**2))),1e-12)),3),
            'samplePeakDb':round(20*np.log10(max(float(np.max(np.abs(a))),1e-12)),3)}


def shaped_score(root,c,begin,n,variant,source_hashes):
    """Same tracks, phase, edits and EQ; different authored level policy only."""
    out=np.zeros((n,2),np.float32);end=begin+n/SR;records=[]
    for cue in c['music']:
        at=cue['atFrame']/24;length=cue['frames']/24
        lo=max(begin,at);hi=min(end,at+length)
        if hi<=lo:continue
        p=bound(root,cue['source']);source_hashes[str(p)]=cue['source']['sha256']
        raw=decode(p,filters=cue.get('filters','highpass=f=30,equalizer=f=2400:t=q:w=0.7:g=-3'))
        whole=active_level(raw);gain=10**((cue['activeRmsDb']+variant['baseGainDb'])/20)/whole
        a=round(cue['sourceIn']*SR);count=cue['frames']*2000
        assert a+count<=len(raw)
        part=raw[a:a+count].copy()*gain;ramp(part,cue['fadeIn'],cue['fadeOut'])
        if cue.get('envelope'):
            apply_envelope(part,[x[0] for x in cue['envelope']],[10**(x[1]/20) for x in cue['envelope']])
        l=round((lo-at)*SR);r=round((hi-at)*SR);dest=round((lo-begin)*SR)
        out[dest:dest+r-l]+=part[l:r]
        records.append({'cue':cue['id'],'sourceSha256':sha(p),'gain':gain,'sourceIn':cue['sourceIn']+l/SR,'sourceOut':cue['sourceIn']+r/SR})
    t=begin+np.arange(n)/SR
    if c.get('musicEnvelope'):
        points=c['musicEnvelope']
        gains=[10**((x[1] if x[1]<=-90 or x[1]>=0 else x[1]*variant['sceneReductionScale'])/20) for x in points]
        out*=np.interp(t,[x[0] for x in points],gains).astype(np.float32)[:,None]
    for hole in c.get('quietWindows',[]):
        lo=max(begin,hole['start']);hi=min(end,hole['end'])
        if hi<=lo:continue
        db=hole['musicDb'];db=db if db<=-90 else db*variant['quietReductionScale']
        g=0 if db<=-90 else 10**(db/20)
        full_a,full_b=hole['start'],hole['end'];r=min(hole.get('ramp',.25),(full_b-full_a)/2)
        env=np.interp(t,[full_a,full_a+r,full_b-r,full_b],[1,g,g,1]) if r else np.where((t>=full_a)&(t<full_b),g,1)
        out*=env.astype(np.float32)[:,None]
    # No new global full-band speech duck: retained 2.4kHz score EQ makes space.
    # Genuine authored silence stays; automatic blanket SFX handoffs are removed.
    return out,records


def build(plan_path):
    assert platform.system()=='Linux','Heavy media stays on GCP'
    run(['nvidia-smi','-L']);plan=read(plan_path);ph=sha(plan_path);ch=sha(__file__)
    assert plan['generationApiAllowed'] is False and plan['filmMutationAllowed'] is False
    out=Path(plan['output']).resolve();assert '/work/' in str(out)
    out.mkdir(parents=True,exist_ok=False)
    sources={};episodes={};draft=[];global_peak=0
    for key,spec in plan['episodes'].items():
        root=Path(spec['root']).resolve();paths={k:bound(root,v) for k,v in spec['inputs'].items()}
        sources.update({str(paths[k]):v['sha256'] for k,v in spec['inputs'].items()})
        episodes[key]=(root,paths,read(paths['mixConfig']),read(paths['clock']))
    assert sum(s['maxSeconds'] for s in plan['scenes'])<=240
    for s in plan['scenes']:
        root,paths,c,clock=episodes[s['episode']]
        events=[e for e in clock['events'] if s['shots'][0]<=e['shot']<=s['shots'][1]]
        assert len(events)==s['shots'][1]-s['shots'][0]+1
        start=round(events[0]['start']*24)/24;end=round(events[-1]['end']*24)/24
        frames=round((end-start)*24);n=frames*2000;assert n>0 and n/SR<=s['maxSeconds']
        voice=excerpt(paths['voice'],start,n);fx=excerpt(paths['sfx'],start,n);oldmusic=excerpt(paths['music'],start,n)
        if 'backgroundRepair' in paths:
            repair=read(paths['backgroundRepair']);lookup={e['id']:e for e in clock['events']}
            for w in repair['windows']:
                e=lookup[w['event']];lo=e['start']+e['subtitleAudioAt'];hi=lo+e['subtitleAudioDuration']
                full_a,full_b=round(lo*SR),round(hi*SR);r=min(round(w.get('rampSeconds',.12)*SR),(full_b-full_a)//2)
                g=10**(w['backgroundDb']/20);env=np.full(full_b-full_a,g,dtype=np.float32)
                if r:env[:r]=np.linspace(1,g,r);env[-r:]=np.linspace(g,1,r)
                base=round(start*SR);left=max(base,full_a);right=min(base+n,full_b)
                if right>left:
                    oldmusic[left-base:right-base]*=env[left-full_a:right-full_a,None]
                    fx[left-base:right-base]*=env[left-full_a:right-full_a,None]
        current=voice+oldmusic+fx
        reference=excerpt(paths['premaster'],start,n)
        error=float(np.max(np.abs(current-reference)));assert error<2e-6,('Current balance mismatch',s['id'],error)
        newmusic,records=shaped_score(root,c,start,n,plan['restoredPolicy'],sources)
        musics=[oldmusic,newmusic,newmusic*10**(plan['boldExtraDb']/20)]
        mixes=[voice+m+fx for m in musics]
        global_peak=max(global_peak,*[float(np.max(np.abs(a))) for a in mixes])
        for key,a in zip(('old','1','2'),mixes):write_wav(out/(s['id']+'-'+key+'-premaster.wav'),a)
        for key,m in zip(('old','1','2'),musics):write_wav(out/(s['id']+'-'+key+'-music.wav'),m)
        write_wav(out/(s['id']+'-voice.wav'),voice)
        draft.append((s,paths,start,end,frames,events,records,error,[levels(m) for m in musics]))
        print(json.dumps({'prepared':s['id'],'seconds':n/SR}),flush=True)
    common_gain=min(1,.78/global_peak);scenes=[]
    for s,paths,start,end,frames,events,records,error,mlevels in draft:
        proxy=out/(s['id']+'-picture.mp4');a=round(start*24);b=a+frames
        # Retained L4 driver is compute-only: libnvidia-encode is absent. Encode
        # this short proxy on the GPU worker's CPU; never use the Mac or rebuild
        # its qualified model/driver environment to prepare a mix audition.
        run(['ffmpeg','-y','-v','error','-i',paths['picture'],'-an','-vf',f"select=between(n\\,{a}\\,{b-1}),setpts=N/(24*TB),scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",'-frames:v',frames,'-c:v','libx264','-preset','fast','-crf','22','-threads','4','-pix_fmt','yuv420p',proxy])
        payload=packet_hash(proxy);variants=[]
        for key,label in [('old','Current balance'),('1','A · Score present'),('2','B · Score dominant')]:
            premaster=out/(s['id']+'-'+key+'-premaster.wav');mp4=out/(s['id']+'-'+key+'.mp4')
            run(['ffmpeg','-y','-v','error','-i',proxy,'-i',premaster,'-map','0:v:0','-map','1:a:0','-c:v','copy','-af',f'volume={common_gain:.12f}','-c:a','aac','-b:a','320k','-ar',SR,'-t',frames/24,'-movflags','+faststart',mp4])
            run(['ffmpeg','-v','error','-xerror','-i',mp4,'-f','null','-'])
            assert packet_hash(mp4)==payload
            probe=json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',mp4]).stdout)
            vs=[x for x in probe['streams'] if x['codec_type']=='video'];au=[x for x in probe['streams'] if x['codec_type']=='audio']
            assert len(vs)==len(au)==1 and int(vs[0]['nb_read_frames'])==frames
            assert vs[0]['r_frame_rate']=='24/1' and abs(float(vs[0]['start_time']))<.001
            assert abs(float(au[0]['start_time']))<.001 and abs(float(au[0]['duration'])-frames/24)<.025
            measured=run(['ffmpeg','-v','info','-i',mp4,'-af','loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json','-f','null','-']).stderr.decode()
            loud=json.loads(measured[measured.rfind('{'):]);assert float(loud['input_tp'])<=-1.0
            musicfile=out/(s['id']+'-'+key+'-music.m4a')
            run(['ffmpeg','-y','-v','error','-i',out/(s['id']+'-'+key+'-music.wav'),'-af',f'volume={common_gain:.12f}','-c:a','aac','-b:a','256k',musicfile])
            variants.append({'key':key,'label':label,'url':mp4.name,'sha256':sha(mp4),'wave':peaks(decode(mp4)),
                'music':{'url':musicfile.name,'sha256':sha(musicfile),'sourceStart':0,'sourceEnd':frames/24,'prompt':'Existing chosen score takes, unchanged timing and musical content. Only mix-level treatment differs. Music-only playback uses the same gain as in the comparison video.'},
                'loudness':loud,'commonGain':common_gain,'musicLevels':mlevels[len(variants)],'decodedFrames':frames,'audioStart':float(au[0]['start_time'])})
        scenes.append({**s,'start':start,'end':end,'duration':frames/24,'text':'\n'.join(e['text'] for e in events),
            'lines':[{'shot':e['shot'],'start':e['start']-start+e['subtitleAudioAt'],'end':e['start']-start+e['subtitleAudioAt']+e['subtitleAudioDuration'],'text':e['text']} for e in events],
            'effects':[],'variants':variants,'currentPremasterMaxError':error,'picturePacketSha256':payload,'scoreSources':records})
        print(json.dumps({'rendered':s['id'],'versions':3}),flush=True)
    for path,h in sources.items():assert sha(path)==h,path
    assert sha(plan_path)==ph and sha(__file__)==ch
    result={'episode':' + '.join(plan['episodes']),'version':1,'scenes':scenes,'sounds':[],
        'scope':'Current balance reconstructs the delivered premaster, not its mastered AAC. All versions retain identical voice and SFX at one common linear headroom trim; only music differs. No per-version normalization or limiter. Existing music takes, phase and picture stay unchanged. A reduces excessive stacked suppression and raises the score bed; B adds 3 dB of music over A. Genuine authored silence remains. Auditions only; neither final film has been replaced.',
        'planSha256':ph,'runnerSha256':ch,'musicManifestSha256':ph,'sourceHashes':sources,
        'filmChanged':False,'creativeApproval':False,'commonGain':common_gain,'policy':plan['restoredPolicy'],
        'compute':'All processing on qualified GPU worker L4 worker; CPU audio processing and software H.264 proxy encoding. No Mac media compute. NVENC unavailable in retained compute-only driver.'}
    (out/'review-data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    template=Path(__file__).with_name('music_balance_review_template.html').read_text()
    (out/'MUSIC_BALANCE_REVIEW.html').write_text(template.replace('/*__DATA__*/null',json.dumps(result,ensure_ascii=False).replace('<','\\u003c')))
    print(json.dumps({'page':str(out/'MUSIC_BALANCE_REVIEW.html'),'scenes':len(scenes),'commonGain':common_gain}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('plan',type=Path);a=ap.parse_args();build(a.plan.resolve())
