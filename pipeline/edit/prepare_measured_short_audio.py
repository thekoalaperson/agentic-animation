"""GPU-only complete-performance Short conform with two authored score windows."""
import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import numpy as np
from render_designed_sfx import sha, decode, save
from short_audio_contract import gap_frames, gap_room_tone

def main():
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('config',type=Path);a=p.parse_args()
    ep=a.episode.resolve();cfg=json.loads(a.config.read_text());cfgsha=sha(a.config)
    if platform.system()!='Linux' or not shutil.which('nvidia-smi'):raise ValueError('GCP media worker required')
    paths={}
    for key,b in cfg['inputs'].items():
        q=(ep/b['path']).resolve()
        if not q.is_relative_to(ep) or sha(q)!=b['sha256']:raise ValueError('Changed input '+key)
        paths[key]=q
    spec=json.loads(paths['spec'].read_text());source_clock=json.loads(paths['sourceClock'].read_text())
    source_audit=json.loads(paths['sourceVoiceAudit'].read_text())
    if source_audit['outputs']['voice']['sha256']!=sha(paths['sourceVoice']) or source_audit['clockSha256']!=sha(paths['sourceClock']):raise ValueError('Source voice receipt mismatch')
    root=(ep/cfg['outputDirectory']).resolve()
    if not root.is_relative_to(ep/'work') or (root/'mix-audit.json').exists():raise ValueError('New working output required')
    root.mkdir(parents=True,exist_ok=True)
    n=spec['totalFrames']*2000
    voice=np.zeros((n,2),np.float32);music=voice.copy();sfx=voice.copy()
    source_voice=decode(paths['sourceVoice']);source_sfx=decode(paths['sourceSfx'])
    events={e['id']:e for e in source_clock['events']};selected=[];sfx_windows=[];cursor=0
    assert spec['beats'][0]['id']==source_clock['events'][0]['id']
    previous=-1
    for b in spec['beats']:
        e=events[b['id']];pos=source_clock['events'].index(e)
        assert pos>previous and b['startFrame']==cursor and b['sourceStartFrame']==round(e['start']*24)
        assert 0 < b['frames'] <= round(e['duration']*24) and b['spokenTextHi']==e['text']
        first=b['sourceStartFrame']*2000;count=b['frames']*2000;dest=cursor*2000
        assert e['audioSourceDuration']<=count/48000+1/48000
        voice[dest:dest+count]=source_voice[first:first+count]
        # Sound contacts may move when a Short uses newly authored picture.
        # This never changes the complete voice window above.
        sfx_frame=b.get('sfxSourceStartFrame',b['sourceStartFrame'])
        assert type(sfx_frame) is int and sfx_frame>=0
        if sfx_frame!=b['sourceStartFrame']:
            assert isinstance(b.get('sfxOffsetReason'),str) and b['sfxOffsetReason'].strip()
        sfx_first=sfx_frame*2000
        assert sfx_first+count<=len(source_sfx)
        fx=source_sfx[sfx_first:sfx_first+count].copy();fade=min(4800,len(fx)//2)
        fx[:fade]*=np.linspace(0,1,fade)[:,None];fx[-fade:]*=np.linspace(1,0,fade)[:,None]
        sfx[dest:dest+count]=fx
        sfx_windows.append(dict(event=b['id'],sourceStartFrame=sfx_frame,
            shortStartFrame=cursor,frames=b['frames'],reason=b.get('sfxOffsetReason')))
        se=copy.deepcopy(e);se.update(start=cursor/24,end=(cursor+b['frames'])/24,duration=b['frames']/24)
        gap=gap_frames(b)
        if gap:
            voice[dest+count:dest+count+gap*2000]=gap_room_tone(spec,source_voice,gap)
        selected.append(se);cursor+=b['frames']+gap;previous=pos
    assert cursor==spec['bodyFrames']
    score_rows=[]
    for c in cfg['score']:
        raw=decode(paths[c['source']],f"atrim=start={c['sourceIn']}:duration={c['seconds']},asetpts=PTS-STARTPTS,highpass=f=30,equalizer=f=2400:t=q:w=.7:g=-3")
        count=round(c['seconds']*48000);assert abs(len(raw)-count)<=2400
        raw=np.pad(raw,((0,max(0,count-len(raw))),(0,0)))[:count]
        block=raw[:len(raw)//4800*4800].reshape(-1,4800,2)
        level=float(np.percentile(np.sqrt(np.mean(block.astype(float)**2,axis=(1,2))),90))
        assert level>1e-7
        gain=np.clip(c['activeRmsDb']-20*np.log10(level),-18,12);raw*=10**(gain/20)
        for edge,length in [('in',c['fadeIn']),('out',c['fadeOut'])]:
            m=min(round(length*48000),len(raw)//2)
            if m:
                if edge=='in':raw[:m]*=np.linspace(0,1,m)[:,None]
                else:raw[-m:]*=np.linspace(1,0,m)[:,None]
        start=round(c['at']*48000);assert start+count<=n
        music[start:start+count]+=raw
        score_rows.append(dict(**c,actualGainDb=float(gain)))
    reduction=cfg.get('backgroundSpeechReductionDb',0)
    assert type(reduction) in (int,float) and 0<=reduction<=8
    if reduction:
        assert cfg.get('backgroundSpeechReductionReason')
        envelope=np.ones(n,np.float64)
        gain=10**(-reduction/20)
        for beat in spec['beats']:
            start=beat['startFrame']*2000
            end=min(n,round((beat['startFrame']/24+beat['audioDuration'])*48000))
            lo=max(0,start-4800);hi=min(n,end+16800)
            t=np.arange(lo,hi)
            amount=np.minimum(np.clip((t-lo)/max(1,start-lo),0,1),
                              np.clip((hi-t)/max(1,hi-end),0,1))
            envelope[lo:hi]=np.minimum(envelope[lo:hi],1-amount*(1-gain))
        music*=envelope[:,None];sfx*=envelope[:,None]
    premix=voice.astype(np.float64)+music.astype(np.float64)+sfx.astype(np.float64)
    recovery=float(np.max(np.abs((premix-music-sfx)-voice)))
    assert recovery<2e-6 and np.isfinite(premix).all()
    clock=copy.deepcopy(source_clock);clock.update(events=selected,runtimeSeconds=n/48000,
        eventCount=len(selected),
        kind='measured-speech-clock-not-renderable-edit-plan',renderable=False,finalApproved=False,
        note='Complete original performances in film order, followed by speech-free CTA')
    clock.pop('clockFingerprint',None)
    def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    def bind(path):return {'path':str(path.relative_to(ep)),'sha256':sha(path)}
    cp=root/'measured-clock.json';write(cp,clock)
    vp=root/'voice-48kf32.wav';mp=root/'working-premaster-48kf32.wav'
    save(vp,voice);save(mp,premix)
    va=root/'voice-audit.json';write(va,dict(clockSha256=sha(cp),outputs={'voice':bind(vp)},
        method='Exact complete-performance PCM windows; optional hash-bound source room tone in authored gaps; no speech fades/retime/generation',finalApproved=False))
    ma=root/'mix-audit.json';write(ma,dict(kind='working-audio-context-not-final',voiceGain=1,
        unchangedVoiceRecoveryMaxError=recovery,inputs={'clock':bind(cp),'voice':bind(vp)},
        outputs={'working-premaster-48kf32.wav':bind(mp)},samples=n,sampleRate=48000,
        score=score_rows,sfxWindows=sfx_windows,backgroundSpeechReductionDb=reduction,
        backgroundSpeechReductionReason=cfg.get('backgroundSpeechReductionReason'),configSha256=cfgsha,
        sourceBindings=cfg['inputs'],finalApproved=False))
    spec['inputs']={'clock':bind(cp),'voice':bind(vp),'voiceAudit':bind(va),'premix':bind(mp),'mixAudit':bind(ma),
        'sourceVoice':cfg['inputs']['sourceVoice'],'sourceVoiceAudit':cfg['inputs']['sourceVoiceAudit'],'sourceClock':cfg['inputs']['sourceClock']}
    spec['status']='Prepared audio bound; actual rendered portrait and sound review pending'
    target=ep/cfg['specOutput'];assert not target.exists();write(target,spec)
    for key,path in paths.items():assert sha(path)==cfg['inputs'][key]['sha256']
    assert sha(a.config)==cfgsha
    print(json.dumps({'spec':str(target),'seconds':n/48000,'voiceRecoveryError':recovery}),flush=True)

if __name__=='__main__':main()
