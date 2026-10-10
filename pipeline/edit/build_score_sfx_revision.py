"""Build a source-bound linear score/SFX revision on a qualified GPU host.

Episode files own every selection, level envelope and contact. No provider,
picture generation, automatic take approval or final-delivery mutation lives here.
"""
import argparse
import json
import platform
import math
from pathlib import Path
import numpy as np
from build_sound_revision_review import SR, sha, read, run, decode, write_wav, block_rms, active_level, fit_length
from build_working_audio import validate_working_clock


def bound(ep, spec):
    p=(ep/spec['path']).resolve()
    if not p.is_relative_to(ep) or sha(p)!=spec['sha256']:
        raise RuntimeError('Changed or escaped input: '+str(p))
    return p


def ramp(data, fade_in, fade_out):
    ni,no=[min(round(x*SR),len(data)//2) for x in (fade_in,fade_out)]
    if ni:data[:ni]*=np.linspace(0,1,ni,dtype=np.float32)[:,None]
    if no:data[-no:]*=np.linspace(1,0,no,dtype=np.float32)[:,None]


def apply_envelope(data, times, values):
    # Chunked interpolation avoids multi-gigabyte temporary full-film arrays.
    for start in range(0,len(data),SR*30):
        stop=min(len(data),start+SR*30)
        env=np.interp(np.arange(start,stop)/SR,times,values).astype(np.float32)
        data[start:stop]*=env[:,None]


def validate_clock(c, plan):
    working = c.get('workingAudioOnly') is True
    if working:
        if c.get('finalApproved') is not False:
            raise RuntimeError('Working audio cannot claim final approval')
        validate_working_clock(plan, c['episode'])
    elif (plan.get('renderable') is not True
          or plan.get('kind') == 'measured-speech-clock-not-renderable-edit-plan'
          or plan.get('episode') != c['episode']):
        raise RuntimeError('Final mix requires its renderable episode plan')
    fps = plan.get('deliveryFps', plan.get('fps', 24))
    if fps != 24:
        raise RuntimeError('This mixer requires the exact 24fps delivery clock')
    values = [plan['runtimeSeconds']]
    values += [e[k] for e in plan['events'] for k in ('start', 'end', 'duration')]
    if any(not math.isfinite(v) or abs(v*24-round(v*24)) > 0.00005 for v in values):
        raise RuntimeError('Off-grid score or speech timing')
    return working


def validate_working_stems(c, paths):
    voice_audit = read(paths['voiceAudit'])
    sfx_audit = read(paths['sfxAudit'])
    clock_hash = c['inputs']['clock']['sha256']
    if (voice_audit.get('kind') != 'working-audio-stems-not-final'
            or voice_audit.get('finalApproved') is not False
            or voice_audit['outputs']['voice']['sha256'] != c['inputs']['voice']['sha256']):
        raise RuntimeError('Voice stem lacks working admission provenance')
    if voice_audit.get('clockSha256') is not None:
        if voice_audit['clockSha256'] != clock_hash:
            raise RuntimeError('Voice stem clock changed')
    else:
        voice_config = paths['voiceConfig']
        if sha(voice_config) != voice_audit['configSha256'] or read(voice_config)['clock'] != c['inputs']['clock']:
            raise RuntimeError('Legacy voice config does not bind this clock')
    if (sfx_audit.get('workingAudioOnly') is not True
            or sfx_audit.get('finalApproved') is not False
            or sfx_audit.get('clockSha256') != clock_hash
            or sfx_audit['sha256'] != c['inputs']['sfx']['sha256']):
        raise RuntimeError('SFX stem clock or receipt changed')


def exact_stem(data, n, label):
    if abs(len(data)-n) > 1:
        raise RuntimeError('Stem length differs from full clock: '+label)
    return fit_length(data, n, tolerance=1)


def build(ep, config):
    if platform.system()!='Linux':raise RuntimeError('Production audio belongs on the qualified GPU worker')
    run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'])
    digest=sha(config); code=sha(__file__); c=read(config)
    assert c['generationApiAllowed'] is False and c['voiceGain']==1 and c['pictureMutationAllowed'] is False
    paths={k:bound(ep,v) for k,v in c['inputs'].items()}
    plan=read(paths['clock']);working=validate_clock(c,plan)
    if working:validate_working_stems(c,paths)
    frames=round(plan['runtimeSeconds']*24);n=frames*2000
    root=(ep/c['output']).resolve();assert root.is_relative_to(ep/'work')
    root.mkdir(exist_ok=False,parents=True)
    voice=exact_stem(decode(paths['voice']),n,'voice')
    sfx=exact_stem(decode(paths['sfx']),n,'sfx')
    music=np.zeros_like(voice);sources={};records=[]
    for cue in c['music']:
        source=bound(ep,cue['source']);sources[source]=cue['source']['sha256']
        raw=decode(source,filters=cue.get('filters','highpass=f=30,equalizer=f=2400:t=q:w=0.7:g=-3'))
        level=active_level(raw)
        at=round(cue['atFrame']*2000);start=round(cue['sourceIn']*SR);length=cue['frames']*2000
        if min(at,start)<0 or at+length>n or start+length>len(raw):
            raise RuntimeError('Music window exceeds source/film: '+cue['id'])
        part=raw[start:start+length].copy();gain=10**(cue['activeRmsDb']/20)/level;part*=gain
        ramp(part,cue['fadeIn'],cue['fadeOut'])
        if cue.get('envelope'):
            pts=cue['envelope'];assert pts[0][0]==0 and pts[-1][0]<=length/SR
            assert all(a[0]<b[0] for a,b in zip(pts,pts[1:]))
            apply_envelope(part,[p[0] for p in pts],[10**(p[1]/20) for p in pts])
        music[at:at+length]+=part
        records.append({**cue,'wholeSourceActiveRms':level,'gain':gain,'actualSourceEnd':(start+length)/SR})
        print(json.dumps({'score':cue['id'],'seconds':length/SR}),flush=True)
    # Slow scene-wise energy arc, independent from mood or harmonic brightness.
    if c.get('musicEnvelope'):
        apply_envelope(music,[x[0] for x in c['musicEnvelope']],[10**(x[1]/20) for x in c['musicEnvelope']])
    vr=block_rms(voice);mr=block_rms(music);fr=block_rms(sfx)
    controls=c['mix'];active=vr>controls['speechRmsThreshold']
    protection=np.where(active,np.minimum(1,vr*controls['musicToVoiceRatio']/np.maximum(mr,1e-7)),1)
    handoff=np.where(fr>controls['effectHandoffRms'],controls['musicDuringEffectsGain'],1.)
    times=np.arange(len(vr))*.1+.05
    apply_envelope(music,times,protection*handoff)
    # Protect speech without retiming/processing it; no blanket SFX beat sidechain.
    fxprotect=np.where(active,np.minimum(1,vr*controls['sfxToVoiceRatio']/np.maximum(fr,1e-7)),1)
    apply_envelope(sfx,times,fxprotect)
    spectral_audit = None
    if c.get('speechBandProtection'):
        # Preserve bass/air and speech-free score presence; yield only the
        # masking band around weak voiced syllables. The voice stays untouched.
        spec = c['speechBandProtection']
        low, high = spec['lowHz'], spec['highHz']
        ratio, maximum = spec['backgroundToVoiceRatio'], spec['maximumReductionDb']
        assert 120 <= low <= 400 and 3000 <= high <= 6000
        assert .3 <= ratio <= 1 and 0 <= maximum <= 9
        def speech_band(data):
            result = np.empty_like(data)
            for start in range(0, n, 30*SR):
                end = min(n, start+30*SR)
                left, right = max(0,start-SR), min(n,end+SR)
                part = data[left:right]
                fft_size = 1 << (len(part)-1).bit_length()
                hz = np.fft.rfftfreq(fft_size, 1/SR)
                weight = 1/(1+(low/np.maximum(hz,1e-9))**4)
                weight *= 1/(1+(hz/high)**4)
                filtered = np.fft.irfft(np.fft.rfft(part,n=fft_size,axis=0)*weight[:,None],n=fft_size,axis=0)
                result[start:end] = filtered[start-left:end-left]
            return result
        vb, mb, fb = [speech_band(x) for x in (voice, music, sfx)]
        block = 960  #20ms resolves syllables rather than whole phrases.
        def band_rms(data):
            count = math.ceil(n/block)
            padded = np.pad(data, ((0,count*block-n),(0,0)))
            return np.sqrt(np.mean(padded.reshape(count,block,2).astype(float)**2,axis=(1,2)))
        vrb, brb = band_rms(vb), band_rms(mb+fb)
        desired = np.where(vrb>10**(-38/20),
                           np.clip(vrb*ratio/np.maximum(brb,1e-8),10**(-maximum/20),1),1)
        # A short hold covers weak consonants adjacent to vowels without fast pumping.
        extended = np.pad(desired,(8,8),mode='edge')
        held = np.minimum.reduce([extended[i:i+len(desired)] for i in range(17)])
        band_times = np.arange(len(held))*.02+.01
        # Explicit scene-specific masking corrections retain bass and air.
        # They cover the complete spoken phrase, including short internal pauses.
        local_reductions = spec.get('eventReductionDb', {})
        events_by_id = {e['id']:e for e in plan['events']}
        for event_id, reduction in local_reductions.items():
            assert event_id in events_by_id and type(reduction) in (int,float) and 0 <= reduction <= 9
            event = events_by_id[event_id]
            first = event['start'] + event['subtitleAudioAt']
            last = first + event['subtitleAudioDuration']
            envelope = np.clip(np.minimum((band_times-first+.15)/.15,(last+.2-band_times)/.2),0,1)
            held *= 10**(-reduction*envelope/20)
        for start in range(0,n,30*SR):
            end = min(n,start+30*SR)
            gain = np.interp(np.arange(start,end)/SR,band_times,held).astype(np.float32)
            music[start:end] += mb[start:end]*(gain[:,None]-1)
            sfx[start:end] += fb[start:end]*(gain[:,None]-1)
        spectral_audit = dict(**spec,method='Zero-phase smooth FFT band extraction in30s blocks with1s margins;20ms detector with320ms hold; bass/air retained; voice untouched',
                              minimumBandGain=float(held.min()),medianBandGain=float(np.median(held)))
        del vb, mb, fb
    # Explicit authored foreground/background holes, including the impact aftermath.
    for hole in c.get('quietWindows',[]):
        start,end=[round(hole[k]*SR) for k in ('start','end')]
        assert 0<=start<end<=n
        for name,data in [('music',music),('sfx',sfx)]:
            gain=10**(hole[name+'Db']/20) if hole[name+'Db']>-100 else 0
            part=data[start:end];r=min(round(hole.get('ramp',.25)*SR),len(part)//2)
            if r:
                part[:r]*=np.linspace(1,gain,r)[:,None]
                part[-r:]*=np.linspace(gain,1,r)[:,None]
                part[r:-r]*=gain
            else:part*=gain
    ramp(music,0,c['finalBackgroundFade']);ramp(sfx,0,c['finalBackgroundFade'])
    background=music+sfx;premaster=voice+background
    assert np.isfinite(premaster).all()
    # FLOAT premaster may exceed unity; delivery's measured mastering controls it.
    outputs={}
    for name,data in [('music-48kf32.wav',music),('sfx-shaped-48kf32.wav',sfx),
                      ('background-48kf32.wav',background),('working-premaster-48kf32.wav',premaster)]:
        dest=root/name;write_wav(dest,data);outputs[name]={'path':str(dest.relative_to(ep)),'sha256':sha(dest),'samples':len(data),'peak':float(np.max(np.abs(data)))}
    # Exact linear voice recovery tolerance for float summation; preserved source is authority.
    error=float(np.max(np.abs((premaster-background)-voice)))
    assert error<2e-6
    rows=[]
    for e in plan['events']:
        a,b=[round(e[k]*SR) for k in ('start','end')]
        v=voice[a:b];bg=background[a:b]
        rmsv=float(np.sqrt(np.mean(v.astype(float)**2)));rmsb=float(np.sqrt(np.mean(bg.astype(float)**2)))
        rows.append({'shot':e['shot'],'kind':e['kind'],'voiceRms':rmsv,'backgroundRms':rmsb,
            'voiceAboveBackgroundDb':float(20*np.log10(max(rmsv,1e-9)/max(rmsb,1e-9)))})
    for key,path in paths.items():assert sha(path)==c['inputs'][key]['sha256']
    for path,d in sources.items():assert sha(path)==d
    assert sha(config)==digest and sha(__file__)==code
    receipt=dict(episode=c['episode'],configSha256=digest,runnerSha256=code,inputs=c['inputs'],
        kind='working-audio-context-not-final' if working else 'source-bound-score-sfx-revision',
        workingAudioOnly=working,finalApproved=False if working else c.get('finalApproved',False),
        renderablePictureCreated=False,clockSha256=c['inputs']['clock']['sha256'],
        route='source-bound new score + self-hosted text-to-audio SFX; linear unity-gain Hindi voice',
        bodyFrames=frames,sampleRate=SR,samples=n,outputs=outputs,music=records,
        unchangedVoiceRecoveryMaxError=error,voiceGain=1,musicProtectionMinimum=float(protection.min()),
        sfxProtectionMinimum=float(fxprotect.min()),speechBandProtection=spectral_audit,eventLevels=rows,
        scope='Delegated studio mix execution and technical provenance. Not a fabricated human full-film listen.')
    (root/'mix-audit.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(root),'scorePlacements':len(records),'voiceRecoveryMaxError':error,'premasterPeak':outputs['working-premaster-48kf32.wav']['peak']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('config',type=Path);a=p.parse_args()
    build(a.episode.resolve(),a.config.resolve())
