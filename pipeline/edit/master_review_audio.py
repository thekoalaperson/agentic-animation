#!/usr/bin/env python3
"""Two-pass master an existing premix without changing picture or granting approval."""
import argparse
from array import array
from fractions import Fraction
import json
import platform
from pathlib import Path
import socket
import subprocess
import mix_absolute_sfx_longform as mix


def master_mux_command(picture,premaster,output,normalizer,duration,sample_clock=False,movie_clock=True):
    if sample_clock:
        normalizer+=',aformat=channel_layouts=stereo,aresample=48000,asetpts=N/SR/TB'
    command=mix.body_mux_command(picture,premaster,output,normalizer,duration)
    if sample_clock and movie_clock:
        command[-1:-1]=['-movie_timescale','48000']
    return command


def qualified_limiter(limit=.75, release=80):
    """Measure the installed limiter's delay using a known isolated impulse."""
    assert .5 <= limit <= .85 and release in (10,20,80)
    base=f'alimiter=limit={limit}:attack=5:release={release}:level=false'
    signal=array('f',[0.0])*48000; signal[10000]=0.1
    result=subprocess.run(['ffmpeg','-v','error','-f','f32le','-ar','192000',
        '-ac','1','-i','pipe:0','-af',base,'-f','f32le','pipe:1'],
        input=signal.tobytes(),capture_output=True,check=True)
    decoded=array('f');decoded.frombytes(result.stdout)
    delay=max(range(len(decoded)),key=lambda i:abs(decoded[i]))-10000
    assert 0<=delay<=3840 and len(decoded)==len(signal), 'Unqualified limiter clock'
    compensate=f',atrim=start_sample={delay},asetpts=N/SR/TB,apad=pad_len={delay}' if delay else ''
    return base+compensate,dict(sampleRate=192000,impulseInputSample=10000,
        measuredDelaySamples=delay,delaySeconds=delay/192000,
        compensation='Exact measured leading-delay trim and equal trailing pad; no voice retime')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('config',type=Path)
    a=p.parse_args(); ep=a.episode.resolve(); config_path=(ep/a.config).resolve()
    assert platform.system()=='Linux', 'Mastering belongs on the qualified GPU worker'
    subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],check=True)
    config_sha=mix.sha256_file(config_path); tool_sha=mix.sha256_file(Path(__file__))
    c=json.loads(config_path.read_text()); assert c['kind']=='source-bound-review-master'
    sample_clock=c.get('sampleClock48k',False);assert type(sample_clock) is bool
    movie_clock=c.get('movieTimescale48k',True);assert type(movie_clock) is bool
    # Older retained FFmpeg muxers cannot set the container movie timescale.
    # Explicit opt-out preserves the 48kHz emitted audio sample clock; it does
    # not change picture packets. Record the distinction in the receipt.
    if sample_clock and movie_clock:
        options=subprocess.run(['ffmpeg','-hide_banner','-h','muxer=mp4'],
            capture_output=True,text=True,check=True)
        assert '-movie_timescale' in options.stdout+options.stderr, 'Muxer lacks movie_timescale; explicit config opt-out required'
    fast=c.get('gpuFinishingProbe',False);assert type(fast) is bool
    if fast:import gpu_finishing_probe as gpu_probe
    probe_media=gpu_probe.probe_media if fast else mix.probe_media
    probe_tool_sha=mix.sha256_file(Path(gpu_probe.__file__)) if fast else None
    paths={}
    for key,spec in c['inputs'].items():
        paths[key]=mix.episode_path(ep,spec['path'],key)
        mix.verify_hash(paths[key],spec['sha256'],key)
    frames=c['expectedVideoFrames']; assert type(frames) is int and frames>0
    duration=frames/24; out=mix.work_output(ep,c['output'],'review master')
    assert not out.exists(), 'Version completed masters rather than overwrite'
    source=probe_media(paths['picture']); assert len(source['video'])==1 and not source['audio']
    v=source['video'][0]
    assert v['frames']==frames and Fraction(v['frameRate'])==Fraction(24)
    mix.verify_audio_probe(probe_media(paths['premaster']),duration,round(duration*48000),'premaster',codec='pcm_f32le')
    settings={'targetIntegratedLufs':c.get('targetIntegratedLufs',-14),
              'targetTruePeakDbtp':c.get('targetTruePeakDbtp',-1.5),'targetLra':11}
    assert settings['targetIntegratedLufs'] in (-14,-10)
    assert -3 <= settings['targetTruePeakDbtp'] <= -1.5
    mastering_source=paths['premaster'];compression=None
    if c.get('socialCompression'):
        assert c['socialCompression'] is True and settings['targetIntegratedLufs']==-10
        # Qualified social-master recipe from finish_opener_short.py. Compression
        # is explicit and platform-only; original film/voice/premix are immutable.
        out.parent.mkdir(parents=True,exist_ok=True)
        mastering_source=out.with_suffix('.controlled.wav')
        assert not mastering_source.exists(), 'Version controlled premaster outputs'
        ratio=c.get('socialCompressionRatio',4)
        assert type(ratio) in (int,float) and 1<=ratio<=4
        compressor=f'acompressor=threshold=0.10:ratio={ratio}:attack=5:release=160:makeup=1'
        mix.run(['ffmpeg','-y','-v','error','-i',str(paths['premaster']),'-vn',
                 '-af',compressor,'-ar','48000','-ac','2','-c:a','pcm_f32le',str(mastering_source)])
        mix.verify_audio_probe(probe_media(mastering_source),duration,round(duration*48000),
                               'controlled social premaster',codec='pcm_f32le')
        compression=dict(filter=compressor,sourceSha256=c['inputs']['premaster']['sha256'],
                         controlledSha256=mix.sha256_file(mastering_source),scope='Reels only; no retime')
    measured=mix.loudness(mastering_source,settings)
    normalizer=(f"loudnorm=I={settings['targetIntegratedLufs']}:TP={settings['targetTruePeakDbtp']}:LRA=11:"
        f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true")
    fixed_social=c.get('fixedGainSocial',False)
    assert type(fixed_social) is bool
    fixed_master=c.get('fixedGainMaster',False)
    assert type(fixed_master) is bool
    if fixed_social:
        assert c.get('socialCompression') is True and settings['targetIntegratedLufs']==-10
        # A peak-infeasible loudnorm linear request silently becomes dynamic.
        # For the explicit social alternative, use one measured fixed gain and
        # a latency-qualified limiter instead of automatic long-term gain riding.
    if fixed_social or fixed_master:
        normalizer=f"volume={settings['targetIntegratedLufs']-float(measured['input_i']):.6f}dB"
    output_gain=c.get('postNormalizationGainDb',0)
    assert type(output_gain) in (int,float) and -1<=output_gain<=1, 'Bound encoded-output gain correction'
    if output_gain:
        normalizer+=f',volume={output_gain:.6f}dB'
    out.parent.mkdir(parents=True,exist_ok=True); partial=out.with_suffix('.partial.mp4')
    initial_filter=normalizer
    limiter_limit=c.get('limiterLimit',.75)
    limiter_release=c.get('limiterReleaseMs',80)
    initial_proof=None
    if fixed_social or fixed_master:
        initial_limiter,initial_proof=qualified_limiter(limiter_limit,limiter_release)
        initial_filter+=',aresample=192000,'+initial_limiter+',aformat=channel_layouts=stereo,aresample=48000'
    command=master_mux_command(paths['picture'],mastering_source,partial,initial_filter,duration,sample_clock,movie_clock)
    mix.run(command)
    delivered=mix.loudness(partial,settings)
    corrections=[]
    limiter_qualification=initial_proof
    if c.get('socialCompression') or fixed_master:
        # Retained social-master repair: measure encoded audio, then compensate
        # the shortfall with an oversampled, latency-compensated peak limiter.
        # Each attempt re-encodes the lossless source, not a prior AAC encode.
        gain=0.0
        limiter,limiter_qualification=qualified_limiter(limiter_limit,limiter_release)
        for attempt in range(3):
            delta=settings['targetIntegratedLufs']-float(delivered['input_i'])
            if abs(delta)<=.5 and float(delivered['input_tp'])<=-1:
                break
            gain+=delta
            correction=(f'aresample=192000,volume={gain:.4f}dB,'
                        +limiter+',aformat=channel_layouts=stereo,aresample=48000')
            tuned=out.with_suffix(f'.correction-{attempt+1}.mp4')
            assert not tuned.exists(), 'Version encoded-correction evidence'
            mix.run(master_mux_command(paths['picture'],mastering_source,tuned,
                                         normalizer+','+correction,duration,sample_clock,movie_clock))
            delivered=mix.loudness(tuned,settings)
            corrections.append(dict(filter=correction,encoded=delivered,
                                    path=str(tuned),sha256=mix.sha256_file(tuned),
                                    source='lossless controlled premaster; no AAC cascade'))
            partial=tuned
        # If limiting leaves measured AAC headroom, use a bounded final trim
        # instead of compressing more. Always re-encode the lossless source.
        deficit=settings['targetIntegratedLufs']-float(delivered['input_i'])
        headroom=-1.15-float(delivered['input_tp'])
        if deficit>.5 and headroom>0 and corrections:
            trim=min(deficit,headroom,1.0)
            correction=corrections[-1]['filter']+f',volume={trim:.4f}dB'
            tuned=out.with_suffix('.headroom.mp4')
            assert not tuned.exists(), 'Version final measured headroom trim'
            mix.run(master_mux_command(paths['picture'],mastering_source,tuned,
                                         normalizer+','+correction,duration,sample_clock,movie_clock))
            delivered=mix.loudness(tuned,settings)
            corrections.append(dict(filter=correction,encoded=delivered,
                path=str(tuned),sha256=mix.sha256_file(tuned),
                measuredHeadroomDb=headroom,finalTrimDb=trim,
                source='lossless controlled premaster; no AAC cascade'))
            partial=tuned
    if not (c.get('socialCompression') or fixed_master) and abs(float(delivered['input_i'])-settings['targetIntegratedLufs'])>.5:
        delta=settings['targetIntegratedLufs']-float(delivered['input_i'])
        gain=max(-1.0,min(delta,1.0,-1.05-float(delivered['input_tp'])))
        if gain*delta>0:
            tuned=out.with_suffix('.encoded-correction.mp4')
            assert not tuned.exists(), 'Preserve encoded-output correction evidence'
            correction=f',volume={gain:.6f}dB'
            mix.run(master_mux_command(paths['picture'],mastering_source,tuned,
                                       normalizer+correction,duration,sample_clock,movie_clock))
            delivered=mix.loudness(tuned,settings)
            corrections.append(dict(filter=correction,encoded=delivered,path=str(tuned),
                sha256=mix.sha256_file(tuned),source='Lossless premaster; correction bounded by measured peak headroom'))
            partial=tuned
    assert abs(float(delivered['input_i'])-settings['targetIntegratedLufs'])<=.5, delivered
    assert float(delivered['input_tp'])<=-1, delivered
    probe=probe_media(partial); ov=probe['video'][0]
    assert len(probe['video'])==1 and len(probe['audio'])==1
    assert (ov['width'],ov['height'],ov['frames'],Fraction(ov['frameRate']))==(v['width'],v['height'],frames,Fraction(24))
    assert probe['audio'][0]['sampleRate']==48000 and probe['audio'][0]['channels']==2
    assert abs(probe['durationSeconds']-duration)<.05
    picture_hash=mix.packet_hash(paths['picture']); assert mix.packet_hash(partial)==picture_hash
    gpu_decode=None
    if fast:
        gpu_decode=gpu_probe.decode_h264_worker(partial,frames)
    else:
        mix.run(['ffmpeg','-v','error','-xerror','-err_detect','explode','-threads','3',
                 '-reinit_filter','0','-i',str(partial),'-vsync','0','-f','null','-'])
    for key,path in paths.items():mix.verify_hash(path,c['inputs'][key]['sha256'],key)
    assert mix.sha256_file(config_path)==config_sha and mix.sha256_file(Path(__file__))==tool_sha
    if fast:assert mix.sha256_file(Path(gpu_probe.__file__))==probe_tool_sha
    partial.replace(out)
    result=dict(kind='source-bound-two-pass-review-master',configSha256=config_sha,
        toolSha256=tool_sha,computeHost=socket.gethostname(),inputs=c['inputs'],
        output=c['output'],outputSha256=mix.sha256_file(out),frames=frames,fps=24,
        resolution=[v['width'],v['height']],runtimeSeconds=duration,probe=probe,
        loudness=delivered,preNormalizationLoudness=measured,normalizer=normalizer,
        fixedGainSocial=fixed_social,fixedGainMaster=fixed_master,initialEncodeFilter=initial_filter,
        postNormalizationGainDb=output_gain,
        socialCompression=compression,
        encodedCorrections=corrections,
        limiterQualification=limiter_qualification,
        picturePacketsUnchanged=True,picturePacketSha256=picture_hash,
        fullDecode=True,creativeApproval=False,finalApproved=False,
        note='Measured mastering only. Separate source-bound studio review and delivery admission are required.')
    if fast:result.update(gpuFinishingProbe=True,gpuDecode=gpu_decode,probeToolSha256=probe_tool_sha)
    if sample_clock:result.update(sampleClockFilter='aformat=channel_layouts=stereo,aresample=48000,asetpts=N/SR/TB',movieTimescale=48000 if movie_clock else None,
        movieTimescalePolicy='explicit48k' if movie_clock else 'retained muxer default; audio remains exact 48kHz sample clock')
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
