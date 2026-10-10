"""Versioned audio-only master revision; preserve picture payloads and A/V clocks."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from render_designed_sfx import sha, read, run
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'open_media'))
from hd_audio_remux import remux_approved_audio, packet_hash, streams, start

def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ], text=True))


def loudness(path: Path, target_peak: float = -1.0) -> dict:
    stderr = run([
        "ffmpeg", "-hide_banner", "-i", path, "-vn", "-af",
        f"loudnorm=I=-14:TP={target_peak}:LRA=11:print_format=json",
        "-f", "null", "-",
    ], capture_output=True, text=True).stderr
    return json.loads(stderr[stderr.rfind("{"):stderr.rfind("}") + 1])


def mix(ep,root):
    code_hash=sha(__file__); design_hash=sha(root/'design.json')
    work=ep/'work/assembly-v1';plan=read(work/'edit-plan.json');settings=read(work/'mix-settings.json')
    design=read(root/'design.json');audit=read(root/'mix-audit.json')
    assert sha(root/'design.json')==audit['designSha256'] and sha(root/'sfx-48kf32.wav')==audit['sha256']
    voice=work/'voice-hi-48k24.wav';music=work/'music-wave-48k24.wav';sfx=root/'sfx-48kf32.wav'
    source_hashes={str(p):sha(p) for p in [voice,music,sfx,work/'edit-plan.json',work/'mix-settings.json']}
    events={e['id']:e for e in plan['events']};dips=[]
    for d in settings.get('musicEventDips',[]):
        e=events[d['event']];dips.append((e['start'],e['end'],d['gainDb'],d['entrySeconds'],d['exitSeconds']))
    for c in audit['cues']:
        if c.get('scoreDipDb'):dips.append((c['at'],c['at']+c['duration'],c['scoreDipDb'],.6,1.5))
    env=[]
    for a,b,g,fi,fo in dips:
        assert b>a and g<=0 and fi>0 and fo>0
        env.append(f'if(between(t,{a},{b}),pow(10,({g}*min(1,min((t-{a})/{fi},({b}-t)/{fo})))/20),1)')
    envelope=("volume='"+'*'.join(env)+"':eval=frame,") if env else ''
    graph=(f"[1:a]volume={settings['musicGainDb']}dB,{envelope}asplit=2[md][ms];"
           "[ms]aecho=0.8:0.45:80|160|320:0.10|0.06|0.025,volume=0.22[mr];"
           "[md][mr]amix=inputs=2:normalize=0[space];"
           f"[space][0:a]sidechaincompress=threshold={settings['musicDuckThreshold']}:ratio={settings['musicDuckRatio']}:attack={settings['musicDuckAttackMs']}:release={settings['musicDuckReleaseMs']}[music];"
           "[2:a][0:a]sidechaincompress=threshold=0.035:ratio=4:attack=8:release=360[sfx];"
           "[0:a][music][sfx]amix=inputs=3:normalize=0:dropout_transition=0[out]")
    old720=ep/design['masters']['review'];old1080=ep/design['masters']['hd']
    a=next(x for x in streams(old720) if x['codec_type']=='audio');full_samples=round(float(a['duration'])*48000)
    premaster=root/'premaster-full-48kf32.wav'
    graph=graph.replace('[out]',f"[body];[body]atrim=end_sample={round(plan['runtimeSeconds']*48000)},apad=whole_len={full_samples},atrim=end_sample={full_samples}[out]")
    run(['ffmpeg','-y','-v','error','-i',voice,'-i',music,'-i',sfx,'-filter_complex',graph,
         '-map','[out]',
         '-ar',48000,'-c:a','pcm_f32le',premaster])
    measured=loudness(premaster,-1.5)
    norm=(f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
          f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true")
    new720=root/old720.name;v=next(x for x in streams(old720) if x['codec_type']=='video')
    scale=math.lcm(Fraction(v['time_base']).denominator,48000)
    run(['ffmpeg','-y','-v','error','-copyts','-i',old720,'-i',premaster,'-map','0:v:0','-map','1:a:0',
         '-c:v','copy','-af',norm,'-ar',48000,'-c:a','aac','-b:a','256k','-avoid_negative_ts','disabled',
         '-movie_timescale',scale,'-video_track_timescale',Fraction(v['time_base']).denominator,'-movflags','+faststart',new720])
    assert packet_hash(old720,'0:v:0')==packet_hash(new720,'0:v:0')
    ov=next(s for s in streams(old720) if s['codec_type']=='video');nv=next(s for s in streams(new720) if s['codec_type']=='video')
    assert start(ov)==start(nv)
    result=loudness(new720)
    assert abs(float(result['input_i'])+14)<.7 and float(result['input_tp'])<=-1
    new1080=root/old1080.name
    hd=remux_approved_audio(old1080,new720,new1080)
    assert packet_hash(old1080,'0:v:0')==packet_hash(new1080,'0:v:0')
    for source,digest in source_hashes.items():assert sha(source)==digest
    assert sha(__file__)==code_hash and sha(root/'design.json')==design_hash, 'Inputs or runner changed during render'
    products=[]
    for old,new in [(old720,new720),(old1080,new1080)]:
        data=probe(new);video=next(s for s in data['streams'] if s['codec_type']=='video')
        prior=next(s for s in streams(old) if s['codec_type']=='video')
        assert int(video['nb_frames'])==int(prior['nb_frames']) and Fraction(video['r_frame_rate'])==Fraction(prior['r_frame_rate'])
        assert [video['width'],video['height']]==[prior['width'],prior['height']]
        record=dict(output=str(new),sha256=sha(new),previousPath=str(old),previousSha256=sha(old),
                    previousSidecar=str(old.with_suffix('.json')),resolution=[video['width'],video['height']],
                    fps=float(Fraction(video['r_frame_rate'])),frames=int(video['nb_frames']),duration=float(video['duration']),containerDuration=float(data['format']['duration']),
                    videoStartSeconds=float(video['start_time']),picturePacketsUnchanged=True,
                    soundRevision=str(root),sfxMixAudit=str(root/'mix-audit.json'),sfxPlacements=audit['placements'],
                    sourceHashes=source_hashes,loudness=result,sourcePicturePacketSha256=packet_hash(old,'0:v:0'),
                    decode='pending',finalHumanReview='pending',generatedVideo=False,deterministic=True)
        if video['height']==1080:record['audioRemux']=hd
        new.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');products.append(record)
    report=dict(products=products,sourceHashes=source_hashes,filterGraph=graph,normalizationFilter=norm,
                preNormalizationLoudness=measured,deliveredLoudness=result,
                sfxDuck=dict(threshold=.035,ratio=4,attackMs=8,releaseMs=360),
                musicDips=dips,sourceSettings=settings,codeSha256=code_hash,designSha256=design_hash,
                title='Original duration preserved; added effects end before silent end title.',
                status='Rendered to versioned work directory; verification and publication pending.')
    (root/'master-revision.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(rendered=[p['output'] for p in products],loudness=result),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('root',type=Path);a=p.parse_args();mix(a.episode.resolve(),a.root.resolve())
