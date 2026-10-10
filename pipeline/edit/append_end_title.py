#!/usr/bin/env python3
"""Append the standard deterministic 6.5s title card without re-encoding the film body."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile


def run(*args, capture=False):
    return subprocess.run([str(x) for x in args],check=True,capture_output=capture,text=capture)


def probe(path, count=False):
    args=['ffprobe','-v','error']
    if count:args+=['-count_packets']
    args+=['-show_streams','-show_format','-of','json',path]
    return json.loads(subprocess.check_output([str(x) for x in args],text=True))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def packet_hash(path, stream, frames=None):
    cmd=['ffmpeg','-v','error','-i',path,'-map',stream]
    if frames is not None:cmd += [('-frames:v' if stream.startswith('0:v') else '-frames:a'),str(frames)]
    cmd+=['-c','copy','-f','hash','-hash','sha256','-']
    return subprocess.check_output([str(x) for x in cmd],text=True).strip()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('core',type=Path);ap.add_argument('card',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--seconds',type=float,default=6.5);a=ap.parse_args()
    if platform.system()!='Linux': raise RuntimeError('End-card media rendering belongs on GCP')
    run('nvidia-smi','--query-gpu=name','--format=csv,noheader',capture=True)
    if a.output.exists(): raise ValueError('Preserve earlier masters; use a new output path')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    source=probe(a.core,True);sv=next(x for x in source['streams'] if x['codec_type']=='video')
    sa=next(x for x in source['streams'] if x['codec_type']=='audio')
    core_frames=int(sv['nb_frames']);core_audio_packets=int(sa['nb_read_packets'])
    if sv['codec_name'] != 'h264' or int(sv['nb_read_packets']) != core_frames:
        raise ValueError('Expected one H264 MP4 sample per coded frame')
    width,height=int(sv['width']),int(sv['height'])
    fps=Fraction(sv['r_frame_rate'])
    if fps <= 0: raise ValueError('Invalid source frame rate')
    fps_arg=f'{fps.numerator}/{fps.denominator}'
    time_base=Fraction(sv['time_base'])
    track_timescale=time_base.denominator if time_base.numerator == 1 else 12288
    added_frames=round(a.seconds*float(fps))
    with tempfile.TemporaryDirectory(prefix='end-title-',dir=a.output.parent) as td:
        td=Path(td);segment=td/'title.mp4';concat=td/'concat.txt'
        run('ffmpeg','-y','-v','error','-loop','1','-framerate',fps_arg,'-i',a.card,
            '-f','lavfi','-i',f'anullsrc=r={sa["sample_rate"]}:cl=stereo','-t',f'{a.seconds:.6f}',
            '-vf',f'fade=t=in:d=0.35,fade=t=out:st={a.seconds-1.3:.3f}:d=1.3',
            '-c:v','libx264','-preset','slow','-crf','17','-profile:v','high',
            '-pix_fmt','yuv420p','-r',fps_arg,'-video_track_timescale',str(track_timescale),
            '-c:a','aac','-b:a','256k','-ar',str(sa['sample_rate']),'-ac','2','-shortest',segment)
        concat.write_text(f"file '{a.core.resolve()}'\nduration {core_frames/float(fps):.12f}\nfile '{segment.resolve()}'\nduration {added_frames/float(fps):.12f}\n")
        run('ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',concat,
            '-map','0:v:0','-map','0:a:0','-c','copy','-movflags','+faststart',a.output)
    final=probe(a.output,True);fv=next(x for x in final['streams'] if x['codec_type']=='video')
    fa=next(x for x in final['streams'] if x['codec_type']=='audio')
    if [int(fv['width']),int(fv['height'])]!=[width,height] or Fraction(fv['r_frame_rate'])!=fps:
        raise ValueError('Final picture contract mismatch')
    if int(fv['nb_frames'])!=core_frames+added_frames or int(fv['nb_read_packets'])!=int(fv['nb_frames']):
        raise ValueError('Final frame count mismatch')
    if fa['codec_name']!='aac' or fa['sample_rate']!='48000' or int(fa['channels'])!=2:
        raise ValueError('Final audio contract mismatch')
    if packet_hash(a.core,'0:v:0')!=packet_hash(a.output,'0:v:0',core_frames):
        raise ValueError('Original picture packets changed')
    if packet_hash(a.core,'0:a:0')!=packet_hash(a.output,'0:a:0',core_audio_packets):
        raise ValueError('Original approved audio packets changed')
    decoded=run('ffmpeg','-v','error','-xerror','-reinit_filter','0','-i',a.output,'-vsync','0','-f','null','-',capture=True)
    if decoded.stderr.strip(): raise ValueError(f'Final decode errors: {decoded.stderr}')
    record={'output':str(a.output),'sha256':sha(a.output),'core':str(a.core),
            'coreSha256':sha(a.core),'card':str(a.card),'cardSha256':sha(a.card),
            'titleSeconds':a.seconds,'resolution':[width,height],'fps':float(fps),
            'frames':int(fv['nb_frames']),'duration':float(fv['duration']),
            'containerDuration':float(final['format']['duration']),
            'videoStartSeconds':float(fv.get('start_time',0)),
            'frameCountMethod':'H264 MP4 sample count matched packet count, unchanged body packets and full final decode',
            'bodyPicturePacketsUnchanged':True,'bodyAudioPacketsUnchanged':True,
            'decode':'passed','generatedVideo':False,'deterministic':True}
    a.output.with_suffix('.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
