#!/usr/bin/env python3
"""Audit custom HD cache layout and export exact-ordinal before/after sheets on GCP."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from hd_range_cache import digest_file, range_layout, layout_fingerprint, PIXEL_CONTRACT
from hd_audio_remux import packet_hash, audio_packet_clock, streams, start


def run(cmd, **kw):
    return subprocess.run([str(x) for x in cmd],check=True,**kw)


def sample(path, ordinals):
    select = '+'.join(f'eq(n,{n})' for n in ordinals)
    data = subprocess.check_output(['ffmpeg','-v','error','-threads','4','-reinit_filter','0',
        '-i',str(path),'-an','-vf',f"select='{select}',scale=480:270:flags=area",
        '-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','-'])
    if len(data) != len(ordinals)*480*270*3:
        raise ValueError('Exact ordinal sheet sample count differs')
    return np.frombuffer(data,np.uint8).reshape(len(ordinals),270,480,3)


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','hd','range_plan','cache','output','font'): ap.add_argument(name,type=Path)
    a=ap.parse_args()
    if sys.platform != 'linux': raise RuntimeError('Run on qualified GPU host, never Mac')
    run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'])
    plan=json.loads(a.range_plan.read_text())
    layout=range_layout(plan['totalFrames'],600,plan)
    cfg=json.loads((a.cache/'configuration.json').read_text())
    if cfg['rangeLayoutSha256'] != layout_fingerprint(layout): raise ValueError('Range fingerprint differs')
    if cfg['sourceVideoSha256'] != plan['sourceVideoSha256'] or packet_hash(a.source,'0:v:0') != plan['sourceVideoSha256']:
        raise ValueError('Range/source video identity differs')
    source_sha,hd_sha=digest_file(a.source),digest_file(a.hd)
    body_audit=json.loads(a.hd.with_suffix('.json').read_text())
    if body_audit['inputSha256']!=source_sha or body_audit['outputSha256']!=hd_sha:
        raise ValueError('Body output audit hashes differ')
    evidence=[]
    for shard in range(cfg['shards']):
        latest={}
        for line in (a.cache/f'state-shard-{shard:02d}.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row.get('time',0)>=latest.get(row.get('chunk'),{}).get('time',0): latest[row['chunk']]=row
        for i,r in enumerate(layout):
            if i%cfg['shards']!=shard: continue
            state=latest.get(r['id'],{})
            expected={k:v for k,v in cfg.items() if k not in ('shards','chunkFrames','rangeLayoutSha256')}
            expected.update(startFrame=r['startFrame'],frames=r['frames'],sourceContainerSha256=source_sha,
                            sourcePixelContract=PIXEL_CONTRACT,status='complete')
            if any(state.get(k)!=v for k,v in expected.items()): raise ValueError(f'Range contract failed: {r["id"]}')
            if len(state.get('sourceRgbSha256',''))!=64: raise ValueError('Missing decoded-pixel identity')
            if digest_file(a.cache/'chunks'/f'{r["id"]}.mp4') != state['sha256']:
                raise ValueError(f'Cache output hash differs: {r["id"]}')
            evidence.append({**r,'shard':shard,'sha256':state['sha256'],'sourceRgbSha256':state['sourceRgbSha256']})
    sv=next(s for s in streams(a.source) if s['codec_type']=='video')
    hv=next(s for s in streams(a.hd) if s['codec_type']=='video')
    if (int(hv['width']),int(hv['height']),Fraction(hv['r_frame_rate']),int(hv['nb_frames']))!=(1920,1080,Fraction(24),plan['totalFrames']):
        raise ValueError('Full HD geometry/frame count differs')
    if abs(start(sv)-start(hv))>Fraction(sv['time_base']): raise ValueError('Picture start differs')
    if packet_hash(a.source,'0:a:0')!=packet_hash(a.hd,'0:a:0') or audio_packet_clock(a.source)!=audio_packet_clock(a.hd):
        raise ValueError('Source AAC payload or clocks differ')
    a.output.mkdir(parents=True,exist_ok=True)
    ordinals=[r['startFrame']+r['frames']//2 for r in layout]
    before,after=sample(a.source,ordinals),sample(a.hd,ordinals)
    sheets=[]
    # Two source/HD pairs per row, three rows: six ranges per1920x900 sheet.
    for base in range(0,len(layout),6):
        canvas=np.full((900,1920,3),18,dtype=np.uint8)
        filters=[]
        for j in range(min(6,len(layout)-base)):
            x=(j%2)*960; y=(j//2)*300
            canvas[y+30:y+300,x:x+480]=before[base+j]
            canvas[y+30:y+300,x+480:x+960]=after[base+j]
            r=layout[base+j]
            label=f"{r['id']} {r.get('sceneId','')} frame{ordinals[base+j]}  ORIGINAL LEFT | HD RIGHT"
            filters.append(f"drawtext=fontfile={a.font}:text='{label}':fontsize=17:fontcolor=white:x={x+10}:y={y+5}")
        path=a.output/f'range-sheet-{base//6+1:02d}.png'
        run(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x900','-i','-',
             '-vf',','.join(filters),'-frames:v','1','-threads','2',path],input=canvas.tobytes())
        sheets.append({'path':str(path),'sha256':digest_file(path),'rangeIds':[r['id'] for r in layout[base:base+6]]})
    if source_sha!=digest_file(a.source) or hd_sha!=digest_file(a.hd): raise ValueError('Input mutated during review export')
    result={'kind':'source-bound-custom-hd-range-completion','sourceSha256':source_sha,'hdSha256':hd_sha,
            'sourceVideoSha256':plan['sourceVideoSha256'],'rangePlanSha256':digest_file(a.range_plan),
            'configurationSha256':digest_file(a.cache/'configuration.json'),'rangeLayoutSha256':cfg['rangeLayoutSha256'],
            'ranges':sorted(evidence,key=lambda x:x['startFrame']),'rangeCount':len(layout),'frames':plan['totalFrames'],
            'sourceAacPacketsAndClocksUnchanged':True,'sheets':sheets,'technicalChecksPassed':True,
            'visualApproval':False,'limits':'Midframe comparisons are sampled visual diagnostics, not a full temporal watch or automated perceptual pass.'}
    (a.output/'range-completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'ranges':len(layout),'frames':plan['totalFrames'],'sheets':len(sheets),'technicalChecksPassed':True}))


if __name__=='__main__':main()
