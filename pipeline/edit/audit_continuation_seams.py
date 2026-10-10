#!/usr/bin/env python3
"""GPU-worker native-boundary evidence for source-bound continuation candidates.

This does not grant creative approval. It checks exact source hashes, streams,
full decodability, actual adjacent native frames and a silent seam preview.
"""
import argparse
import hashlib
import json
import math
import platform
import re
import socket
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def run(args):
    return subprocess.run(args, check=True, capture_output=True).stdout


def probe(path):
    doc = json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))
    videos = [s for s in doc['streams'] if s['codec_type']=='video']
    assert len(videos)==1 and not any(s['codec_type']=='audio' for s in doc['streams'])
    v = videos[0]
    assert v['avg_frame_rate']=='24/1'
    return {k:v[k] for k in ('width','height','nb_frames','avg_frame_rate')}


def frames(path, start, end):
    raw = run(['ffmpeg','-v','error','-threads','2','-i',str(path),'-vf',
        f'trim=start_frame={start}:end_frame={end},scale=480:270',
        '-vsync','0','-an','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
    a = np.frombuffer(raw,dtype=np.uint8).reshape((-1,270,480,3))
    assert len(a)==end-start, (path,start,end,len(a))
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path)
    p.add_argument('review',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    tool_path=Path(__file__).resolve(); tool_sha=sha(tool_path)
    assert platform.system()=='Linux', 'Production media decoding must run on GCP, not Mac'
    run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'])
    ep=a.episode.resolve(); review=(ep/a.review).resolve(); out=(ep/a.output).resolve()
    out.relative_to(ep/'work')
    assert not (out/'seam-audit.json').exists(), 'Version completed evidence rather than overwrite'
    out.mkdir(parents=True,exist_ok=True)
    review_sha=sha(review); doc=json.loads(review.read_text()); results=[]
    selected_rows=[r for r in doc['items'] if r['verdict']=='sampled-compatible-used-window-candidate']
    ids=[r['id'] for r in selected_rows]
    assert len(ids)==len(set(ids)), 'Duplicate candidate ID would overwrite seam evidence'
    for row in doc['items']:
        if row['verdict']!='sampled-compatible-used-window-candidate':
            continue
        ident=row['id']; assert re.fullmatch(r'[A-Za-z0-9_-]+',ident)
        b=row['sourceBinding']; use=row['plannedUse']
        parent=(ep/b['parentSilentSourcePath']).resolve(); child=(ep/b['silentPath']).resolve()
        for path,expected in [(parent,b['parentSilentSourceSha256']),(child,b['silentSha256'])]:
            path.relative_to(ep); assert sha(path)==expected
            run(['ffmpeg','-v','error','-xerror','-err_detect','explode','-threads','2',
                 '-i',str(path),'-map','0:v:0','-an','-f','null','-'])
        pi=probe(parent); ci=probe(child)
        assert (pi['width'],pi['height'])==(ci['width'],ci['height'])
        endpoint=b['parentAnchorFrame']; count=use['lastFrameInclusive']
        assert endpoint==int(pi['nb_frames'])-1 and 1<=count<int(ci['nb_frames'])
        assert use['firstFrame']==1 and use['editorialSpeed']==1
        assert use['frames']==count and use['nativeFps']==24
        assert use['dropConditioningFrame0'] is True and use['contiguousParentExtensionOnly'] is True
        assert abs(use['seconds']-count/24)<1e-6
        left=frames(parent,max(0,endpoint-11),endpoint+1)
        right=frames(child,0,min(count,12)+1)
        selected=np.concatenate([left,right[1:]])
        diffs=np.mean(np.abs(np.diff(selected.astype(np.float32),axis=0)),axis=(1,2,3))
        boundary=float(diffs[len(left)-1])
        ordinary=np.delete(diffs,len(left)-1)
        median=float(np.median(ordinary)) if len(ordinary) else 0
        anchor_difference=float(np.mean(np.abs(left[-1].astype(np.float32)-right[0])))
        labels=[f'parent f{endpoint-len(left)+1+i}' for i in range(len(left))]+[
            f'child f{i} USED' for i in range(1,len(right))]
        tiles=math.ceil(len(selected)/4)
        sheet=Image.new('RGB',(4*480,tiles*294),(18,18,18)); draw=ImageDraw.Draw(sheet)
        for i,(frame,label) in enumerate(zip(selected,labels)):
            x=(i%4)*480;y=(i//4)*294
            sheet.paste(Image.fromarray(frame),(x,y));draw.text((x+6,y+272),label,fill='white')
        sheet_path=out/(ident+'-native-seam.jpg');sheet.save(sheet_path,quality=94)
        preview=out/(ident+'-native-seam.mp4')
        graph=(f'[0:v]trim=start_frame={max(0,endpoint-23)}:end_frame={endpoint+1},setpts=PTS-STARTPTS[a];'
            f'[1:v]trim=start_frame=1:end_frame={min(count,24)+1},setpts=PTS-STARTPTS[b];'
            '[a][b]concat=n=2:v=1:a=0,format=yuv420p[v]')
        run(['ffmpeg','-v','error','-y','-threads','2','-i',str(parent),'-i',str(child),
             '-filter_complex',graph,'-map','[v]','-an','-c:v','libx264','-threads','2',
             '-preset','fast','-crf','16','-r','24','-movflags','+faststart',str(preview)])
        for path,expected in [(parent,b['parentSilentSourceSha256']),(child,b['silentSha256'])]:
            assert sha(path)==expected
        results.append(dict(id=ident,sceneId=row['sceneId'],sourceBinding=b,plannedUse=use,
            parentProbe=pi,childProbe=ci,fullSourcesDecode=True,
            boundaryMeanAbsoluteRgbDifference=boundary,neighborMedianDifference=median,
            boundaryToNeighborRatio=boundary/max(median,1e-6),conditioningFrameDifference=anchor_difference,
            nativeAdjacentDifferences=diffs.tolist(),
            advisoryFlag=boundary>max(12,median*4),
            grid=dict(path=str(sheet_path.relative_to(ep)),sha256=sha(sheet_path)),
            preview=dict(path=str(preview.relative_to(ep)),sha256=sha(preview))))
        print(json.dumps(dict(id=ident,boundaryRatio=results[-1]['boundaryToNeighborRatio'],
            advisoryFlag=results[-1]['advisoryFlag'])),flush=True)
    assert sha(review)==review_sha
    assert sha(tool_path)==tool_sha, 'Seam tool changed during execution'
    result=dict(kind='source-bound-native-continuation-seam-audit',reviewSha256=review_sha,
        toolSha256=tool_sha,computeHost=socket.gethostname(),
        computeMethod='Software FFmpeg decoding/encoding on qualified GPU worker; no Mac media processing',
        continuousPlaybackPerformed=False,creativeApproval=False,
        limitations='Adjacent-frame differences and full decode are technical diagnostics, not perceptual smoothness certification.',
        items=results)
    (out/'seam-audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
