"""Rank newly dark/high-frequency patches in smooth source areas for review."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def decode(path):
    return subprocess.Popen([
        "ffmpeg","-v","error","-reinit_filter","0","-i",str(path),"-an",
        "-vf","select=not(mod(n\\,24)),scale=640:360:flags=area",
        "-fps_mode","passthrough","-pix_fmt","gray","-f","rawvideo","-"
    ],stdout=subprocess.PIPE)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("source",type=Path);ap.add_argument("hd",type=Path);ap.add_argument("output",type=Path)
    ap.add_argument("--review-dir",type=Path,help="Export strongest candidate per600-frame chunk, source left and HD right")
    a=ap.parse_args()
    before=[sha(a.source),sha(a.hd)]
    readers=[decode(a.source),decode(a.hd)];rows=[];count=0;review={}
    try:
        while True:
            raw=[p.stdout.read(640*360) for p in readers]
            if not any(raw): break
            if any(len(x)!=640*360 for x in raw): raise RuntimeError("Sample count mismatch")
            s,t=[np.frombuffer(x,np.uint8).reshape(360,640).astype(np.float32) for x in raw]
            sg=np.abs(np.diff(s,axis=1,append=s[:,-1:]));tg=np.abs(np.diff(t,axis=1,append=t[:,-1:]))
            def pool(x): return x.reshape(45,8,80,8).mean(axis=(1,3))
            darkness=pool(np.maximum(s-t-5,0));source_detail=pool(sg);extra_detail=np.maximum(pool(tg)-source_detail,0)
            score=darkness*extra_detail/(source_detail+1)
            score[source_detail>5]=0
            for position in np.argpartition(score.ravel(),-8)[-8:]:
                y,x=np.unravel_index(position,score.shape)
                rows.append({"frame":count*24,"seconds":count,"score":round(float(score[y,x]),5),
                             "hdBox":[int(x*24),int(y*24),int((x+1)*24),int((y+1)*24)],
                             "sourceDetail":round(float(source_detail[y,x]),4),"darkResidual":round(float(darkness[y,x]),4)})
                chunk=count*24//600
                if a.review_dir and (chunk not in review or rows[-1]["score"]>review[chunk][0]["score"]):
                    # 80px at scan resolution corresponds to240px of final HD.
                    x0=max(0,min(640-80,int(x*8)-36));y0=max(0,min(360-80,int(y*8)-36))
                    pair=np.hstack([s[y0:y0+80,x0:x0+80],t[y0:y0+80,x0:x0+80]]).astype(np.uint8)
                    review[chunk]=(dict(rows[-1],cropHd=[x0*3,y0*3,240,240]),pair)
            count+=1
        for p in readers:
            p.stdout.close()
            if p.wait(): raise RuntimeError("Residual decoder failed")
    finally:
        for p in readers:
            if p.poll() is None:p.kill();p.wait()
    if before!=[sha(a.source),sha(a.hd)]:raise RuntimeError("Input changed during audit")
    rows.sort(key=lambda x:x["score"],reverse=True)
    result={"sourceSha256":before[0],"hdSha256":before[1],"sampledFrames":count,
            "method":"Every24th exact ordinal at640x360; rank8x8 low-source-texture blocks by newly dark, high-horizontal-frequency residual. Diagnostic candidates only, not a perceptual pass/fail gate.",
            "candidates":rows[:500]}
    if a.review_dir:
        a.review_dir.mkdir(parents=True,exist_ok=True)
        review_rows=[]
        for chunk,(row,pair) in sorted(review.items()):
            image=a.review_dir/f"c{chunk:04d}-f{row['frame']}.png"
            subprocess.run(["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","gray","-s","160x80","-i","-","-vf","scale=640:320:flags=neighbor","-frames:v","1",str(image)],input=pair.tobytes(),check=True)
            review_rows.append(dict(row,image=str(image)))
        result["chunkReview"]=review_rows
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"sampledFrames":count,"topCandidates":rows[:20]},indent=2))


if __name__=="__main__":main()
