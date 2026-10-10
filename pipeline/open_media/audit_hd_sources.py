#!/usr/bin/env python3
"""Audit copied HD sources against immutable input and media contracts.

Partial runs are explicitly incomplete. Reuse decode evidence only when source,
output, model and frame-contract hashes are unchanged. No GPU/process changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def contract(item):
    native=Fraction(item['probe']['avg_frame_rate'])
    factor=Fraction(item['targetFps'],1)/native
    if factor.denominator != 1: raise ValueError('Non-integer finishing cadence')
    count=min(int(item['probe']['nb_read_frames']),math.ceil(item['requiredSeconds']*float(native))+1)
    return {'width':item['targetResolution'][0],'height':item['targetResolution'][1],
            'fps':item['targetFps'],'frames':count*int(factor),'nativeFrames':count,
            'duration':count*int(factor)/item['targetFps']}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('batch',type=Path)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--require-complete',action='store_true')
    args=ap.parse_args();root=args.batch.resolve()
    manifest=root/'manifest.json';m=json.loads(manifest.read_text())
    auditpath=root/'local-source-audit.json'
    old=json.loads(auditpath.read_text()).get('items',{}) if auditpath.exists() else {}
    state={}
    for line in (root/'state.jsonl').read_text().splitlines():
        r=json.loads(line);state[r['id']]=r
    ready=[];pending=[]
    for item in m['items']:
        r=state.get(item['id'],{})
        if r.get('status')=='completed' and (root/item['output']).exists(): ready.append(item)
        else: pending.append(item['id'])

    def check(item):
        key=item['id'];record=state[key];path=root/item['output'];expected=contract(item)
        if sha(root/item['source']) != item['sha256']: raise ValueError(key+' native source changed')
        outputhash=sha(path)
        if outputhash != record['sha256']: raise ValueError(key+' output hash differs from remote record')
        if record.get('fallback') is not False or record['sourceSha256'] != item['sha256']:
            raise ValueError(key+' neural lineage differs')
        if record['modelSha256'] != '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1':
            raise ValueError(key+' unexpected reconstruction model')
        if record['rifeSha256'] != '6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2':
            raise ValueError(key+' unexpected interpolation checkpoint')
        identity=hashlib.sha256(json.dumps([item,expected,record['modelSha256'],record['rifeSha256'],outputhash],sort_keys=True).encode()).hexdigest()
        cached=old.get(key,{})
        if cached.get('identity')==identity and cached.get('decode')=='passed': return key,cached
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','2','-count_frames','-show_streams','-of','json',str(path)]))
        video=[s for s in info['streams'] if s['codec_type']=='video']
        if len(video)!=1 or any(s['codec_type']=='audio' for s in info['streams']):
            raise ValueError(key+' expected one silent picture stream')
        v=video[0]
        actual={'width':v['width'],'height':v['height'],'fps':float(Fraction(v['avg_frame_rate'])),
                'frames':int(v['nb_read_frames']),'duration':float(v['duration'])}
        for field in ('width','height','fps','frames'):
            if actual[field]!=expected[field] or record[field]!=expected[field]:
                raise ValueError(key+' '+field+' mismatch')
        if abs(actual['duration']-expected['duration'])>.0001:
            raise ValueError(key+' duration mismatch')
        subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(path),'-map','0:v:0','-f','null','-'],check=True)
        return key,{'identity':identity,'sha256':outputhash,'sourceSha256':item['sha256'],
                    'expected':expected,'actual':actual,'decode':'passed','time':time.time()}

    results={};failures={}
    with ThreadPoolExecutor(max_workers=max(1,min(args.workers,4))) as pool:
        jobs={pool.submit(check,item):item['id'] for item in ready}
        for job in as_completed(jobs):
            key=jobs[job]
            try:
                key,result=job.result();results[key]=result
                print('verified',key,flush=True)
            except Exception as exc:
                failures[key]=str(exc);print('FAILED',key,str(exc),flush=True)
    report={'manifestSha256':sha(manifest),'expected':len(m['items']),'verified':len(results),
            'complete':len(results)==len(m['items']) and not failures,'pending':pending,'failures':failures,
            'items':results,'visualReview':'separate sampled inspection; this report is technical only','time':time.time()}
    auditpath.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('expected','verified','complete','pending','failures')}),flush=True)
    if failures or (args.require_complete and not report['complete']): raise SystemExit(1)


if __name__=='__main__': main()
