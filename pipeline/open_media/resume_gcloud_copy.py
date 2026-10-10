#!/usr/bin/env python3
"""Resume one large VM-to-local copy in independently retried binary chunks."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess
import time


def digest(path, limit=None):
    h=hashlib.sha256();remaining=limit
    with path.open('rb') as f:
        while remaining is None or remaining>0:
            block=f.read(1024*1024 if remaining is None else min(1024*1024,remaining))
            if not block:break
            h.update(block)
            if remaining is not None:remaining-=len(block)
    return h.hexdigest()


def ssh(vm,project,zone,command,binary=False):
    args=['gcloud','compute','ssh',vm,'--project',project,'--zone',zone,'--quiet','--command',command]
    return subprocess.run(args,capture_output=True,text=not binary,timeout=900)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('remote');ap.add_argument('local',type=Path)
    ap.add_argument('--vm',required=True);ap.add_argument('--project',required=True)
    ap.add_argument('--zone',required=True);ap.add_argument('--chunk-mib',type=int,default=64)
    ap.add_argument('--sha256',required=True)
    a=ap.parse_args();remote=shlex.quote(a.remote);a.local.parent.mkdir(parents=True,exist_ok=True)
    size_result=ssh(a.vm,a.project,a.zone,f'stat -c %s {remote}')
    if size_result.returncode:raise RuntimeError(size_result.stderr)
    total=int(size_result.stdout.strip());offset=a.local.stat().st_size if a.local.exists() else 0
    if offset>total:raise ValueError('Local file exceeds remote size')
    if offset:
        remote_prefix=ssh(a.vm,a.project,a.zone,f'head -c {offset} {remote} | sha256sum')
        if remote_prefix.returncode:raise RuntimeError(remote_prefix.stderr)
        if digest(a.local)!=remote_prefix.stdout.split()[0]:raise ValueError('Existing local prefix does not match remote')
        print(f'validated prefix {offset}/{total}',flush=True)
    chunk=a.chunk_mib*1024*1024
    with a.local.open('r+b' if a.local.exists() else 'wb') as out:
        while offset<total:
            count=min(chunk,total-offset);result=None
            command=f'dd if={remote} iflag=skip_bytes,count_bytes skip={offset} count={count} status=none'
            for attempt in range(1,6):
                result=ssh(a.vm,a.project,a.zone,command,binary=True)
                if result.returncode==0 and len(result.stdout)==count:break
                print(f'retry offset={offset} attempt={attempt} bytes={len(result.stdout) if result else 0}',flush=True)
                time.sleep(3)
            if result is None or result.returncode or len(result.stdout)!=count:
                raise RuntimeError('Chunk transfer failed at offset '+str(offset))
            out.seek(offset);out.write(result.stdout);out.flush();offset+=count
            print(f'copied {offset}/{total}',flush=True)
    actual=digest(a.local)
    if actual!=a.sha256:raise ValueError(f'Final SHA mismatch: {actual}')
    print(f'verified {actual} {a.local}',flush=True)


if __name__=='__main__':main()
