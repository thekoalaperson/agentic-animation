"""GPU-worker-only, source-bound contact sheets for authored native-frame windows."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(audit, windows, output):
    if platform.system() != 'Linux':
        raise RuntimeError('Media review belongs on the qualified GPU worker')
    subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],check=True,capture_output=True)
    rows={r['id']:r for r in json.loads(audit.read_text())['outputs']}
    plans=json.loads(windows.read_text())
    if output.exists():
        raise ValueError('Choose a new diagnostic output directory')
    output.mkdir(parents=True)
    result=[]
    for item in plans:
        source=rows[item['id']]['silent']
        path=Path(source['file'])
        assert sha(path)==source['sha256']
        first,last,stride=item['first'],item['last'],item['stride']
        assert 0<=first<=last<source['frames'] and stride>=1
        frames=sorted(set(range(first,last+1,stride))|{last})
        assert len(frames)<=60
        columns=6
        selection="select='"+'+'.join(f'eq(n,{n})' for n in frames)+"'"
        target=output/f"{item['id']}-{first}-{last}.jpg"
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-vf',
                        f'{selection},scale=448:-1,tile={columns}x{(len(frames)+columns-1)//columns}',
                        '-frames:v','1',str(target)],check=True)
        result.append(dict(**item,frameOrdinals=frames,sourceSha256=source['sha256'],
                           sheet=target.name,sheetSha256=sha(target),verdict='pending'))
    (output/'audit.json').write_text(json.dumps(dict(sourceAuditSha256=sha(audit),
        windowsSha256=sha(windows),scriptSha256=sha(Path(__file__)),computeHost=platform.node(),
        scope='Native-frame diagnostic windows; not continuous-watch certification',outputs=result),indent=2)+'\n')
    print(json.dumps(dict(completed=len(result),output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path);p.add_argument('windows',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.audit,a.windows,a.output)
