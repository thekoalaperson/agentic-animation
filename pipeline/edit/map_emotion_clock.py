"""Map authored emotion anchors to a measured film clock; no audio inference."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def map_clock(clock, anchors):
    events=clock['events'];by_id={e['id']:e for e in events}
    dims=anchors['dimensions'];points=[]
    if not dims or len(set(dims))!=len(dims) or not anchors.get('anchors'):
        raise ValueError('Non-empty unique dimensions and authored anchors are required')
    if len(by_id)!=len(events):raise ValueError('Duplicate event identity')
    for a in anchors['anchors']:
        e=by_id[a['shotId']]
        if set(a['values'])!=set(dims) or any(not 0<=v<=100 for v in a['values'].values()):
            raise ValueError('Invalid authored emotion dimensions')
        points.append(dict(time=e['start'],shot=e['shot'],values=a['values']))
    if any(a['time']>=b['time'] for a,b in zip(points,points[1:])):
        raise ValueError('Anchors must be in film order')
    times=[p['time'] for p in points];hard=set(anchors.get('hardTransitions',[]));rows=[]
    for e in events:
        i=max(0,bisect.bisect_right(times,e['start'])-1);a=points[i]
        b=points[min(i+1,len(points)-1)]
        t=0 if a is b or b['shot'] in hard else max(0,min(1,(e['start']-a['time'])/(b['time']-a['time'])))
        blend=t*t*(3-2*t)
        rows.append(dict(id=e['id'],shot=e['shot'],start=e['start'],end=e['end'],
                         values={d:round(a['values'][d]+blend*(b['values'][d]-a['values'][d]),3) for d in dims}))
    return dict(kind='authored-emotion-clock-not-inferred-audio',episode=clock.get('episode'),dimensions=dims,
                runtimeSeconds=clock['runtimeSeconds'],anchors=points,shots=rows,
                method='Smoothstep between actual timed anchors; hold before explicit hard scene transitions, then reset to the authored new anchor. Each dimension remains independent.',
                listeningVerdict=None)


def main():
    p=argparse.ArgumentParser();p.add_argument('clock',type=Path);p.add_argument('anchors',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();result=map_clock(json.loads(a.clock.read_text()),json.loads(a.anchors.read_text()))
    result['clockSha256']=digest(a.clock)
    result['inputs']={'clockSha256':result['clockSha256'],'anchorsSha256':digest(a.anchors)}
    text=json.dumps(result,indent=2)+'\n'
    if a.output.exists() and a.output.read_text()!=text:raise RuntimeError('Use a new version')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(json.dumps({'shots':len(result['shots']),'dimensions':len(result['dimensions']),'runtimeSeconds':result['runtimeSeconds']}))


if __name__=='__main__':main()
