"""Map authored emotion anchors to a measured film clock; no audio inference."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path

HELP = """\
CLOCK is a measured clock or edit-plan.json with events [{id, shot, start, end}].
ANCHORS is {dimensions: [your own axis names], anchors: [{shotId, values}],
optional hardTransitions: [shot numbers], optional musicEnergy}. The dimensions
are the film's own; none is required.

musicEnergy (optional) also writes shots[].music_energy, the per-shot score
level assemble_longform.py reads from --wave:
  {"fromDimension": "<one of your dimensions>"}  follow that axis, or
  {"perAnchor": true}  each anchor carries its own musicEnergy 0-100,
                       blended between anchors like any dimension.
Without it the output is unchanged and carries no music_energy.

Give the edit plan as CLOCK when the wave feeds the final mix: the output's
clockSha256 is CLOCK's hash and its shots follow CLOCK's events in order,
which is what mix_absolute_sfx_longform.py checks."""


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def _energy_spec(anchors, dims):
    spec=anchors.get('musicEnergy')
    if spec is None:return None
    if spec=={'perAnchor':True}:
        if any(not isinstance(a.get('musicEnergy'),(int,float)) or isinstance(a.get('musicEnergy'),bool)
               or not 0<=a['musicEnergy']<=100 for a in anchors['anchors']):
            raise ValueError('perAnchor musicEnergy needs a 0-100 musicEnergy on every anchor')
        return spec
    if isinstance(spec,dict) and set(spec)=={'fromDimension'} and spec['fromDimension'] in dims:
        return spec
    raise ValueError('musicEnergy must be {"fromDimension": <declared dimension>} or {"perAnchor": true}')


def map_clock(clock, anchors):
    events=clock['events'];by_id={e['id']:e for e in events}
    dims=anchors['dimensions'];points=[]
    if not dims or len(set(dims))!=len(dims) or not anchors.get('anchors'):
        raise ValueError('Non-empty unique dimensions and authored anchors are required')
    if len(by_id)!=len(events):raise ValueError('Duplicate event identity')
    energy=_energy_spec(anchors,dims)
    for a in anchors['anchors']:
        e=by_id[a['shotId']]
        if set(a['values'])!=set(dims) or any(not 0<=v<=100 for v in a['values'].values()):
            raise ValueError('Invalid authored emotion dimensions')
        point=dict(time=e['start'],shot=e['shot'],values=a['values'])
        if energy=={'perAnchor':True}:point['musicEnergy']=a['musicEnergy']
        points.append(point)
    if any(a['time']>=b['time'] for a,b in zip(points,points[1:])):
        raise ValueError('Anchors must be in film order')
    times=[p['time'] for p in points];hard=set(anchors.get('hardTransitions',[]));rows=[]
    for e in events:
        i=max(0,bisect.bisect_right(times,e['start'])-1);a=points[i]
        b=points[min(i+1,len(points)-1)]
        t=0 if a is b or b['shot'] in hard else max(0,min(1,(e['start']-a['time'])/(b['time']-a['time'])))
        blend=t*t*(3-2*t)
        row=dict(id=e['id'],shot=e['shot'],start=e['start'],end=e['end'],
                 values={d:round(a['values'][d]+blend*(b['values'][d]-a['values'][d]),3) for d in dims})
        if energy is not None:
            row['music_energy']=(row['values'][energy['fromDimension']] if 'fromDimension' in energy
                                 else round(a['musicEnergy']+blend*(b['musicEnergy']-a['musicEnergy']),3))
        rows.append(row)
    result=dict(kind='authored-emotion-clock-not-inferred-audio',episode=clock.get('episode'),dimensions=dims,
                runtimeSeconds=clock['runtimeSeconds'],anchors=points,shots=rows,
                method='Smoothstep between actual timed anchors; hold before explicit hard scene transitions, then reset to the authored new anchor. Each dimension remains independent.',
                listeningVerdict=None)
    if energy is not None:result['musicEnergy']=energy
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__,epilog=HELP,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('clock',type=Path);p.add_argument('anchors',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();result=map_clock(json.loads(a.clock.read_text()),json.loads(a.anchors.read_text()))
    result['clockSha256']=digest(a.clock)
    result['inputs']={'clockSha256':result['clockSha256'],'anchorsSha256':digest(a.anchors)}
    text=json.dumps(result,indent=2)+'\n'
    if a.output.exists() and a.output.read_text()!=text:raise RuntimeError('Use a new version')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(json.dumps({'shots':len(result['shots']),'dimensions':len(result['dimensions']),'runtimeSeconds':result['runtimeSeconds']}))


if __name__=='__main__':main()
