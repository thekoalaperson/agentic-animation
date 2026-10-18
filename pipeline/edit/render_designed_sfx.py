"""Render explicit, hash-bound SFX edits; pitch/tempo never touch voice or picture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import socket
import sys
import math
import re
import numpy as np

SR = 48000
if not __debug__:
    raise SystemExit("Optimized Python disables production safety gates")
# What a source batch declares it came from, and the route its receipt names.
# A generated batch declares ``model`` (the qualified Small-SFX model when it
# names none); a recorded or library batch declares ``route`` below instead.
SFX_MODELS = {
    'stabilityai/stable-audio-3-small-sfx': 'self-hosted-text-to-audio/StableAudio3-Small-SFX+deterministic-edits',
}
DEFAULT_SFX_MODEL = 'stabilityai/stable-audio-3-small-sfx'
RECORDED_ROUTE = 'recorded-or-library'
RECORDED_RECEIPT_ROUTE = 'recorded-or-library-sound+deterministic-edits'


def source_route(brief):
    """The declared origin of one batch: a known model, or recorded/library sound."""
    if brief.get('route') == RECORDED_ROUTE:
        if 'model' in brief:
            raise RuntimeError('A recorded or library batch names no generation model')
        for job in brief['jobs']:
            terms = job.get('licence', job.get('terms'))
            if (job.get('prompt') is not None or not str(job.get('origin') or '').strip()
                    or not isinstance(terms, str) or not terms.strip()):
                raise RuntimeError('A recorded or library source has prompt null, an origin and a licence or terms record: '
                                   + str(job.get('id')))
        return RECORDED_ROUTE
    model = brief.get('model', DEFAULT_SFX_MODEL)
    if model not in SFX_MODELS:
        raise RuntimeError('This renderer has no qualified route for SFX model ' + str(model))
    if any(job.get('prompt') is None for job in brief['jobs']):
        raise RuntimeError('A generated SFX job keeps the prompt it was generated from')
    return model


def receipt_route(routes):
    """One receipt route per kind of source, so a receipt says what it mixed."""
    names = sorted({RECORDED_RECEIPT_ROUTE if route == RECORDED_ROUTE else SFX_MODELS[route]
                    for route in routes.values()})
    return ' + '.join(names) if names else SFX_MODELS[DEFAULT_SFX_MODEL]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def run(c,**kw): return subprocess.run(list(map(str,c)),check=True,**kw)
def decode(p,filters=None):
    c=['ffmpeg','-v','error','-i',p]
    if filters:c+=['-af',filters]
    return np.frombuffer(run(c+['-ar',SR,'-ac',2,'-f','f32le','-'],capture_output=True).stdout,dtype='<f4').reshape(-1,2).copy()
def save(p,a):
    assert np.isfinite(a).all()
    run(['ffmpeg','-y','-v','error','-f','f32le','-ar',SR,'-ac',2,'-i','pipe:0','-c:a','pcm_f32le',p],input=a.astype('<f4').tobytes())

def bound_file(ep, spec):
    path=(ep/spec['path']).resolve()
    if not path.is_relative_to(ep.resolve()) or not path.is_file():
        raise RuntimeError('Missing or escaped SFX contract input')
    if sha(path)!=spec['sha256']:
        raise RuntimeError('Changed SFX contract input: '+str(path))
    return path


LEGACY_EMOTION_KEYS = ('awe', 'dread', 'devotion', 'grief', 'hope', 'scale', 'kinetic', 'intimacy')


def emotion_values(row, dimensions=None):
    """The wave's own declared dimensions; a wave that declares none keeps the legacy eight.

    The legacy 'kineticEnergy' spelling still fills 'kinetic'.
    """
    values = row.get('emotions', row.get('values', row))
    if not isinstance(values, dict):
        raise RuntimeError('Emotion map must be an object')
    keys = LEGACY_EMOTION_KEYS if dimensions is None else tuple(dimensions)
    return {key: values.get(key, values.get('kineticEnergy') if key == 'kinetic' else None)
            for key in keys}

def load_contract(ep, m):
    """V2 supports independently verified palettes and scene-bound working clocks."""
    if m.get('version',1) not in (1,2):
        raise RuntimeError('Unsupported SFX contract version')
    if m.get('version',1) == 1:
        batch=ep/m['sourceBatch']; brief=read(batch/'probe-brief.json'); audit=read(batch/'local-audit.json')
        assert audit['technicalPass'] and sha(batch/'probe-brief.json')==m['briefSha256']==audit['briefSha256']
        work=ep/'work/assembly-v1'
        return ({'legacy':(batch,{j['id']:j for j in brief['jobs']},{r['id']:r['sha256'] for r in audit['outputs']})},
                read(work/'edit-plan.json'),read(work/'picture-segments.json'),
                read(ep/'work/dramatic-wave-v1/dramatic-wave.json'),
                dict(editPlanSha256=sha(work/'edit-plan.json'),picturePlanSha256=sha(work/'picture-segments.json'),
                     dramaticWaveSha256=sha(ep/'work/dramatic-wave-v1/dramatic-wave.json'),sourceBriefSha256=sha(batch/'probe-brief.json')))
    clock=bound_file(ep,m['clock']); wave=bound_file(ep,m['wave']); plan=read(clock)
    working=m.get('workingAudioOnly') is True
    if working:
        if m.get('finalApproved') is not False or plan.get('kind')!='measured-speech-clock-not-renderable-edit-plan' or plan.get('renderable') is not False:
            raise RuntimeError('Working SFX requires an explicitly non-final measured clock')
        picture=[]
    else:
        if plan.get('renderable') is not True or plan.get('kind')=='measured-speech-clock-not-renderable-edit-plan':
            raise RuntimeError('Final SFX requires a renderable picture plan')
        picture=read(bound_file(ep,m['picture']))
        position=0.0
        if not picture:
            raise RuntimeError('Final picture cannot be empty')
        for part in picture:
            start=float(part['timelineStart']); duration=float(part['duration'])
            if not math.isfinite(start+duration) or duration<=0 or abs(start-position)>1/48000:
                raise RuntimeError('Final picture must cover the exact continuous clock')
            position=start+duration
        if abs(position-float(plan['runtimeSeconds']))>1/48000:
            raise RuntimeError('Picture duration does not match current clock')
    if plan.get('episode')!=m['episode'] or read(wave).get('clockSha256')!=m['clock']['sha256']:
        raise RuntimeError('SFX wave and episode must bind the exact current clock')
    batches={}; routes={}
    for key,spec in m['sourceBatches'].items():
        brief_path=bound_file(ep,spec['brief']); audit_path=bound_file(ep,spec['audit'])
        if brief_path.parent!=audit_path.parent:
            raise RuntimeError('SFX brief/audit batch mismatch')
        brief=read(brief_path); audit=read(audit_path)
        if not audit['technicalPass'] or audit['briefSha256']!=spec['brief']['sha256'] or brief['generationApiAllowed'] is not False:
            raise RuntimeError('Unverified self-hosted SFX batch')
        routes[key]=source_route(brief)
        ids=[j['id'] for j in brief['jobs']]; verified_ids=[r['id'] for r in audit['outputs']]
        if len(ids)!=len(set(ids)) or len(verified_ids)!=len(set(verified_ids)) or set(ids)!=set(verified_ids):
            raise RuntimeError('Duplicate or incomplete SFX batch identities')
        if any(r.get('verified') is False for r in audit['outputs']):
            raise RuntimeError('Source output explicitly failed verification')
        batches[key]=(brief_path.parent,{j['id']:j for j in brief['jobs']},{r['id']:r['sha256'] for r in audit['outputs']})
    by_shot={e['shot']:e for e in plan['events']}
    if len(by_shot)!=len(plan['events']):
        raise RuntimeError('SFX scene anchors require unique current events')
    for event in plan['events']:
        values=[float(event[k]) for k in ('start','duration','end')]
        if not all(math.isfinite(v) for v in values) or values[0]<0 or values[1]<=0 or abs(values[0]+values[1]-values[2])>1/48000 or values[2]>plan['runtimeSeconds']+1/48000:
            raise RuntimeError('Invalid finite SFX scene clock')
    for c in m['cues']:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',c['id']):
            raise RuntimeError('Unsafe SFX output id')
        if 'at' in c:
            raise RuntimeError('V2 SFX must anchor to scene plus offset, not stale absolute time')
        event=by_shot[c['shot']]; offset=float(c.get('offset',0))
        if not 0<=offset<event['duration']:
            raise RuntimeError('SFX onset outside its declared scene')
        c['at']=event['start']+offset
        maximum=float(c['maxSeconds'])
        if not math.isfinite(maximum) or maximum<=0 or c['at']+maximum>plan['runtimeSeconds']+1/48000:
            raise RuntimeError('Invalid bounded SFX duration')
        if maximum>event['duration']-offset+1/48000 and not str(c.get('tailReason','')).strip():
            raise RuntimeError('A cross-scene SFX tail needs an editorial reason')
    return batches,plan,picture,read(wave),dict(clockSha256=sha(clock),dramaticWaveSha256=sha(wave),
        workingAudioOnly=working,finalApproved=False if working else m.get('finalApproved',False),
        sourceBatches=m['sourceBatches'],picturePlanSha256=None if working else m['picture']['sha256'],
        sourceRoutes=routes)

def render(ep,root):
    if sys.platform!='linux' or not shutil.which('nvidia-smi'):
        raise RuntimeError('Production SFX processing belongs on the qualified GPU worker')
    original_design=(root/'design.json').read_bytes(); design_digest=hashlib.sha256(original_design).hexdigest()
    m=json.loads(original_design)
    if m.get('workingAudioOnly') is True and not root.resolve().is_relative_to((ep/'work').resolve()):
        raise RuntimeError('Working-only SFX must stay outside deliverables')
    assert m['authority'] and m['generationApiAllowed'] is False
    batches,plan,picture,wave,contract=load_contract(ep,m)
    waves={w['shot']:w for w in wave['shots']}
    n=round(plan['runtimeSeconds']*SR); stem=np.zeros((n,2),dtype=np.float32)
    dest=root/'edited';dest.mkdir(exist_ok=True); rows=[]
    assert len({c['id'] for c in m['cues']})==len(m['cues'])
    for c in m['cues']:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',c['id']):
            raise RuntimeError('Unsafe SFX output id')
        batch,jobs,hashes=batches[c.get('sourceBatch','legacy')]
        job=jobs[c['job']];src=batch/job['output'];assert sha(src)==hashes[job['id']]
        assert src.resolve().is_relative_to(batch.resolve())
        assert -12<=c['semitones']<=12 and .5<=c['speed']<=2
        assert 1<=c.get('loops',1)<=4 and c['at']>=0
        rate=round(SR*2**(c['semitones']/12));ratio=rate/SR
        tempo=c['speed']/ratio
        filters=[f'aresample={SR}']
        if c.get('sourceIn') is not None:
            assert 0<=c['sourceIn']<c['sourceOut']<=job['seconds']
            filters += [f"atrim=start={c['sourceIn']}:end={c['sourceOut']}",'asetpts=PTS-STARTPTS']
        filters += [f'asetrate={rate}',f'aresample={SR}']
        while tempo<.5: filters+=['atempo=0.5'];tempo/=.5
        while tempo>2: filters+=['atempo=2'];tempo/=2
        filters += [f'atempo={tempo:.12f}',f"highpass=f={c.get('highpass',45)}",f"lowpass=f={c.get('lowpass',12000)}"]
        a=decode(src,','.join(filters));original_seconds=c.get('sourceOut',job['seconds'])-c.get('sourceIn',0)
        expected=round(original_seconds/c['speed']*SR)
        length_error=(len(a)-expected)/SR
        assert abs(length_error)<.2, (c['id'],length_error)
        if len(a)<expected:a=np.pad(a,((0,expected-len(a)),(0,0)))
        a=a[:expected]
        if c.get('loops',1)>1:
            assert c.get('loopReason') and c.get('crossfade',0)>0
            cross=round(c['crossfade']*SR);assert 0<cross<len(a)/2
            unit=a.copy();ramp=np.linspace(0,1,cross)[:,None]
            for _ in range(c['loops']-1):
                seam=a[-cross:]*(1-ramp)+unit[:cross]*ramp
                a=np.concatenate([a[:-cross],seam,unit[cross:]])
        duration=min(len(a)/SR,c.get('maxSeconds',len(a)/SR));a=a[:round(duration*SR)]
        # Level is authored by scene, not inferred sonic approval.
        blocks=np.sqrt(np.mean(a[:len(a)//4800*4800].reshape(-1,4800,2).astype(float)**2,axis=(1,2)))
        active=float(np.percentile(blocks,90));assert active>1e-7
        gain=float(np.clip(c['activeRmsDb']-20*np.log10(active),-18,12))
        a*=10**(gain/20)
        peak=float(np.max(np.abs(a)));cap=10**(c.get('peakCapDb',-9)/20)
        safety=min(1,cap/max(peak,1e-9));a*=safety
        fi,fo=[min(round(c[k]*SR),len(a)//2) for k in ['fadeIn','fadeOut']]
        if fi:a[:fi]*=np.linspace(0,1,fi)[:,None]
        if fo:a[-fo:]*=np.linspace(1,0,fo)[:,None]
        at=round(c['at']*SR);assert at+len(a)<=n
        stem[at:at+len(a)]+=a
        out=dest/(c['id']+'.wav');save(out,a)
        overlapping=[dict(id=p['id'],start=p['timelineStart'],end=p['timelineStart']+p['duration']) for p in picture if p['timelineStart']<c['at']+len(a)/SR and p['timelineStart']+p['duration']>c['at']]
        w=waves[c['shot']]
        origin={k:job[k] for k in ('origin','licence','terms') if k in job}
        rows.append({**c,'source':str(src),'sourceSha256':sha(src),'sourcePrompt':job['prompt'],**origin,'edited':str(out),'sha256':sha(out),'duration':len(a)/SR,
                     'filters':filters,'tempoOutputLengthCorrectionSeconds':-length_error,'gainDb':gain,'peakSafetyGain':safety,
                     'emotion':emotion_values(w,wave.get('dimensions')),'picture':overlapping})
        print(c['id'],round(len(a)/SR,3),flush=True)
    assert np.max(np.abs(stem))<.95
    if sha(root/'design.json')!=design_digest:
        raise RuntimeError('SFX design changed during render')
    # Revalidate the exact immutable contracts before publishing a success receipt.
    load_contract(ep,json.loads(original_design))
    out=root/'sfx-48kf32.wav';save(out,stem)
    result=dict(route=receipt_route(contract.get('sourceRoutes',{})),authority=m['authority'],
                designSha256=design_digest,rendererSha256=sha(__file__),computeHost=socket.gethostname(),**contract,
                stem=str(out),sha256=sha(out),seconds=n/SR,placements=len(rows),cues=rows,
                sampleRate=SR,channels=2,frames=n,codec='pcm_f32le',peakLinear=float(np.max(np.abs(stem))),
                review=m.get('reviewScope','Delegated studio edit choices; final mixed review pending. Not individual human take verdicts.'),
                pitchMethod='Deterministic asetrate + resample with compensating atempo; speed independently authored. Source originals unchanged.',
                ffmpegVersion=run(['ffmpeg','-version'],capture_output=True,text=True).stdout.splitlines()[0])
    (root/'mix-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(cues=len(rows),stem=str(out),peak=float(np.max(np.abs(stem))))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    render(a.episode.resolve(),a.root.resolve())
