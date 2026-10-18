"""Render a silent, work-only picture component while final score is pending.

Uses the shared renderer and exact validated allocation; cannot create a final
edit plan or waive any picture, speech, source-hash or continuation contract.
"""
import argparse
import json
import platform
from pathlib import Path
from assemble_longform import build_picture, bound_path, verify_sha, sha
from build_working_audio import validate_working_clock


def validate(readiness, clock, clock_sha):
    if readiness.get('errors') or any(x.get('type')!='score-admission' for x in readiness.get('missing',[])):
        raise RuntimeError('Unresolved non-score input gate')
    if readiness.get('mode')!='measurement-only-score-gates-preserved':
        raise RuntimeError('Full measured allocation is required')
    if readiness.get('measuredClock',{}).get('sha256')!=clock_sha:
        raise RuntimeError('Readiness refers to a different clock')
    validate_working_clock(clock,readiness['episode'])
    blocks=readiness.get('pictureBlocks',[])
    if not blocks or readiness.get('exactCoverageDeficits'):
        raise RuntimeError('Picture coverage is incomplete')
    cursor=0
    for block in blocks:
        if block['status'] not in ('ready','ready-native-picture-audio-sync') or abs(block['start']*24-cursor)>0.001:
            raise RuntimeError('Picture blocks are not ready and contiguous')
        cursor+=round(block['duration']*24)
    if cursor!=round(clock['runtimeSeconds']*24):
        raise RuntimeError('Picture clock does not cover all speech')


def ownership_contracts(ep, readiness):
    """A film whose every row is a lip-sync take needs no picture ledger."""
    doc=readiness['documents'].get('h3ReviewLedger')
    if doc is None:
        return []
    lp=bound_path(ep,doc['path'],'picture ledger')
    verify_sha(lp,doc['sha256'],'picture ledger')
    return json.loads(lp.read_text()).get('pictureOwnershipContracts',[])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('clock',type=Path)
    p.add_argument('readiness',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();ep=a.episode.resolve()
    if platform.system()!='Linux':raise RuntimeError('Rendering belongs on GCP')
    c=json.loads(a.clock.read_text());r=json.loads(a.readiness.read_text())
    ch=sha(a.clock);rh=sha(a.readiness);validate(r,c,ch)
    plan=dict(c,pictureBlocks=r['pictureBlocks'],pictureOwnershipContracts=ownership_contracts(ep,r),
              transitions=r.get('transitions',[]),frame=r.get('frame'))
    out=a.output.resolve()
    if not out.is_relative_to(ep/'work'):raise RuntimeError('Work-only output required')
    out.mkdir(parents=True,exist_ok=False)
    result=build_picture(ep,out,plan,1280,720)
    verify_sha(a.clock,ch,'clock');verify_sha(a.readiness,rh,'readiness')
    receipt=dict(kind='silent-working-picture-component',finalApproved=False,renderableFinalPlan=False,
                 clockSha256=ch,readinessSha256=rh,runtimeSeconds=c['runtimeSeconds'],
                 output=str(result),sha256=sha(result),finalScoreGatePreserved=True)
    (out/'picture-component-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
