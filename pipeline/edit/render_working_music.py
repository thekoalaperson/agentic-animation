"""Render an explicitly provisional score using the approved strong-balance DSP.

Never creates an edit-plan, listening verdict or canonical deliverable.
Without --audition this runs on the qualified GPU worker. --audition renders
on any host with ffmpeg into `<outputDirectory>-audition-not-final`, records
the real host and ffmpeg build, and writes a receipt of kind
working-score-audition-not-final that no stem or master step accepts.
"""
import argparse
import json
import platform
from pathlib import Path
import sys
import numpy as np
from build_music_balance_review import shaped_score
from build_score_sfx_revision import bound
from build_sound_revision_review import SR, sha, read, run, write_wav
from build_working_audio import validate_working_clock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common

AUDITION = '-audition-not-final'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('config',type=Path)
    p.add_argument('--audition',action='store_true',help='Render on this host; output is a labelled, inadmissible audition')
    a=p.parse_args(argv);ep=a.episode.resolve();c=read(a.config)
    if a.audition:
        host=_common.preview_host_record(None)
    else:
        if platform.system()!='Linux':raise RuntimeError('Media processing belongs on GCP')
        run(['nvidia-smi','-L'])
    if c.get('finalApproved') is not False or c.get('kind')!='working-score-structural-selection':
        raise RuntimeError('Only explicitly provisional score selections accepted')
    clock=read(bound(ep,c['clock']));validate_working_clock(clock,c['episode'])
    root=(ep/(c['outputDirectory']+(AUDITION if a.audition else ''))).resolve()
    if not root.is_relative_to(ep/'work'):raise RuntimeError('Working output escapes work')
    root.mkdir(parents=True,exist_ok=False)
    n=round(clock['runtimeSeconds']*24)*2000
    for cue in c['music']:
        if not -32<=cue['activeRmsDb']<=-18 or cue['atFrame']<0 or cue['frames']<=0:
            raise RuntimeError('Invalid authored cue contract')
    sources={}
    music,records=shaped_score(ep,c,0,n,c['restoredPolicy'],sources)
    if not np.isfinite(music).all():raise RuntimeError('Non-finite score bus')
    peak=float(np.max(np.abs(music)))
    if peak>=1:raise RuntimeError('Working music exceeds full scale; revise gains')
    target=root/('music-48kf32'+(AUDITION if a.audition else '')+'.wav');write_wav(target,music)
    for path,digest in sources.items():
        if sha(path)!=digest:raise RuntimeError('Source changed during render')
    receipt=dict(kind='working-score-not-listening-approval',episode=c['episode'],
                 finalApproved=False,configSha256=sha(a.config),clockSha256=c['clock']['sha256'],
                 sourceHashes=sources,cues=records,runtimeSeconds=n/SR,samples=n,peak=peak,
                 output=dict(path=str(target),sha256=sha(target)),
                 policy=c['restoredPolicy'],listeningVerdict=None)
    if a.audition:
        receipt.update(kind='working-score-audition-not-final',audition=True,admissible=False,hostRecord=host)
    (root/'music-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
