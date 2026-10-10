"""Export authored bilingual clauses for a whole-performance Hindi Short."""
import argparse
import json
from pathlib import Path
from export_bilingual_subtitles import (Cue, LANGUAGE_WIDTH, _wrapped_lines,
    render_srt, sha256_file, validate_cues)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('episode','spec','long_plan','voice_plan','sync','master','output'):
        ap.add_argument(name,type=Path)
    a=ap.parse_args();ep=a.episode.resolve()
    paths={k:(ep/getattr(a,k)).resolve() for k in
           ('spec','long_plan','voice_plan','sync','master')}
    for p in paths.values():p.relative_to(ep)
    bindings={k:sha256_file(p) for k,p in paths.items()}
    docs={k:json.loads(p.read_text()) for k,p in paths.items() if k!='master'}
    spec,plan,voice,sync=(docs[k] for k in ('spec','long_plan','voice_plan','sync'))
    assert spec['kind']=='authored-motion-short' and spec['language']=='hi'
    assert sync['passed'] and sync['masterSha256']==bindings['master']
    assert sync['planSha256']==bindings['voice_plan'] and sync['events']==len(spec['beats'])
    assert voice['sourceSpecSha256']==bindings['spec']
    lookup={e['id']:e for e in plan['events']};ids=list(lookup)
    selected=[b['id'] for b in spec['beats']]
    assert selected[0]==ids[0] and [ids.index(i) for i in selected]==sorted(ids.index(i) for i in selected)
    cues={'hi':[],'en':[]};coverage=[]
    for beat,v in zip(spec['beats'],voice['events']):
        event=lookup[beat['id']];start=beat['startFrame']/spec['fps']
        assert v['id']==beat['id'] and abs(v['start']-start)<1e-6
        assert beat['spokenTextHi']==event['subtitleTextHi']
        assert abs(beat['audioDuration']-event['subtitleAudioDuration'])<1e-5
        pairs=event['subtitleCuePairs'];weights=event['subtitleCueWeights'];total=sum(weights)
        elapsed=0.;first=len(cues['hi'])+1
        for pair,weight in zip(pairs,weights):
            lo=start+beat['audioDuration']*elapsed/total;elapsed+=weight
            hi=start+beat['audioDuration']*elapsed/total
            for language in ('hi','en'):
                text='\n'.join(_wrapped_lines(pair[language],LANGUAGE_WIDTH[language]))
                cues[language].append(Cue(lo,hi,text,beat['id']))
        coverage.append(dict(id=beat['id'],firstCue=first,cues=len(pairs),
                             start=start,spokenEnd=start+beat['audioDuration']))
    for language in cues:validate_cues(cues[language],language)
    assert [(c.start,c.end) for c in cues['hi']]==[(c.start,c.end) for c in cues['en']]
    out=(ep/a.output).resolve();out.relative_to(ep/'work');out.mkdir(exist_ok=False)
    prefix=ep.name.split('-')[0]+'-short'
    for language,rows in cues.items():(out/f'{prefix}-{language}.srt').write_text(render_srt(rows))
    result=dict(kind='authored-whole-performance-short-subtitles',bindings=bindings,
        timeline='Authored long-form meaning units remapped to complete original Short performances; not forced alignment',
        sceneCount=len(selected),cueCount=len(cues['hi']),coverage=coverage,
        outputs={p.name:sha256_file(p) for p in out.iterdir()},
        sourceWordsPreserved=True,narrationOrderPreserved=True,masterSyncPassed=True,
        creativeApproval=False)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
