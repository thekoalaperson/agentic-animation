"""Version subtitle-only text/clauses while proving picture and speech unchanged."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from export_bilingual_subtitles import build_synchronized_cues


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode',type=Path);ap.add_argument('recipe',type=Path)
    a=ap.parse_args();ep=a.episode.resolve();recipe_path=(ep/a.recipe).resolve()
    c=json.loads(recipe_path.read_text());assert c['kind']=='source-bound-subtitle-derivation'
    def bound(b):
        p=(ep/b['path']).resolve();p.relative_to(ep);assert sha(p)==b['sha256'];return p
    source=bound(c['sourcePlan']);manuscript=bound(c['manuscript'])
    m=json.loads(manuscript.read_text());assert m['kind']=='source-bound-authored-subtitle-pairs' and m['authority']
    timing={}
    for b in m['sourceBindings']:
        d=json.loads(bound(b).read_text());timing[d['language']]=d
    assert set(timing)=={'hi','en'}
    original=json.loads(source.read_text());out=copy.deepcopy(original)
    assert out['renderable'] is True and out['finalApproved'] is False
    # Wordless picture events carry no words, so they have no manuscript item
    # and no timing shot; every spoken event keeps its own.
    spoken=[e for e in out['events'] if e.get('kind')!='picture']
    assert len(m['items'])==len(spoken)>0
    changed=[]
    for i,(event,item) in enumerate(zip(spoken,m['items'])):
        assert event['id']==item['id'] and item['hi'] and len(item['hi'])==len(item['en'])
        hi=' '.join(item['hi']);en=' '.join(item['en'])
        assert hi==event['subtitleTextHi'],'Never rewrite performed Hindi'
        if en!=event['subtitleTextEn']:
            assert item.get('editReason');changed.append(item['id'])
        for lang in ('hi','en'):
            old=timing[lang]['shots'][i]
            assert old['id']==event['id'] and old['text']==event['subtitleText'+lang.title()]
            assert abs(old['start']-event['start'])<1e-5 and abs(old['dur']-event['duration'])<1e-5
        duration=event['subtitleAudioDuration'];n=len(item['hi'])
        assert duration>=n,'Do not flash a caption below1second'
        raw=[len(p.replace(' ','')) for p in item['hi']]
        weights=[1+(duration-n)*v/sum(raw) for v in raw]
        if 'cueWeights' in item:
            weights=item['cueWeights']
            assert len(weights)==n and all(type(w) in (int,float) and w>0 for w in weights)
            assert abs(sum(weights)-duration)<1e-5, 'Explicit merge weights must preserve utterance duration'
        event['subtitleTextEn']=en
        event['subtitleCuePairs']=[dict(hi=h,en=e) for h,e in zip(item['hi'],item['en'])]
        event['subtitleCueWeights']=weights
        for lang in ('hi','en'):
            timing[lang]['shots'][i].update(text=hi if lang=='hi' else en,
                cueParts=item[lang],cueWeights=weights)
    def without_subtitles(plan):
        p=copy.deepcopy(plan)
        for e in p['events']:
            for k in list(e):
                if k in ('subtitleTextEn','subtitleCuePairs','subtitleCueWeights'):del e[k]
        return p
    assert without_subtitles(out)==without_subtitles(original),'Picture/performance changed'
    out['parentTimelineFingerprint']=original['timelineFingerprint']
    out['subtitleDerivation']={**c,'recipeSha256':sha(recipe_path),'englishTextEdits':changed,
        'pictureAndSpeechUnchanged':True,'clockUnchanged':True,
        'timingPolicy':'Authored paired clauses, proportional complete-utterance allocation with1s minimum; not forced alignment'}
    fingerprint=hashlib.sha256(json.dumps(dict(parent=original['timelineFingerprint'],
        manuscriptSha256=sha(manuscript)),sort_keys=True).encode()).hexdigest()
    out['timelineFingerprint']=fingerprint
    for d in timing.values():d['timelineFingerprint']=fingerprint
    hi_cues,en_cues,coverage=build_synchronized_cues(timing['hi']['shots'],timing['en']['shots'])
    dest=(ep/c['outputDirectory']).resolve();dest.relative_to(ep/'work');dest.mkdir(exist_ok=False)
    def write(name,value):(dest/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    write('edit-plan.json',out)
    for lang,d in timing.items():write(f'subtitle-{lang}-timing.json',d)
    receipt=dict(kind='subtitle-only-derivation-audit',sourcePlanSha256=sha(source),
        manuscriptSha256=sha(manuscript),recipeSha256=sha(recipe_path),toolSha256=sha(Path(__file__)),
        outputs={p.name:sha(p) for p in dest.iterdir() if p.suffix=='.json'},
        sceneCount=len(spoken),cueCount=len(hi_cues),englishTextEdits=changed,
        pictureAndSpeechUnchanged=True,clockUnchanged=True,timelineFingerprint=fingerprint,
        minimumCueSeconds=min(c.end-c.start for c in hi_cues),coverage=coverage)
    write('derivation-audit.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='coverage'}))


if __name__=='__main__':main()
