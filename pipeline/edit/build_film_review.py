"""Populate the compact film-first review from local source-bound metadata."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();data=json.loads(a.data.read_text())
    for k in ('episode','title','reviewKey','reviewStatus','files','events','shortEvents','chapters','subtitles','shortSubtitles','thumbnail','thumbnailPng','thumbnailJpeg','uploadKit'):assert k in data,k
    for k in ('720','1080','short','reels'):assert data['files'][k]['path'] and len(data['files'][k]['sha256'])==64
    template=Path(__file__).with_name('film_review_template.html')
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    page=template.read_text().replace('__FILM_REVIEW_DATA__',payload)
    assert '__FILM_REVIEW_DATA__' not in page
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(page)
    print(json.dumps(dict(output=str(a.output),sha256=hashlib.sha256(page.encode()).hexdigest(),templateSha256=hashlib.sha256(template.read_bytes()).hexdigest())))


if __name__=='__main__':main()
