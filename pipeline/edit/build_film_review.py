"""Populate the compact film-first review from local source-bound metadata.

A working cut alone is enough: ``files`` needs a ``720`` or ``1080`` entry,
each with a ``path`` and its ``sha256``. Short and Reels files (which then
need ``shortEvents`` and ``shortSubtitles``), ``subtitles``, ``chapters``,
thumbnails and the upload kit are optional panels. Optional
``reviewSummary`` (the timeline readiness report's object, or lines),
``checked`` (what was checked and by which method) and ``notChecked`` (what
could not be) are shown beside the film, so a person sees what the review
covered. The page is self-contained apart from relative media; nothing here
uploads or publishes.
"""
import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = ('episode', 'title', 'reviewKey', 'reviewStatus', 'files', 'events')
ASSETS = ('720', '1080', 'short', 'reels')


def validate(data):
    for key in REQUIRED:
        if key not in data:
            raise ValueError(f'film review data needs {key}')
    files = data['files']
    if not isinstance(files, dict) or not ({'720', '1080'} & set(files)):
        raise ValueError('film review needs a working cut: a 720 or 1080 file')
    unknown = sorted(set(files) - set(ASSETS))
    if unknown:
        raise ValueError(f'unknown review files {unknown}; use {", ".join(ASSETS)}')
    for key, row in files.items():
        if not isinstance(row, dict) or not row.get('path') or len(str(row.get('sha256', ''))) != 64:
            raise ValueError(f'review file {key} needs a path and its sha256')
    if {'short', 'reels'} & set(files):
        for key in ('shortEvents', 'shortSubtitles'):
            if key not in data:
                raise ValueError(f'a Short or Reels file needs {key}')
    for key in ('checked', 'notChecked'):
        if key in data and not isinstance(data[key], list):
            raise ValueError(f'{key} must be a list')
    if 'reviewSummary' in data and not isinstance(data['reviewSummary'], (list, dict)):
        raise ValueError('reviewSummary must be the readiness object or a list of lines')


def render(data, template_text):
    validate(data)
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace(
        '\u2028', '\\u2028').replace('\u2029', '\\u2029')
    page = template_text.replace('__FILM_REVIEW_DATA__', payload)
    if '__FILM_REVIEW_DATA__' in page:
        raise ValueError('film review template placeholder survived')
    return page


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args(argv)
    data = json.loads(a.data.read_text())
    template = Path(__file__).with_name('film_review_template.html')
    try:
        page = render(data, template.read_text())
    except ValueError as exc:
        raise SystemExit(f'refused: {exc}') from exc
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(page)
    print(json.dumps(dict(output=str(a.output), sha256=hashlib.sha256(page.encode()).hexdigest(),
                          templateSha256=hashlib.sha256(template.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
