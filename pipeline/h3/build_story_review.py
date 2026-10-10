"""Build a local review from audited footage and explicit narration provenance."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', '--project', dest='episode', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--narration-manifest', type=Path,
                        help='Optional narration manifest; paths resolve relative to that manifest')
    args = parser.parse_args()
    project, output = args.episode.resolve(), args.output.resolve()
    data = json.loads(args.audit.read_text())
    data['episode'] = data.get('episode') or project.name
    narration = {}
    if args.narration_manifest:
        manifest_path = args.narration_manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
        for item in manifest['items']:
            path = (manifest_path.parent / item['output']).resolve()
            if not path.is_relative_to(project):
                raise ValueError('Narration output escapes the project')
            narration[item.get('shotId', item['id'].removesuffix('-hi'))] = path

    def url(path):
        path = path.resolve()
        if not path.is_relative_to(project) or not path.is_file():
            raise ValueError('Missing or escaped review asset')
        return quote(os.path.relpath(path, output.parent), safe='/')

    for shot in data['shots']:
        for candidate in shot['candidates']:
            candidate['url'] = url(project / candidate['path'])
        voice = narration.get(shot.get('id'))
        shot['narration'] = url(voice) if voice and voice.is_file() and shot['speaker'] == 'Narrator' else None
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('story_review_template.html').read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__DATA__', payload))
    print(output)


if __name__ == '__main__':
    main()
