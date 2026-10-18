#!/usr/bin/env python3
"""Inspect source provenance and downstream impact in a private project.

Usage: asset_registry.py PROJECT [--impact ASSET [--projects P ...]]
       [--diff ASSET PROMPT] [--facts STILL_ID --policy P [--storyboard S]]
       [--board --policy P --storyboard S --manifest M [--out work/....html]]
Paths, not shared basenames, identify assets. The default is a read-only JSON
inventory; --write explicitly saves it inside the project's story directory.
Series canon is referenced as ``canon:<path>``; ``--impact canon:<path>
--projects ...`` lists its dependents across episode projects. Sidecars under
``art/retired/`` are listed at their retired location, never at the path a
retake freed.
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import os
from pathlib import Path

if __package__:
    from ._common import project_root
else:
    from _common import project_root


SIDECARS = {
    '.imagegen.json': '.png', '.refpack.json': '.png', '.kie.json': '.png',
    '.vid.json': '.mp4', '.ingest.json': '.mp4', '.or.json': '.png',
    '.still.json': '.png', '.derive.json': '.png',
}
RETIRED = 'art/retired/'


def asset_key(value: str, root: Path) -> str:
    root = root.resolve()
    if value.startswith(('https://', 'gs://', 'canon:')):
        return value
    path = (root / value).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # External sources remain visible dependencies. Never open them or
        # silently bind a same-named file inside the project.
        return str(path)


def references(record: dict) -> list[str]:
    result = []
    values = record.get('imagegen_refs', record.get('refs', []))
    if isinstance(values, dict):
        values = values.get('all', [])
    if isinstance(values, list):
        result.extend(x for x in values if isinstance(x, str) and x)
    for key in ('source', 'first_frame', 'last_frame'):
        value = record.get(key)
        if isinstance(value, str) and value:
            result.append(value)
    parent = record.get('parent')
    if isinstance(parent, dict) and isinstance(parent.get('path'), str) and parent['path']:
        result.append(parent['path'])
    if record.get('kind') == 'built_in_imagegen_reference_pack':
        result.extend(row['source'] for row in record.get('sources', [])
                      if isinstance(row, dict) and isinstance(row.get('source'), str))
    return list(dict.fromkeys(result))


def load_registry(root: Path) -> dict[str, dict]:
    root = root.resolve()
    assets = {}
    for directory in ('art', 'audio', 'motion', 'video', 'work'):
        for sidecar in sorted((root / directory).rglob('*.json')):
            suffix = next((s for s in SIDECARS if sidecar.name.endswith(s)), None)
            if suffix is None:
                continue
            record = json.loads(sidecar.read_text(encoding='utf-8'))
            if not isinstance(record, dict):
                raise ValueError(f'Sidecar must contain an object: {sidecar}')
            retired = sidecar.relative_to(root).as_posix().startswith(RETIRED)
            target = None if retired else next(
                (record[k] for k in ('target_output_path', 'pack', 'file', 'output')
                 if isinstance(record.get(k), str) and record[k]), None)
            if target is None:
                stem = sidecar.name[:-len(suffix)]
                candidates = [sidecar.with_name(stem + ext)
                              for ext in ('.png', '.jpg', '.jpeg', '.mp4', '.wav', '.mp3')
                              if sidecar.with_name(stem + ext).is_file()]
                if len(candidates) > 1:
                    raise ValueError(f'Ambiguous sidecar target: {sidecar}')
                target = str(candidates[0] if candidates else sidecar.with_name(stem + SIDECARS[suffix]))
            key = asset_key(target, root)
            row = {
                'sidecar': sidecar.relative_to(root).as_posix(),
                'model': record.get('model'),
                'prompt': record.get('prompt_file'),
                'refs': [asset_key(x, root) for x in references(record)],
                'sha256': record.get('target_sha256', record.get('outputSha256', record.get('sha256'))),
                'costUsd': record.get('cost_usd'),
            }
            if retired:
                row['retired'] = True
            if key in assets and assets[key] != row:
                raise ValueError(f'Multiple sidecars claim the same asset: {key}')
            assets[key] = row
    return assets


def select_asset(value: str, assets: dict[str, dict], root: Path) -> str:
    exact = asset_key(value, root)
    if exact in assets or exact.startswith('canon:'):
        return exact
    candidates = [key for key in assets if value in (Path(key).name, Path(key).stem)]
    if len(candidates) != 1:
        raise ValueError('Use an exact asset path; its short name is missing or ambiguous')
    return candidates[0]


def downstream(assets: dict[str, dict], source: str) -> list[str]:
    reverse = {}
    for key, row in assets.items():
        for ref in row['refs']:
            reverse.setdefault(ref, set()).add(key)
    seen, pending = {source}, [source]
    result = []
    while pending:
        parent = pending.pop()
        for child in sorted(reverse.get(parent, ())):
            if child not in seen:
                seen.add(child)
                result.append(child)
                pending.append(child)
    return sorted(result)


def _explicit_file(root: Path, value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f'{label} must be given explicitly; nothing is discovered')
    path = (value if value.is_absolute() else root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'{label} must be an existing file inside the project: {value}')
    return path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _fact_resolver():
    if __package__:
        from .register_builtin_imagegen import resolve_continuity_facts
    else:
        from register_builtin_imagegen import resolve_continuity_facts
    return resolve_continuity_facts


def active_facts(policy: dict, still_id: str, order: list[str] | None) -> dict:
    """Facts a registration of this still must acknowledge, for pasting into prompts."""
    required, skipped = _fact_resolver()(policy, still_id, order)
    return {'still_id': still_id,
            'facts': [{'id': str(f['id']), 'value': str(f['value'])} for f in required],
            'skipped': skipped}


def board_html(root: Path, policy: dict, storyboard: dict, manifest: dict, out: Path) -> str:
    """A read-only identity board: each authority's sheets, then every still that
    claims or references it, in story order, captioned with its active facts."""
    order = [str(row.get('still_id')) for row in storyboard.get('stills', [])]
    speakers = {str(row.get('still_id')): row.get('speaker') for row in storyboard.get('stills', [])}
    records = {str(row.get('still_id')): row for row in manifest.get('stills', [])}
    ordered = [sid for sid in order if sid in records]
    ordered += [sid for sid in records if sid not in order]

    def image(value: str | None) -> str:
        if not value or value.startswith(('canon:', 'https://', 'gs://')):
            return f'<div class="none">{html.escape(value or "no image")}</div>'
        target = (root / value).resolve()
        if not target.is_file():
            return f'<div class="none">missing: {html.escape(value)}</div>'
        rel = os.path.relpath(target, out.parent)
        return f'<img loading="lazy" src="{html.escape(Path(rel).as_posix())}" alt="">'

    def refs(record: dict) -> set[str]:
        values = record.get('imagegen_refs', record.get('refs', {}))
        if isinstance(values, dict):
            values = values.get('all', [])
        return {str(x) for x in values or []}

    sections = []
    authorities = policy.get('grounding_guard', {}).get('identity_authorities', [])
    for authority in authorities:
        name = str(authority.get('name', ''))
        sources = [str(x) for x in authority.get('sources', [])]
        names = {name.casefold(), *(str(a).casefold() for a in authority.get('aliases', []))}
        cards = [f'<figure>{image(src)}<figcaption>sheet: {html.escape(src)}</figcaption></figure>'
                 for src in sources]
        for sid in ordered:
            record = records[sid]
            claims = record.get('grounding_claims', {}).get('exact_identity_claims', [])
            claimed = any(str(c.get('name', '')).casefold() in names for c in claims)
            if not claimed and not refs(record) & set(sources):
                continue
            try:
                facts = active_facts(policy, sid, order)['facts']
            except ValueError as exc:
                facts = [{'id': 'error', 'value': str(exc)}]
            caption = [f'<b>{html.escape(sid)}</b> · {html.escape(str(record.get("status")))}']
            if speakers.get(sid):
                caption.append(f'speaker: {html.escape(str(speakers[sid]))}')
            caption.append('claims identity' if claimed else 'references a sheet')
            caption += [f'{html.escape(f["id"])}: {html.escape(f["value"])}' for f in facts]
            cards.append(f'<figure>{image(record.get("target_output_path") if record.get("status") == "generated" else None)}'
                         f'<figcaption>{"<br>".join(caption)}</figcaption></figure>')
        sections.append(f'<section><h2>{html.escape(name)}</h2><div class="grid">{"".join(cards)}</div></section>')
    return ('<!doctype html><html><head><meta charset="utf-8"><title>Still board</title>'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<style>body{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}'
            'figure{margin:0}img{width:100%;height:auto;background:#eee}'
            '.none{padding:40px 8px;background:#eee;text-align:center;font-size:12px}'
            'figcaption{font-size:12px;line-height:1.4;margin-top:4px}</style></head><body>'
            '<h1>Still board</h1><p>Read-only view of registered stills by identity authority, '
            'in storyboard order. Captions list the continuity facts active at each still. '
            'Images are referenced by path; nothing here records a verdict.</p>'
            + ''.join(sections) + '</body></html>\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project', type=Path, nargs='?')
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument('--impact', metavar='ASSET')
    operations.add_argument('--diff', nargs=2, metavar=('ASSET', 'PROMPT'))
    operations.add_argument('--write', action='store_true')
    operations.add_argument('--facts', metavar='STILL_ID',
                            help='Print the continuity facts active at a still (needs --policy)')
    operations.add_argument('--board', action='store_true',
                            help='Write a read-only HTML identity board under work/')
    parser.add_argument('--projects', nargs='+', type=Path,
                        help='With --impact canon:<path>: episode projects to search')
    parser.add_argument('--policy', type=Path, help='Explicit project policy')
    parser.add_argument('--storyboard', type=Path, help='Explicit storyboard (story order)')
    parser.add_argument('--manifest', type=Path, help='Explicit generation_manifest.json')
    parser.add_argument('--out', type=Path, default=Path('work/still-board.html'),
                        help='Board path inside the project work/ directory')
    args = parser.parse_args()
    if args.projects and not (args.impact or '').startswith('canon:'):
        parser.error('--projects applies to --impact canon:<path>')
    if args.impact and args.projects:
        result = {'source': args.impact, 'projects': {},
                  'status': 'dependency inventory; each change requires explicit review'}
        for project in args.projects:
            root = project_root(project)
            result['projects'][str(root)] = downstream(load_registry(root), args.impact)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    root = project_root(args.project)
    if args.facts or args.board:
        policy = _load(_explicit_file(root, args.policy, '--policy'))
        storyboard = (_load(_explicit_file(root, args.storyboard, '--storyboard'))
                      if args.storyboard or args.board else None)
        order = ([str(r.get('still_id')) for r in storyboard.get('stills', [])]
                 if storyboard is not None else None)
        if args.facts:
            print(json.dumps(active_facts(policy, args.facts, order), ensure_ascii=False, indent=2))
            return
        manifest = _load(_explicit_file(root, args.manifest, '--manifest'))
        out = (root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
        if not out.is_relative_to(root / 'work') or out.suffix != '.html':
            raise ValueError('--out must be an .html file inside the project work/ directory')
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + '.tmp')
        temporary.write_text(board_html(root, policy, storyboard, manifest, out), encoding='utf-8')
        temporary.replace(out)
        print(json.dumps({'board': out.relative_to(root).as_posix()}))
        return
    assets = load_registry(root)
    if args.diff:
        key = select_asset(args.diff[0], assets, root)
        old = assets[key]['prompt']
        if not old or not (root / old).is_file():
            raise ValueError('The selected asset has no retained prompt file')
        print(''.join(difflib.unified_diff(
            (root / old).read_text().splitlines(True),
            Path(args.diff[1]).read_text().splitlines(True),
            fromfile='recorded', tofile='proposed')), end='')
        return
    result = {'assets': assets, 'count': len(assets)}
    if args.impact:
        source = select_asset(args.impact, assets, root)
        result = {'source': source, 'downstream': downstream(assets, source),
                  'status': 'dependency inventory; each change requires explicit review'}
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.write:
        output = root / 'story/assets.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(output)
    print(text, end='')


if __name__ == '__main__':
    main()
