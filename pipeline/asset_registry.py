#!/usr/bin/env python3
"""Inspect source provenance and downstream impact in a private project.

Usage: asset_registry.py PROJECT [--impact ASSET] [--diff ASSET PROMPT]
Paths, not shared basenames, identify assets. The default is a read-only JSON
inventory; --write explicitly saves it inside the project's story directory.
On a case-insensitive disk, keys that differ only in case name one file: the
inventory reports them as caseVariants and never merges them. References that
resolve to no file are listed as unresolvedReferences. --relocation-map maps
absolute paths recorded before a project moved.
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

if __package__:
    from ._common import load_relocation_map, project_root
else:
    from _common import load_relocation_map, project_root


SIDECARS = {
    '.imagegen.json': '.png', '.refpack.json': '.png', '.kie.json': '.png',
    '.vid.json': '.mp4', '.ingest.json': '.mp4', '.or.json': '.png',
}


def case_insensitive(root: Path) -> bool:
    """Whether this project's disk folds case (true on default macOS volumes)."""
    root = root.resolve()
    for part in [root, *root.parents]:
        if part.name.lower() != part.name.upper():
            twin = part.with_name(part.name.swapcase())
            return twin.exists() and twin.samefile(part)
    return False


def fold(key: str, folded: bool) -> str:
    return key.casefold() if folded else key


def asset_key(value: str, root: Path, relocation: dict | None = None) -> str:
    root = root.resolve()
    if value.startswith(('https://', 'gs://')):
        return value
    if relocation and Path(value).is_absolute():
        mapped = relocation.get('paths', {}).get(value)
        original = relocation.get('originalEpisodeRoot')
        if mapped is not None:
            value = mapped
        elif original and Path(value).is_relative_to(original):
            value = Path(value).relative_to(original).as_posix()
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
    if record.get('kind') == 'built_in_imagegen_reference_pack':
        result.extend(row['source'] for row in record.get('sources', [])
                      if isinstance(row, dict) and isinstance(row.get('source'), str))
    return list(dict.fromkeys(result))


def load_registry(root: Path, relocation: dict | None = None) -> dict[str, dict]:
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
            target = next((record[k] for k in ('target_output_path', 'pack', 'file', 'output')
                           if isinstance(record.get(k), str) and record[k]), None)
            if target is None:
                stem = sidecar.name[:-len(suffix)]
                candidates = [sidecar.with_name(stem + ext)
                              for ext in ('.png', '.jpg', '.jpeg', '.mp4', '.wav', '.mp3')
                              if sidecar.with_name(stem + ext).is_file()]
                if len(candidates) > 1:
                    raise ValueError(f'Ambiguous sidecar target: {sidecar}')
                target = str(candidates[0] if candidates else sidecar.with_name(stem + SIDECARS[suffix]))
            key = asset_key(target, root, relocation)
            row = {
                'sidecar': sidecar.relative_to(root).as_posix(),
                'model': record.get('model'),
                'prompt': record.get('prompt_file'),
                'refs': [asset_key(x, root, relocation) for x in references(record)],
                'sha256': record.get('target_sha256', record.get('outputSha256', record.get('sha256'))),
                'costUsd': record.get('cost_usd'),
            }
            if key in assets and assets[key] != row:
                raise ValueError(f'Multiple sidecars claim the same asset: {key}')
            assets[key] = row
    return assets


def integrity_report(assets: dict[str, dict], root: Path) -> dict[str, list]:
    """Case variants (reported, never merged) and references that resolve nowhere."""
    folded = case_insensitive(root)
    spellings: dict[str, set[str]] = {}
    for key, row in assets.items():
        for name in [key, *row['refs']]:
            if not name.startswith(('https://', 'gs://')):
                spellings.setdefault(fold(name, folded), set()).add(name)
    variants = sorted(sorted(names) for names in spellings.values() if len(names) > 1) if folded else []
    known = {fold(key, folded) for key in assets}
    unresolved = []
    for key, row in sorted(assets.items()):
        for ref in row['refs']:
            if ref.startswith(('https://', 'gs://')) or fold(ref, folded) in known:
                continue
            path = Path(ref) if Path(ref).is_absolute() else root / ref
            if not path.exists():
                unresolved.append({'asset': key, 'reference': ref})
    return {'caseVariants': variants, 'unresolvedReferences': unresolved}


def select_asset(value: str, assets: dict[str, dict], root: Path) -> str:
    exact = asset_key(value, root)
    if exact in assets:
        return exact
    if case_insensitive(root):
        same = [key for key in assets if key.casefold() == exact.casefold()]
        if len(same) == 1:
            return same[0]
    candidates = [key for key in assets if value in (Path(key).name, Path(key).stem)]
    if len(candidates) != 1:
        raise ValueError('Use an exact asset path; its short name is missing or ambiguous')
    return candidates[0]


def downstream(assets: dict[str, dict], source: str, folded: bool = False) -> list[str]:
    reverse = {}
    for key, row in assets.items():
        for ref in row['refs']:
            reverse.setdefault(fold(ref, folded), set()).add(key)
    seen, pending = {source}, [source]
    result = []
    while pending:
        parent = pending.pop()
        for child in sorted(reverse.get(fold(parent, folded), ())):
            if child not in seen:
                seen.add(child)
                result.append(child)
                pending.append(child)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path, nargs='?')
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument('--impact', metavar='ASSET')
    operations.add_argument('--diff', nargs=2, metavar=('ASSET', 'PROMPT'))
    operations.add_argument('--write', action='store_true')
    parser.add_argument('--relocation-map', type=Path,
                        help='explicit map for a moved project: originalEpisodeRoot and/or paths')
    args = parser.parse_args()
    root = project_root(args.project)
    assets = load_registry(root, load_relocation_map(args.relocation_map))
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
    result = {'assets': assets, 'count': len(assets), **integrity_report(assets, root)}
    if args.impact:
        source = select_asset(args.impact, assets, root)
        result = {'source': source,
                  'downstream': downstream(assets, source, case_insensitive(root)),
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
