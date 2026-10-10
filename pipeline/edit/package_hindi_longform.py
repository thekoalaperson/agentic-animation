#!/usr/bin/env python3
"""Finalize a seven-file Hindi long-form package from GPU-verified artifacts.

No rendering, probing or decoding occurs here. Source hashes and previously
completed GPU audits are bound to the finalization request. The separate full
episode packager continues to gate Shorts; this lane never admits a draft Short.
"""
import argparse
import html
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone

try:
    from .package_hindi_delivery import (analyze_config, episode_path, sha256_file,
        parse_srt, _copy_block, description_with_chapters, _exclusive_promotion_lock,
        _rename_noreplace, KitAudit, upload_settings_html)
except ImportError:
    from package_hindi_delivery import (analyze_config, episode_path, sha256_file,
        parse_srt, _copy_block, description_with_chapters, _exclusive_promotion_lock,
        _rename_noreplace, KitAudit, upload_settings_html)


def read(path):
    return json.loads(path.read_text())


def checked(episode, path, digest):
    result = episode_path(episode, path, 'long-form input', must_exist=True)
    if sha256_file(result) != digest:
        raise ValueError(f'Input hash mismatch: {path}')
    return result


def shift_srt(text, millis):
    def replacement(match):
        h, m, s, ms = map(int, match.groups())
        total = ((h * 60 + m) * 60 + s) * 1000 + ms + millis
        if total < 0:
            raise ValueError('Negative translated subtitle time')
        seconds, ms = divmod(total, 1000)
        minutes, s = divmod(seconds, 60)
        h, m = divmod(minutes, 60)
        return f'{h:02}:{m:02}:{s:02},{ms:03}'
    return re.sub(r'(\d{2,}):(\d{2}):(\d{2}),(\d{3})', replacement, text)


def thumbnail_title_lines(cfg, recipe):
    if cfg['thumbnail']['method'] == 'gpu-typography':
        variant = next(row for row in recipe['variants'] if row['id'] == cfg['thumbnail']['choice'])
        lines = [row['text'] for row in variant['lines']]
    else:
        lines = recipe['titleLines']
    if not lines or not all(isinstance(line, str) and line.strip() for line in lines):
        raise ValueError('Thumbnail receipt must retain its exact approved display text')
    return lines


def build_kit(cfg, files, ending_link, ending_seconds):
    meta = cfg['metadata']
    esc = html.escape
    links = ''.join(f'<a href="{esc(path, quote=True)}">{label}</a>' for label, path in [
        ('Hindi 1080p master', files['long-hi']),
        ('Upload thumbnail · JPEG', files['thumbnail-hi']),
        ('Lossless thumbnail · PNG', files['thumbnail-master']),
        ('Hindi subtitles', files['subtitles-hi']), ('English subtitles', files['subtitles-en']),
        (f'End-screen preview · {ending_seconds:g}s ending included', ending_link),
        ('File manifest', 'FINAL_MANIFEST.json')])
    return f'''<!doctype html><html lang="en" translate="no"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(meta['title'])} · Upload kit</title>
<style>
:root{{color-scheme:dark;--bg:#101626;--ink:#f5f0e8;--gold:#dab274;--muted:#b7bfc9;--rule:#414257}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1000px;margin:auto;padding:28px 24px 60px}}h1{{font:500 clamp(28px,4vw,44px)/1.2 Georgia,serif;margin:12px 0 22px}}
h2{{font-size:18px;margin:0}}p{{max-width:78ch}}.note{{color:var(--muted)}}img{{display:block;width:100%;height:auto;aspect-ratio:16/9}}
nav{{display:flex;flex-wrap:wrap;gap:10px 22px;margin:24px 0}}a{{color:var(--gold)}}section{{border-top:1px solid var(--rule);padding:22px 0}}
.copy-block>div{{display:flex;align-items:center;justify-content:space-between;gap:14px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}}
button{{font:inherit;color:var(--ink);background:transparent;border:1px solid var(--gold);padding:5px 16px;cursor:pointer}}
button:focus-visible,a:focus-visible{{outline:3px solid #a09ad0;outline-offset:4px}}li{{margin:6px 0}}
</style></head><body><main>
<p class="note">Hindi long-form delivery · {esc(cfg['channel']['handle'])}</p><h1>{esc(meta['title'])}</h1>
<img src="{esc(files['thumbnail-master'], quote=True)}" alt="Approved thumbnail for {esc(meta['title'], quote=True)}">
<nav aria-label="Delivery files">{links}</nav>
<p class="note">The master includes the complete film and its {ending_seconds:g}-second closing screen. Use the JPEG for YouTube upload; the lossless PNG is retained. The Short is separate and is not included.</p>
{_copy_block('Selected title', 'hi-title', meta['title'])}
{_copy_block('Hindi description', 'hi-description', description_with_chapters(meta))}
{_copy_block('Tags', 'common-tags', ', '.join(meta['tags']))}
{upload_settings_html(cfg)}
<section><h2>Upload handoff</h2>
<p>Add both authored SRT tracks as selectable Hindi and English subtitles. The MP4 already contains the closing screen; clickable YouTube end-screen elements, if wanted, are added in Studio after upload—not embedded in the MP4.</p>
<p class="note">Publication not authorized by this package. No visibility or schedule applied. This kit does not grant permission to upload; follow the project's publication workflow when separately authorized.</p></section>
</main><script>
document.querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',async()=>{{
const value=document.getElementById(b.dataset.copy).textContent;try{{
if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(value);
else{{const a=document.createElement('textarea');a.value=value;document.body.append(a);a.select();
if(!document.execCommand('copy'))throw Error('copy');a.remove();}}
b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1200);
}}catch(e){{b.textContent='Select and copy text below';}}}}));
</script></body></html>'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode', type=Path)
    ap.add_argument('config', type=Path)
    ap.add_argument('proof', type=Path)
    ap.add_argument('--promote-to-deliverables', action='store_true')
    args = ap.parse_args()
    ep = args.episode.resolve()
    cfg, gates = analyze_config(ep, args.config.resolve())
    # Full-episode approvals are not silently marked passed. This separate lane
    # uses the exact source-bound long-form finalization request below.
    deferred = {'approval:longForm', 'approval:short', 'approval:subtitles',
                'approval:thumbnail', 'approval:title', 'metadata:chapters', 'short-human-review'}
    short_ids = {'short-hi', 'short-hi-social', 'short-subtitles-hi', 'short-subtitles-en'}
    blocking = [g for g in gates if g not in deferred and g.split(':')[-1] not in short_ids]
    if blocking:
        raise ValueError(f'Long-form inputs incomplete: {blocking}')
    proof = read(args.proof)
    authorization = checked(ep, proof['authorization']['path'], proof['authorization']['sha256'])
    consent = read(authorization)
    if (consent.get('finalizationRequested') is not True or consent.get('publishingAuthorized') is not False
            or consent.get('shortIncluded') is not False or consent['title'] != cfg['metadata']['title']
            or consent['longFormSha256'] != cfg['assets']['long-hi']['sha256']
            or consent['thumbnailSourceSha256'] != cfg['thumbnail']['sha256']
            or consent['thumbnailTextLines'] != thumbnail_title_lines(cfg, read(ep / cfg['thumbnail']['titleRecipe']))):
        raise ValueError('Finalization request does not bind this long-form/title/thumbnail')
    evidence = {key: read(checked(ep, row['path'], row['sha256'])) for key, row in proof['audits'].items()}
    master_sha = cfg['assets']['long-hi']['sha256']
    append, clock, av, thumb = (evidence[k] for k in ['append', 'clock', 'av', 'thumbnail'])
    if (append['sha256'] != master_sha or append['decode'] != 'passed'
            or append['bodyPicturePacketsUnchanged'] is not True or append['bodyAudioPacketsUnchanged'] is not True
            or append['frames'] != cfg['assets']['long-hi']['expected']['frames']
            or append['resolution'] != [1920, 1080] or append['titleSeconds'] != proof['endScreenSeconds']
            or clock['sha256'] != master_sha or clock['passed'] is not True
            or av['finalSha256'] != master_sha or av['passed'] is not True
            or av['bodySha256'] != append['coreSha256'] or av['maxRelativeAvClockChangeSeconds'] > .002
            or thumb['sourceSha256'] != cfg['thumbnail']['sha256'] or thumb['sameSourceComposition'] is not True):
        raise ValueError('GPU audit/source contract mismatch')
    for key in ['png', 'jpeg']:
        if thumb[key]['fullDecode'] != 'passed' or [thumb[key]['width'], thumb[key]['height']] != [1280, 720]:
            raise ValueError('Thumbnail format audit failed')
    if thumb['jpeg']['bytes'] > 2_000_000 or thumb['jpeg']['derivedFromPngSha256'] != thumb['png']['sha256']:
        raise ValueError('Thumbnail upload size/composition mismatch')
    files = {k: cfg['assets'][k]['destination'] for k in ['long-hi', 'subtitles-hi', 'subtitles-en']}
    files.update({'thumbnail-master': 'thumbnails/thumb_hi_titled.png',
                  'thumbnail-hi': 'thumbnails/thumb_hi_upload_1280.jpg', 'upload-kit': 'upload_kit.html'})
    destination = ep / ('deliverables' if args.promote_to_deliverables else proof['stagingDir'])
    if not destination.resolve().is_relative_to(ep):
        raise ValueError('Destination outside episode')
    ending = episode_path(ep, proof['endingPreview'], 'ending preview', must_exist=True)
    stage_parent = ep / 'work/final-delivery-v1'
    stage_parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_promotion_lock(ep), tempfile.TemporaryDirectory(prefix='.longform-', dir=stage_parent) as td:
        stage = Path(td) / 'package'
        stage.mkdir()
        for key in ['long-hi', 'subtitles-hi', 'subtitles-en']:
            row = cfg['assets'][key]
            src = checked(ep, row['source'], row['sha256'])
            out = stage / files[key]
            if key == 'long-hi':
                shutil.copyfile(src, out)
                if sha256_file(out) != master_sha:
                    raise ValueError('Master changed during copy')
            else:
                out.write_text(shift_srt(src.read_text(), proof['subtitleOffsetMs']), encoding='utf-8')
        for key, kind in [('thumbnail-master', 'png'), ('thumbnail-hi', 'jpeg')]:
            src = checked(ep, proof['thumbnailFiles'][kind], thumb[kind]['sha256'])
            (stage / files[key]).parent.mkdir(exist_ok=True)
            shutil.copyfile(src, stage / files[key])
        subtitles = {lang: parse_srt(stage / files['subtitles-' + lang], language=lang,
                     media_duration=append['duration']) for lang in ['hi', 'en']}
        if subtitles['hi']['timings'] != subtitles['en']['timings']:
            raise ValueError('Hindi/English subtitle timing mismatch')
        ending_link = os.path.relpath(ending, destination)
        page = build_kit(cfg, files, ending_link, append['titleSeconds'])
        (stage / 'upload_kit.html').write_text(page, encoding='utf-8')
        audit = KitAudit(); audit.feed(page)
        for key in ['hi-title', 'hi-description', 'common-tags']:
            if audit.ids.get(key) != 1 or audit.copy_targets.count(key) != 1:
                raise ValueError('Copy control wiring failed')
        if audit.copy_text['hi-title'] != consent['title']:
            raise ValueError('Stale kit title')
        manifest = {'schemaVersion': 3, 'layout': 'hindi-longform-only-v1', 'episode': ep.name,
            'status': 'verified long-form delivery; Short separate; publication not authorized',
            'language': 'hi', 'channelHandle': cfg['channel']['handle'], 'channelId': cfg['channel']['id'],
            'publishingAuthorized': False, 'publishAt': None, 'title': consent['title'], 'titleApproved': True,
            'thumbnailChoice': cfg['thumbnail']['choice'], 'thumbnailApproved': True,
            'approvalScope': 'Source-bound finalization request for this master, title and thumbnail',
            'short': {'included': False, 'status': 'unfinished; not approved or packaged'},
            'endScreen': {'seconds': append['titleSeconds'], 'embeddedInMaster': True,
                          'interactiveYouTubeElementsConfigured': False},
            'metadata': cfg['metadata'], 'assets': [{'id': k, 'canonical': 'deliverables/' + p,
                'sha256': sha256_file(stage / p), 'bytes': (stage / p).stat().st_size} for k, p in files.items()],
            'validation': {'gpuMediaAudits': proof['audits'], 'subtitleOffsetMs': proof['subtitleOffsetMs'],
                'subtitleCueCount': subtitles['hi']['cueCount'], 'exactFileCount': 7,
                'noLocalMediaProcessing': True, 'sourcesDeleted': False},
            'provenance': {'authorization': proof['authorization'],
                'config': str(args.config.resolve().relative_to(ep)), 'configSha256': sha256_file(args.config),
                'packagingProof': str(args.proof.resolve().relative_to(ep)), 'packagingProofSha256': sha256_file(args.proof)},
            'createdAtUTC': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
        (stage / 'FINAL_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        expected = set(files.values()) | {'FINAL_MANIFEST.json'}
        if {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()} != expected:
            raise ValueError('Unexpected final file set')
        for link in audit.links:
            if link == ending_link:
                continue
            if not (stage / link).is_file():
                raise ValueError(f'Broken kit asset link: {link}')
        _rename_noreplace(stage, destination)
    print(json.dumps({'output': str(destination), 'files': sorted(expected),
                      'publishingAuthorized': False}, indent=2))


if __name__ == '__main__':
    main()
