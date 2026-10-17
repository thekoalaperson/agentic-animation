#!/usr/bin/env python3
"""Source-bound Kie video generation and durable request recovery.

One invocation makes at most one generation POST. Before that POST it writes
<stem>.pending.json, then records the returned taskId; a later invocation with
the same inputs polls that task instead of posting again. The landed .mp4 and
its .vid.json sidecar bind the request, the inputs and the output bytes.

The finite dialogue runner calls this tool with the four positionals plus
--res, --dur, --ar, --job-id, --manifest-sha256, --source-sha256 and
--execution-request-sha256. Grok routes add each --ref after the first frame in
image_urls (the first frame is @image1, references @image2 onward) and send
aspect_ratio 16:9 on a multi-reference call unless --ar says otherwise. Grok
clips carry embedded audio.

--print-request shows the provider body with placeholder URLs and never
uploads, reads a credential or wallet, or touches a recovery record.
--resume-only polls an already-recorded task and refuses to create one."""
import argparse, hashlib, json, os, pathlib, re, sys, time
import urllib.parse
import urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import (kie, upload_ref, env_key, canonical_sha256, sha256_file as file_sha256,
                     usd_per_credit)

MODELS = {'grok': 'grok-imagine/image-to-video',
          'grok15': 'grok-imagine-video-1-5-preview',
          'omniflash11': 'google/gemini-omni-flash-1-1',
          'geminiomni': 'gemini-omni-video',
          'kling3omni': 'kling-3.0-omni/image-to-video',
          'kling3omniref': 'kling-3.0-omni/reference-to-video',
          'pixverse6': 'pixverse-v6/image-to-video',
          'pixverse6transition': 'pixverse-v6/transition',
          'pixverse6ref': 'pixverse-v6/reference-to-video',
          'pixverse6extend': 'pixverse-v6/extend',
          'seedance': 'bytedance/seedance-2-fast',
          'seedance2': 'bytedance/seedance-2',
          'seedance2mini': 'bytedance/seedance-2-mini',
          'seedance25': 'bytedance/seedance-2-5',
          'veo3fast': 'veo3_fast'}

SHA256 = re.compile(r'[0-9a-f]{64}\Z')
PENDING_VERSION = 2


def download_public_media(url):
    """Fetch a signed/public result URL without forwarding a provider secret."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Provider result must be an HTTPS media URL without userinfo')
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def fsync_parent(path):
    descriptor = os.open(str(pathlib.Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value, *, replace):
    """Durably publish JSON, optionally refusing an existing destination."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
           + '\n').encode('utf-8')
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with tmp.open('xb') as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
        fsync_parent(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_new_bytes(path, raw):
    """Publish immutable media bytes without a check/overwrite race."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.new')
    try:
        with tmp.open('xb') as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
        os.link(tmp, path)
        fsync_parent(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path, label):
    try:
        value = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f'{label} is unreadable: {path}: {exc}')
    if not isinstance(value, dict):
        sys.exit(f'{label} must be a JSON object: {path}')
    return value


def checked_optional_sha(value, label):
    if value is None:
        return None
    if not SHA256.fullmatch(value):
        sys.exit(f'{label} must be a lowercase SHA-256 digest')
    return value


def local_request_contract(*, model, prompt_sha256, source_sha256,
                           manifest_sha256, execution_request_sha256, job_id,
                           res, dur, ar, frame, last, refs, videos, ref_names,
                           reference_only, generate_audio, multi_clip):
    hashes = {
        'primary': source_sha256,
        'last': file_sha256(last) if last else None,
        'references': [file_sha256(path) for path in refs],
        'videos': [file_sha256(path) for path in videos],
    }
    return {
        'version': 1,
        'model': model,
        'jobId': job_id,
        'promptSha256': prompt_sha256,
        'sourceSha256': source_sha256,
        'executionManifestSha256': manifest_sha256,
        'executionRequestSha256': execution_request_sha256,
        'resolution': res,
        'durationSeconds': int(dur) if dur is not None else None,
        'aspectRatio': ar,
        'audio': generate_audio,
        'referenceOnly': reference_only,
        'multiClip': multi_clip,
        'inputHashes': hashes,
        'inputNames': {
            'primary': pathlib.Path(frame).name,
            'last': pathlib.Path(last).name if last else None,
            'references': [pathlib.Path(path).name for path in refs],
            'videos': [pathlib.Path(path).name for path in videos],
            'referenceNames': list(ref_names),
        },
    }


def validate_request_record(record, expected_local_sha):
    request = record.get('request')
    fingerprint = record.get('requestFingerprint')
    if not isinstance(request, dict) or not isinstance(fingerprint, str):
        sys.exit('recovery record lacks a canonical request fingerprint')
    if record.get('requestSha256') != fingerprint:
        sys.exit('recovery request digest aliases disagree')
    if canonical_sha256(request) != fingerprint:
        sys.exit('recovery request fingerprint does not match its frozen request')
    if request.get('localRequestSha256') != expected_local_sha:
        sys.exit('recovery request does not match current local inputs/options')
    return request, fingerprint


def validate_landed_pair(video, sidecar, pending, expected_local,
                         expected_local_sha):
    """Validate provenance and bytes before treating an existing pair as done."""
    video = pathlib.Path(video)
    sidecar = pathlib.Path(sidecar)
    pending = pathlib.Path(pending)
    if video.exists() != sidecar.exists():
        sys.exit(f'incomplete landed pair requires reconciliation: {video}')
    if not video.exists():
        return False
    meta = read_json(sidecar, 'video sidecar')
    required = {
        'provenanceVersion': PENDING_VERSION,
        'model': expected_local['model'],
        'jobId': expected_local['jobId'],
        'executionManifestSha256': expected_local['executionManifestSha256'],
        'executionRequestSha256': expected_local['executionRequestSha256'],
        'sourceSha256': expected_local['sourceSha256'],
        'promptSha256': expected_local['promptSha256'],
        'resolution': expected_local['resolution'],
        'durationSeconds': expected_local['durationSeconds'],
        'aspectRatio': expected_local['aspectRatio'],
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            sys.exit(f'landed sidecar mismatch for {key}: {sidecar}')
    if meta.get('localRequestSha256') != expected_local_sha:
        sys.exit(f'landed sidecar local request mismatch: {sidecar}')
    if (meta.get('localRequest') != expected_local
            or canonical_sha256(meta['localRequest']) != expected_local_sha):
        sys.exit(f'landed sidecar local request is not canonical: {sidecar}')
    request, fingerprint = validate_request_record(meta, expected_local_sha)
    if request.get('model') != expected_local['model']:
        sys.exit(f'landed request model mismatch: {sidecar}')
    output_sha = file_sha256(video)
    if (meta.get('outputSha256') != output_sha or meta.get('sha256') != output_sha):
        sys.exit(f'landed video hash does not match sidecar: {video}')
    task_id = meta.get('taskId')
    if not isinstance(task_id, str) or not task_id:
        sys.exit(f'landed sidecar has no taskId: {sidecar}')
    if pending.exists():
        orphan = read_json(pending, 'pending recovery record')
        _, pending_fingerprint = validate_request_record(orphan, expected_local_sha)
        if (orphan.get('status') not in {'PENDING', 'LANDED'}
                or orphan.get('taskId') != task_id
                or pending_fingerprint != fingerprint):
            sys.exit(f'landed pair conflicts with pending paid task: {pending}')
        pending.unlink()
    return True

def build_input(model, prompt, url, upload, *, res, dur, last, refs, videos,
                ref_names, ar, reference_only, generate_audio, multi_clip):
    """Return one provider input; ``upload`` resolves every extra file to a URL."""
    # Seedance 2 schema (docs.kie.ai, verified 2026-07-22): first frame goes
    # in first_frame_url (SINGULAR STRING — image_urls is invalid and was a
    # rejected API request); last_frame_url enables keyframe choreography;
    # duration 4-15s; fast tier caps at 720p; no seed field; audio defaults
    # ON so we force it off (assembly strips audio anyway).
    if model == 'google/gemini-omni-flash-1-1':
        if refs or videos:
            sys.exit('Gemini Omni Flash 1.1 frame mode accepts its first and '
                     'last frame here; free references are a separate payload')
        inp = {'first_frame_url': url,
               'prompt': prompt,
               'duration': str(dur or 8),
               'audio_ids': [],
               'aspect_ratio': ar or '16:9',
               'character_ids': [],
               'resolution': res,
               'seed': 0}
        if last:
            inp['last_frame_url'] = upload(last)
    elif model.startswith('bytedance/seedance'):
        # Seedance 2.5 has two deliberately exclusive visual modes on KIE:
        # first/last-frame choreography OR free reference-to-video.  Do not
        # smuggle identity refs beside a first frame; the live schema hides
        # all reference fields when first_frame_url is populated.
        if reference_only:
            if last:
                sys.exit('--last cannot be combined with --reference-only')
            inp = {'prompt': prompt,
                   'reference_image_urls': [url] + [upload(r) for r in refs],
                   'reference_video_urls': [upload(v) for v in videos],
                   'generate_audio': generate_audio, 'resolution': res,
                   'aspect_ratio': ar or '16:9', 'output_format': 'mp4',
                   'web_search': False, 'nsfw_checker': True}
        else:
            if videos:
                sys.exit('Seedance video references require --reference-only; '
                         'first-frame and multimodal-reference modes are exclusive')
            if refs:
                sys.exit('Seedance first-frame mode cannot accept --ref; '
                         'use --reference-only or remove the extra refs')
            # Seedance 2.5's first-frame and first/last-frame modes require
            # adaptive aspect handling. The supplied frame geometry remains
            # authoritative; an explicit 16:9 value is rejected by KIE.
            seedance_aspect = ('adaptive' if model == 'bytedance/seedance-2-5'
                               else (ar or '16:9'))
            inp = {'prompt': prompt, 'first_frame_url': url,
                   'generate_audio': generate_audio, 'resolution': res,
                   'aspect_ratio': seedance_aspect, 'output_format': 'mp4',
                   'web_search': False, 'nsfw_checker': True}
            if last:
                inp['last_frame_url'] = upload(last)
    elif model == 'gemini-omni-video':
        omni_urls = [url] + [upload(r) for r in refs]
        if len(videos) > 1:
            sys.exit('Gemini Omni accepts at most one video input')
        if len(omni_urls) + 2 * len(videos) > 7:
            sys.exit('Gemini Omni input quota exceeded: each image is one '
                     'unit, one video is two units, total must be <= 7')
        inp = {'prompt': prompt, 'image_urls': omni_urls,
               'aspect_ratio': ar or '16:9', 'resolution': res, 'seed': 0}
        if videos:
            # Official KIE contract: one video in video_list, two quota units.
            # With video input, Gemini determines output length; duration is
            # intentionally omitted rather than pretending it controls edit
            # length. `ends` is the selected source interval, not output time.
            source_end = int(dur) if dur else 10
            inp['video_list'] = [{'url': upload(videos[0]),
                                  'start': 0, 'ends': source_end}]
    elif model == 'kling-3.0-omni/image-to-video':
        if last:
            sys.exit('Kling 3.0 Omni start/end work requires a dedicated '
                     'two-frame request; --last is not yet exposed here')
        inp = {'image_urls': [url] + [upload(r) for r in refs],
               'prefer_multi_shots': False,
               'customize_multi_shots': False,
               'prompt': prompt, 'elements': [], 'audio': generate_audio,
               'resolution': res, 'aspect_ratio': ar or 'auto'}
    elif model == 'kling-3.0-omni/reference-to-video':
        if last:
            sys.exit('Kling 3.0 Omni reference-to-video does not use --last')
        kling_urls = [url] + [upload(r) for r in refs]
        if len(kling_urls) > 4:
            sys.exit('Kling 3.0 Omni reference-to-video accepts at most four images')
        if len(videos) > 1:
            sys.exit('Kling 3.0 Omni reference-to-video accepts at most one video')
        if ref_names and len(ref_names) != len(kling_urls):
            sys.exit('Kling 3.0 Omni reference-to-video requires one '
                     '--ref-name for every image')
        names = ref_names or [f'ref_{n + 1}' for n in range(len(kling_urls))]
        # The first image remains the scene authority. Every additional image
        # is also exposed as a named Kling element so plain-English prompts can
        # assign identity/object ownership without positional @image syntax.
        inp = {'image_urls': kling_urls,
               'video_urls': [upload(v) for v in videos],
               'customize_multi_shots': False,
               'prompt': prompt,
               'kling_elements': [
                   {'name': names[n], 'description': names[n].replace('_', ' '),
                    'element_input_urls': [u]}
                   for n, u in enumerate(kling_urls[1:], start=1)
               ],
               'resolution': res, 'aspect_ratio': ar or '16:9',
               'audio': generate_audio}
    elif model == 'pixverse-v6/image-to-video':
        # PixVerse V6 image-to-video treats image_urls as source frames rather
        # than free identity references. Keep this lane deliberately narrow:
        # animate one approved scene plate (or at most the two files accepted
        # by KIE), and describe motion rather than repainting the frame.
        if last:
            sys.exit('PixVerse V6 first/last-frame work uses its transition '
                     'endpoint, not image-to-video --last')
        pixverse_urls = [url] + [upload(r) for r in refs]
        if len(pixverse_urls) > 2:
            sys.exit('PixVerse V6 image-to-video accepts at most two images')
        inp = {'prompt': prompt, 'image_urls': pixverse_urls,
               'quality': res, 'generate_audio_switch': generate_audio,
               'seed': 0, 'generate_multi_clip_switch': multi_clip}
        if dur:
            inp['duration'] = int(dur)
    elif model == 'pixverse-v6/transition':
        if not last:
            sys.exit('PixVerse V6 transition requires --last')
        if refs or videos:
            sys.exit('PixVerse V6 transition accepts only its first and last frames')
        inp = {'prompt': prompt,
               'duration': int(dur or 10),
               'first_frame_image_url': url,
               'last_frame_image_url': upload(last),
               'quality': res,
               'generate_audio_switch': generate_audio,
               'seed': 0}
    elif model == 'pixverse-v6/reference-to-video':
        if last:
            sys.exit('PixVerse Fusion does not use --last')
        pixverse_urls = [url] + [upload(r) for r in refs]
        if len(pixverse_urls) > 7:
            sys.exit('PixVerse V6 Fusion accepts at most seven references')
        if ref_names and len(ref_names) != len(pixverse_urls):
            sys.exit('PixVerse Fusion requires one --ref-name for every '
                     'image (the primary frame plus every --ref)')
        names = ref_names or ['assembled', 'mechanism', 'motion_map']
        inp = {'prompt': prompt,
               'image_references': [
                   {'image_url': u, 'type': 'subject',
                    'ref_name': names[n] if n < len(names) else f'ref_{n + 1}'}
                   for n, u in enumerate(pixverse_urls)],
               'aspect_ratio': ar or '16:9', 'quality': res,
               'generate_audio_switch': generate_audio, 'seed': 0}
    elif model == 'pixverse-v6/extend':
        if last or refs or videos:
            sys.exit('PixVerse V6 Extend accepts one parent video only; '
                     'pass it as the primary input without --last/--ref/--video')
        inp = {'video_url': url, 'prompt': prompt,
               'quality': res, 'generate_audio_switch': generate_audio,
               'seed': 0}
    elif model == 'veo3_fast':
        if last:
            sys.exit('Veo reference-to-video does not use --last')
        veo_urls = [url] + [upload(r) for r in refs]
        if len(veo_urls) > 3:
            sys.exit('Veo 3.1 Fast reference-to-video accepts at most three references')
        inp = {'prompt': prompt, 'imageUrls': veo_urls, 'model': 'veo3_fast',
               'aspect_ratio': ar or '16:9', 'resolution': res,
               'generationType': 'REFERENCE_2_VIDEO',
               'enableFallback': False, 'enableTranslation': True}
    else:
        # Grok family: extra --ref images join image_urls in order (@image2..N
        # in the prompt; first_frame is @image1). aspect_ratio is MANDATORY on
        # multi-ref calls (KIE.md: omitting it births portrait clips).
        inp = {'prompt': prompt,
               'image_urls': [url] + [upload(r) for r in refs],
               'resolution': res}
        if ar:
            inp['aspect_ratio'] = ar
        elif refs:
            inp['aspect_ratio'] = '16:9'
    if dur and model not in {'veo3_fast', 'google/gemini-omni-flash-1-1'} and not (model == 'gemini-omni-video' and videos):
        inp['duration'] = str(dur) if model == 'gemini-omni-video' else int(dur)
    return inp


PLACEHOLDER_URL = 'https://placeholder.invalid/not-uploaded/'


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Routes: ' + ', '.join(f'{k}={v}' for k, v in sorted(MODELS.items()))
               + '. Any other value is sent as a literal Kie model id.')
    parser.add_argument('route', help='route alias or literal Kie model id')
    parser.add_argument('out_stem', help='output stem; writes .mp4, .vid.json and .pending.json')
    parser.add_argument('prompt_file', help='prompt text file (its bytes are hash-bound)')
    parser.add_argument('frame', help='first frame, primary reference or parent video')
    parser.add_argument('--res', default='720p', help='provider resolution (default 720p)')
    parser.add_argument('--dur', help='requested seconds, sent in the provider\'s own type')
    parser.add_argument('--ar', help='aspect ratio; Grok multi-reference calls default to 16:9')
    parser.add_argument('--last', help='last-frame image where the route accepts one')
    parser.add_argument('--ref', action='append', help='extra reference image, in order')
    parser.add_argument('--video', action='append', help='reference or source video')
    parser.add_argument('--ref-name', action='append', help='semantic name for each image')
    parser.add_argument('--reference-only', action='store_true',
                        help='Seedance reference mode instead of first-frame mode')
    parser.add_argument('--audio', action='store_true', help='request generated audio')
    parser.add_argument('--multi-clip', action='store_true', help='PixVerse multi-clip switch')
    parser.add_argument('--job-id', help='frozen job id (dialogue runner)')
    parser.add_argument('--manifest-sha256', help='frozen execution manifest SHA-256')
    parser.add_argument('--source-sha256', help='expected primary input SHA-256')
    parser.add_argument('--execution-request-sha256',
                        help='canonical execution request SHA-256 from the runner')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--print-request', action='store_true',
                       help='print the provider request with placeholder URLs; no upload, '
                            'credential, wallet or recovery-record access')
    modes.add_argument('--resume-only', action='store_true',
                       help='poll an already-recorded taskId; refuse rather than POST')
    return parser.parse_intermixed_args(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    options = parse_arguments(argv)
    res, dur, last, ar = options.res, options.dur, options.last, options.ar
    refs, videos, ref_names = options.ref or [], options.video or [], options.ref_name or []
    manifest_sha256, source_sha256_arg = options.manifest_sha256, options.source_sha256
    execution_request_sha256, job_id = options.execution_request_sha256, options.job_id
    reference_only, generate_audio = options.reference_only, options.audio
    multi_clip, resume_only = options.multi_clip, options.resume_only
    model = MODELS.get(options.route, options.route)
    name, prompt_file, frame = options.out_stem, options.prompt_file, options.frame
    embedded_audio = generate_audio or model in {
        'grok-imagine/image-to-video', 'grok-imagine-video-1-5-preview',
        'gemini-omni-video', 'google/gemini-omni-flash-1-1'}
    landed = pathlib.Path(str(name) + '.mp4')
    landed_meta = pathlib.Path(str(name) + '.vid.json')
    landed_pending = pathlib.Path(str(name) + '.pending.json')
    prompt = pathlib.Path(prompt_file).read_text().strip()
    # Provenance binds the complete prompt file bytes (including its authored
    # newline); the canonical provider request separately binds stripped text.
    prompt_sha256 = file_sha256(prompt_file)
    prompt_text_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    actual_source_sha256 = file_sha256(frame)
    source_sha256_arg = checked_optional_sha(source_sha256_arg, '--source-sha256')
    manifest_sha256 = checked_optional_sha(manifest_sha256, '--manifest-sha256')
    execution_request_sha256 = checked_optional_sha(
        execution_request_sha256, '--execution-request-sha256')
    if source_sha256_arg and source_sha256_arg != actual_source_sha256:
        sys.exit('--source-sha256 does not match the primary input bytes')
    source_sha256 = source_sha256_arg or actual_source_sha256
    job_id = job_id or landed.name
    local_request = local_request_contract(
        model=model, prompt_sha256=prompt_sha256,
        source_sha256=source_sha256, manifest_sha256=manifest_sha256,
        execution_request_sha256=execution_request_sha256, job_id=job_id,
        res=res, dur=dur, ar=ar, frame=frame, last=last, refs=refs,
        videos=videos, ref_names=ref_names, reference_only=reference_only,
        generate_audio=embedded_audio, multi_clip=multi_clip)
    local_request_sha256 = canonical_sha256(local_request)
    input_options = dict(res=res, dur=dur, last=last, refs=refs, videos=videos,
                         ref_names=ref_names, ar=ar, reference_only=reference_only,
                         generate_audio=generate_audio, multi_clip=multi_clip)
    if options.print_request:
        def placeholder(path):
            return PLACEHOLDER_URL + pathlib.Path(path).name
        inp = build_input(model, prompt, placeholder(frame), placeholder, **input_options)
        endpoint = 'veo/generate' if model == 'veo3_fast' else 'jobs/createTask'
        body = inp if model == 'veo3_fast' else {'model': model, 'input': inp}
        print(json.dumps({
            'mode': 'PRINT_REQUEST', 'networkCalls': 0,
            'note': 'URLs are placeholders for files a submission would upload',
            'endpoint': endpoint, 'body': body, 'localRequest': local_request,
            'localRequestSha256': local_request_sha256,
        }, ensure_ascii=False, sort_keys=True, indent=2))
        return
    if validate_landed_pair(landed, landed_meta, landed_pending,
                            local_request, local_request_sha256):
        print(f'{landed} exists with a validated sidecar — skip')
        return
    # Refuse ambiguous/legacy recovery state before even uploading references or
    # querying a wallet. Only an already-recorded taskId is safe to poll.
    if landed_pending.exists():
        early_orphan = read_json(landed_pending, 'pending recovery record')
        if early_orphan.get('provenanceVersion') != PENDING_VERSION:
            sys.exit(f'legacy pending record requires manual reconciliation: '
                     f'{landed_pending}')
        if early_orphan.get('localRequestSha256') != local_request_sha256:
            sys.exit(f'pending task payload mismatch at {landed_pending} — '
                     'resolve it before changing any input or option')
        validate_request_record(early_orphan, local_request_sha256)
        if (early_orphan.get('status') != 'PENDING'
                or not isinstance(early_orphan.get('taskId'), str)
                or not early_orphan['taskId']):
            sys.exit(f'pending task state {early_orphan.get("status")!r} requires '
                     f'manual reconciliation: {landed_pending}')
    if resume_only and not landed_pending.exists():
        sys.exit(f'no recorded paid task to resume at {landed_pending}; '
                 'resume-only never creates a generation request')
    # Read before any paid call so a malformed profile cannot fail after one.
    usd_rate = usd_per_credit()
    t0 = time.time()
    if resume_only:
        # Recovery polls a recorded taskId; it never uploads or reads a wallet.
        inp = wallet_before = None
    else:
        url = upload_ref(frame)
        wallet_before = kie('chat/credit')
        inp = build_input(model, prompt, url, upload_ref, **input_options)
    # Persist uncertainty *before* the paid POST.  A crash in the small interval
    # between provider acceptance and receiving a taskId must block, never turn
    # into a second automatic generation.
    pending = landed_pending
    pending.parent.mkdir(parents=True, exist_ok=True)
    if pending.exists():
        orphan = read_json(pending, 'pending recovery record')
        if orphan.get('provenanceVersion') != PENDING_VERSION:
            sys.exit(f'legacy pending record requires manual reconciliation: {pending}')
        if orphan.get('localRequestSha256') != local_request_sha256:
            sys.exit(f'pending task payload mismatch at {pending} — resolve it '
                     'before changing any input or option')
        request, request_fingerprint = validate_request_record(
            orphan, local_request_sha256)
        status = orphan.get('status')
        task = orphan.get('taskId')
        if status == 'POSTING_UNCERTAIN' and not task:
            sys.exit(f'prior generation POST outcome is uncertain at {pending}; '
                     'recover its taskId from provider logs before any retry')
        if status != 'PENDING' or not isinstance(task, str) or not task:
            sys.exit(f'pending task state {status!r} requires manual reconciliation: '
                     f'{pending}')
        inp = request['input']
        wallet_before = orphan.get('walletBefore', wallet_before)
        print(f'resuming paid task {task} from {pending}', flush=True)
    else:
        request = {
            'version': 1,
            'model': model,
            'input': inp,
            'sourceSha256': source_sha256,
            'promptSha256': prompt_sha256,
            'promptTextSha256': prompt_text_sha256,
            'executionManifestSha256': manifest_sha256,
            'executionRequestSha256': execution_request_sha256,
            'localRequestSha256': local_request_sha256,
        }
        request_fingerprint = canonical_sha256(request)
        orphan = {
            'provenanceVersion': PENDING_VERSION,
            'status': 'POSTING_UNCERTAIN',
            'taskId': None,
            'jobId': job_id,
            'model': model,
            'request': request,
            'requestFingerprint': request_fingerprint,
            'requestSha256': request_fingerprint,
            'localRequest': local_request,
            'localRequestSha256': local_request_sha256,
            'executionManifestSha256': manifest_sha256,
            'executionRequestSha256': execution_request_sha256,
            'sourceSha256': source_sha256,
            'promptSha256': prompt_sha256,
            'promptTextSha256': prompt_text_sha256,
            'walletBefore': wallet_before,
            'postStartedAt': time.time(),
        }
        atomic_json(pending, orphan, replace=False)
        if model == 'veo3_fast':
            response = kie('veo/generate', inp)
        else:
            response = kie('jobs/createTask', {'model': model, 'input': inp})
        task = response.get('taskId') if isinstance(response, dict) else None
        if not isinstance(task, str) or not task:
            sys.exit(f'generation POST returned no taskId; outcome remains uncertain: '
                     f'{pending}')
        orphan.update(status='PENDING', taskId=task,
                      postFinishedAt=time.time(), providerCreateResponse=response)
        atomic_json(pending, orphan, replace=True)
        print(f'created paid task {task}; recovery record {pending}', flush=True)
    state, rec = 'waiting', None
    while time.time() - t0 < 1800:  # degraded kie windows queue >10min; pending.json covers a true orphan
        time.sleep(6)
        if model == 'veo3_fast':
            rec = kie(f'veo/record-info?taskId={task}')
            flag = rec.get('successFlag')
            state = 'success' if flag == 1 else ('fail' if flag in (2, 3) else 'waiting')
        else:
            rec = kie(f'jobs/recordInfo?taskId={task}')
            state = rec.get('state')
        if state in ('success', 'fail'):
            break
    if state == 'fail':
        orphan.update(status='GENERATION_FAILED', lastProviderResponse=rec,
                      providerFinishedAt=time.time())
        atomic_json(pending, orphan, replace=True)
    elif state != 'success':
        orphan.update(status='PENDING', lastProviderResponse=rec,
                      lastPolledAt=time.time())
        atomic_json(pending, orphan, replace=True)
    if state != 'success':
        sys.exit(f'task {task} ended {state}: {json.dumps(rec)[:500]}')
    if model == 'veo3_fast':
        urls = (rec.get('response') or {}).get('resultUrls') or []
        if isinstance(urls, str):
            urls = json.loads(urls)
    else:
        urls = json.loads(rec['resultJson']).get('resultUrls') or []
    if not urls:
        sys.exit('no resultUrls: ' + json.dumps(rec)[:500])
    mp4 = download_public_media(urls[0])
    output_sha256 = hashlib.sha256(mp4).hexdigest()
    # A provider receipt is per-task evidence; a wallet delta can include
    # other tasks running at the same time, so it is labelled as such.
    credits = rec.get('creditsConsumed')
    credits_source = 'receipt' if credits is not None else None
    if credits is None:
        saved = read_json(pending, 'pending recovery record')
        before = saved.get('walletBefore', wallet_before)
        if before is not None:
            credits = max(0, float(before) - float(kie('chat/credit')))
            credits_source = 'wallet-delta'
    meta = {'provenanceVersion': PENDING_VERSION,
            'provider': 'kie.ai', 'model': model, 'taskId': task,
            'jobId': job_id,
            'executionManifestSha256': manifest_sha256,
            'executionRequestSha256': execution_request_sha256,
            'requestFingerprint': request_fingerprint,
            'requestSha256': request_fingerprint,
            'request': request,
            'localRequest': local_request,
            'localRequestSha256': local_request_sha256,
            'sourceSha256': source_sha256,
            'promptSha256': prompt_sha256,
            'outputSha256': output_sha256,
            'sha256': output_sha256,
            'durationSeconds': int(dur) if dur is not None else None,
            'resolution': res,
            'aspectRatio': local_request['aspectRatio'],
            'audio': embedded_audio,
            'input': request['input'],
            'credits': credits,
            'creditsSource': credits_source,
            'cost_usd': round(credits * usd_rate, 4) if credits and usd_rate else None,
            'res': res, 'dur': dur, 'first_frame': frame,
            'last_frame': last,
            'refs': [frame] + refs, 'aspect_ratio': inp.get('aspect_ratio'),
            'reference_names': ref_names,
            'video_inputs': videos,
            'reference_only': reference_only,
            'multi_clip': multi_clip,
            'generate_audio': embedded_audio,
            'secs': round(time.time() - t0, 1), 'prompt_file': prompt_file}
    if landed.exists() or landed_meta.exists():
        sys.exit(f'refusing to overwrite output or sidecar: {landed}')
    try:
        atomic_new_bytes(landed, mp4)
        atomic_json(landed_meta, meta, replace=False)
    except FileExistsError as exc:
        sys.exit(f'refusing output publication race at {landed}: {exc}')
    orphan.update(status='LANDED', outputSha256=output_sha256,
                  sidecarSha256=file_sha256(landed_meta), landedAt=time.time())
    atomic_json(pending, orphan, replace=True)
    pending.unlink()
    cost = f'${meta["cost_usd"]}' if meta['cost_usd'] is not None else f'{credits} credits'
    print(f'{landed}  {cost}  {meta["secs"]}s  {model} {res}')

if __name__ == '__main__':
    main()
