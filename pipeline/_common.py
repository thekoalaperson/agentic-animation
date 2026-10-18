"""Small shared configuration, hashing, atomic-file and HTTP helpers for the
production tools.

Code, projects and credentials have separate roots. A profile supplies defaults,
never permission to spend, approve creative work, or publish an artifact.
"""
import base64, hashlib, json, math, mimetypes, os, pathlib, sys, urllib.error, urllib.parse, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_profile():
    """Read an explicitly selected private profile; no machine-specific default."""
    location = os.environ.get('AGENTIC_PROFILE')
    if not location:
        return {}
    path = pathlib.Path(location).expanduser().resolve()
    with path.open(encoding='utf-8') as source:
        profile = json.load(source)
    if not isinstance(profile, dict):
        raise ValueError('AGENTIC_PROFILE must contain a JSON object')
    return profile


def profile_value(*keys, default=None):
    value = load_profile()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def usd_per_credit():
    """The operator's own Kie USD-per-credit figure from the profile, or None.

    There is deliberately no default: a stale built-in rate would read as a
    price quote. Set ``providers.kie.usdPerCredit`` from the provider's page.
    """
    value = profile_value('providers', 'kie', 'usdPerCredit')
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError('providers.kie.usdPerCredit must be a positive number')
    return float(value)


def project_root(path=None):
    """Resolve an explicit project or the configured private project root."""
    value = path or os.environ.get('AGENTIC_PROJECT_ROOT') or profile_value('projectRoot')
    if not value:
        raise ValueError('Provide a project path or set AGENTIC_PROJECT_ROOT')
    result = pathlib.Path(value).expanduser().resolve()
    if not result.is_dir():
        raise ValueError('Project root must be an existing directory')
    return result


def is_below(path, root):
    try:
        pathlib.Path(path).resolve().relative_to(pathlib.Path(root).resolve())
        return True
    except ValueError:
        return False


def portable_path(path, episode):
    """Episode-relative text for a path inside the episode, else its absolute form."""
    try:
        return str(pathlib.Path(path).resolve().relative_to(pathlib.Path(episode).resolve()))
    except ValueError:
        return str(pathlib.Path(path).resolve())


def state_path_matches(recorded, actual, episode, original_episode_root=None):
    """Retain immutable worker receipts across an explicitly declared relocation.

    Callers separately verify the media/sidecar digest and completed state. This
    checks exact episode-relative identity, not a permissive basename match.
    """
    raw = pathlib.Path(str(recorded or ''))
    actual = pathlib.Path(actual)
    episode = pathlib.Path(episode)
    if raw.resolve() == actual.resolve():
        return True
    if not original_episode_root or not raw.is_absolute() or '..' in raw.parts:
        return False
    root = pathlib.Path(str(original_episode_root))
    if not root.is_absolute() or '..' in root.parts or root.name != episode.name:
        return False
    try:
        recorded_relative = raw.relative_to(root)
        actual_relative = actual.resolve().relative_to(episode.resolve())
    except ValueError:
        return False
    return recorded_relative == actual_relative and bool(recorded_relative.parts)


def load_relocation_map(path):
    """Read an explicit relocation map for a moved project.

    ``{"originalEpisodeRoot": "/old/abs/project", "paths": {"/old/abs/file":
    "project/relative/file"}}``; either key may be omitted. A map only says
    where a recorded path now lives; callers still compare content hashes.
    """
    if path is None:
        return None
    value = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not set(value) <= {'originalEpisodeRoot', 'paths'}:
        raise ValueError('relocation map holds only originalEpisodeRoot and paths')
    root = value.get('originalEpisodeRoot')
    if root is not None and (not isinstance(root, str) or not pathlib.Path(root).is_absolute()):
        raise ValueError('relocation map originalEpisodeRoot must be an absolute path')
    paths = value.get('paths', {})
    if (not isinstance(paths, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) or pathlib.Path(v).is_absolute()
                   or '..' in pathlib.Path(v).parts for k, v in paths.items())):
        raise ValueError('relocation map paths must map recorded paths to project-relative ones')
    return {'originalEpisodeRoot': root, 'paths': dict(paths)}


def relocated_path_matches(recorded, actual, episode, relocation=None, recorded_relative=None):
    """True when a recorded path names ``actual`` in this (possibly moved) project.

    Accepted: the same path; an episode-relative path recorded beside the
    absolute one that equals ``actual``'s; or an explicit relocation-map entry.
    Content hashes are the caller's separate, unchanged check.
    """
    actual = pathlib.Path(actual).resolve()
    episode = pathlib.Path(episode).resolve()
    if recorded and pathlib.Path(str(recorded)).resolve() == actual:
        return True
    if isinstance(recorded_relative, str) and recorded_relative:
        relative = pathlib.Path(recorded_relative)
        if not relative.is_absolute() and '..' not in relative.parts and (episode / relative).resolve() == actual:
            return True
    if relocation:
        mapped = relocation.get('paths', {}).get(str(recorded))
        if mapped is not None and (episode / mapped).resolve() == actual:
            return True
        return state_path_matches(recorded, actual, episode, relocation.get('originalEpisodeRoot'))
    return False


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_new_bytes(path, raw):
    """Publish new bytes atomically without ever replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.new')
    try:
        with tmp.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        # Linking a complete same-directory temp file makes publication atomic,
        # while link()'s EEXIST behavior closes the check-then-replace race.
        os.link(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_attempt_json(directory, stem, value):
    """Write ``<stem>-attempt-NNN.json`` as a new file; earlier attempts survive.

    Receipts and credit readings from an interrupted run are evidence for the
    next one, so a resume adds a numbered file instead of replacing the last.
    """
    directory = pathlib.Path(directory)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
    number = len(attempt_records(directory, stem)) + 1
    while True:
        path = directory / f'{stem}-attempt-{number:03d}.json'
        try:
            atomic_new_bytes(path, raw)
            return path
        except FileExistsError:
            number += 1


def attempt_records(directory, stem):
    """Return every ``<stem>-attempt-NNN.json`` record in attempt order."""
    directory = pathlib.Path(directory)
    found = []
    for path in directory.glob(f'{stem}-attempt-*.json'):
        number = path.name[len(stem) + len('-attempt-'):-len('.json')]
        if number.isdigit():
            found.append((int(number), path))
    return [{'attempt': number, 'path': path.name,
             'record': json.loads(path.read_text(encoding='utf-8'))}
            for number, path in sorted(found)]


def adopt_legacy_record(directory, stem):
    """Carry a pre-attempt ``<stem>.json`` into the numbered attempts, once.

    Wrappers older than the attempt files wrote a single ``<stem>.json`` that
    each run replaced, and ``execution-receipt.json`` is now the attempt index.
    Before either is written again, an old full record becomes an attempt
    marked ``legacyRecord`` so the index lists it. The old file is left in
    place; an index or an already adopted record is not adopted again.
    """
    directory = pathlib.Path(directory)
    path = directory / f'{stem}.json'
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(record, dict) and isinstance(record.get('attempts'), list):
        return None
    for attempt in attempt_records(directory, stem):
        if attempt['record'].get('legacyRecord') == path.name:
            return None
    return write_attempt_json(directory, stem, {'legacyRecord': path.name, 'record': record})


def env_key(name):
    # GPU workers receive approved credentials through SSH stdin and child
    # process memory, without staging a secret file in the production bundle.
    if os.environ.get(name):
        return os.environ[name]
    location = os.environ.get('AGENTIC_ENV_FILE')
    if location:
        path = pathlib.Path(location).expanduser().resolve()
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith(name + '='):
                return line.split('=', 1)[1].strip().strip('\"\'')
    sys.exit(f'Set {name} in the process environment or an external AGENTIC_ENV_FILE')


def http_json(url, payload=None, bearer=None, raw=False, timeout=300):
    """POST payload (or GET if None) with Bearer auth; JSON (or raw bytes) back.
    HTTP errors exit with a readable message instead of a traceback."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={**({'Authorization': 'Bearer ' + bearer} if bearer else {}),
                 'Content-Type': 'application/json',
                 # kie.ai's file host sits behind Cloudflare bot protection,
                 # which 403s (error 1010) the default Python-urllib agent
                 'User-Agent': 'Mozilla/5.0 agentic-animation'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read() if raw else json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f'HTTP {e.code} from {url}: {e.read().decode()[:600]}')


API = 'https://api.kie.ai/api/v1/'
UPLOAD = 'https://kieai.redpandaai.co/api/file-base64-upload'

def kie(path, payload=None):
    r = http_json(API + path, payload, bearer=env_key('KIE_KEY'))
    if r.get('code') != 200:
        sys.exit(f'kie.ai error {r.get("code")}: {r.get("msg")}')
    return r['data']


def log_upload(log, raw, host, remote_name):
    """Append one line per provider upload to the project's own log."""
    import datetime
    log = pathlib.Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), 'providerHost': host,
                       'remoteName': remote_name,
                       'uploadedAtUtc': datetime.datetime.now(datetime.timezone.utc)
                       .strftime('%Y-%m-%dT%H:%M:%SZ')}, sort_keys=True)
    with log.open('a', encoding='utf-8') as target:
        target.write(line + '\n')
        target.flush()
        os.fsync(target.fileno())


# Sidecars beside a file that may declare its rights. A supplied-media
# sidecar is <name>.supplied.json; a registered still's is <stem>.imagegen.json.
RIGHTS_SIDECARS = (lambda p: p.with_name(p.name + '.supplied.json'),
                   lambda p: p.with_suffix('.imagegen.json'))


def provider_upload_forbidden(p):
    """The sidecar that forbids sending this file to a provider, or None.

    A guard against accidental upload of material its maker withheld, not DRM:
    the declaration lives in the sidecar the operator registered.
    """
    p = pathlib.Path(p)
    for sidecar_of in RIGHTS_SIDECARS:
        sidecar = sidecar_of(p)
        if not sidecar.is_file():
            continue
        try:
            record = json.loads(sidecar.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'cannot read the rights sidecar {sidecar}: {exc}') from exc
        rights = record.get('rights') if isinstance(record, dict) else None
        if isinstance(rights, dict) and rights.get('providerUpload') == 'forbidden':
            return sidecar
    return None


def upload_ref(p, log=None):
    """Upload one file to Kie's file host and return its URL.

    Each upload is appended to ``log`` (or ``AGENTIC_UPLOAD_LOG``), a
    project-local record of which bytes left the machine and where. With
    neither, nothing is uploaded. A file whose sidecar says
    ``rights.providerUpload: forbidden`` is never uploaded.
    """
    p = pathlib.Path(p)
    forbidden = provider_upload_forbidden(p)
    if forbidden is not None:
        raise ValueError(f'refusing to upload {p.name}: {forbidden.name} says '
                         'rights.providerUpload is forbidden')
    log = log or os.environ.get('AGENTIC_UPLOAD_LOG')
    if not log:
        raise ValueError(f'refusing to upload {p.name} without an upload log: '
                         'pass a log path or set AGENTIC_UPLOAD_LOG')
    mime = mimetypes.guess_type(p.name)[0] or 'image/png'
    raw = p.read_bytes()
    data = f'data:{mime};base64,' + base64.b64encode(raw).decode()
    # The provider keys uploads by path and filename. Episode-local names
    # such as b01.png collide during concurrent production and can silently
    # feed a task another episode's source frame. Content-address the remote
    # filename so different images can never overwrite one another.
    remote_name = f'{hashlib.sha256(raw).hexdigest()[:16]}_{p.name}'
    r = http_json(UPLOAD, {'base64Data': data, 'uploadPath': 'ep-refs',
                           'fileName': remote_name}, bearer=env_key('KIE_KEY'))
    d = r.get('data') or {}
    url = d.get('downloadUrl') or d.get('fileUrl') or d.get('url')
    if not url:
        sys.exit('upload gave no URL: ' + json.dumps(r)[:400])
    log_upload(log, raw, urllib.parse.urlsplit(UPLOAD).hostname, remote_name)
    return url



def image_dimensions(path):
    """Return (format, width, height) read from a PNG, JPEG or WebP header.

    No pixels are decoded, so this runs on any host. An unknown or truncated
    header raises ValueError naming the file.
    """
    path = pathlib.Path(path)
    with path.open('rb') as source:
        head = source.read(64)
        if head[:8] == b'\x89PNG\r\n\x1a\n' and head[12:16] == b'IHDR' and len(head) >= 24:
            return 'png', int.from_bytes(head[16:20], 'big'), int.from_bytes(head[20:24], 'big')
        if head[:4] == b'RIFF' and head[8:12] == b'WEBP' and len(head) >= 30:
            chunk = head[12:16]
            if chunk == b'VP8X':
                return ('webp', 1 + int.from_bytes(head[24:27], 'little'),
                        1 + int.from_bytes(head[27:30], 'little'))
            if chunk == b'VP8 ' and head[23:26] == b'\x9d\x01\x2a':
                return ('webp', int.from_bytes(head[26:28], 'little') & 0x3FFF,
                        int.from_bytes(head[28:30], 'little') & 0x3FFF)
            if chunk == b'VP8L' and head[20] == 0x2F:
                bits = int.from_bytes(head[21:25], 'little')
                return 'webp', (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if head[:2] == b'\xff\xd8':
            source.seek(2)
            while True:
                byte = source.read(1)
                while byte and byte != b'\xff':
                    byte = source.read(1)
                while byte == b'\xff':
                    byte = source.read(1)
                if not byte:
                    break
                marker = byte[0]
                if marker in (0x01, *range(0xD0, 0xD8)):
                    continue
                length = int.from_bytes(source.read(2), 'big')
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    frame = source.read(5)
                    if len(frame) == 5:
                        return ('jpeg', int.from_bytes(frame[3:5], 'big'),
                                int.from_bytes(frame[1:3], 'big'))
                    break
                if length < 2:
                    break
                source.seek(length - 2, 1)
    raise ValueError(f'not a readable PNG, JPEG or WebP header: {path}')


class MediaWorkerRequired(RuntimeError):
    """Raised when exact media decoding is attempted off the qualified worker."""


def require_media_worker(reason):
    """Refuse work that genuinely needs CUDA or a qualified model runtime.

    Production media (H3, upscale, model audio, exact-frame extraction,
    production mixes and qualified typography) stays on the configured GPU
    worker. Review previews and auditions use preview_host_record instead.
    Returns the GPU listing so callers can record it.
    """
    import shutil, subprocess
    if sys.platform != 'linux' or not shutil.which('nvidia-smi'):
        raise MediaWorkerRequired(f'{reason} must run on the qualified GPU worker, not this host')
    try:
        listing = subprocess.run(['nvidia-smi', '-L'], check=True, capture_output=True,
                                 text=True, timeout=60)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise MediaWorkerRequired(f'{reason}: nvidia-smi found no usable GPU on the qualified GPU worker') from exc
    return listing.stdout.strip()


_FFMPEG_CAPABILITIES = None


def ffmpeg_capabilities():
    """The local ffmpeg build: its version line and the filters it offers.

    Read once per process. A missing ffmpeg raises, since every preview needs it.
    """
    global _FFMPEG_CAPABILITIES
    if _FFMPEG_CAPABILITIES is None:
        import shutil, subprocess
        binary = shutil.which('ffmpeg')
        if not binary:
            raise RuntimeError('ffmpeg is required for review previews and auditions')
        version = subprocess.run([binary, '-hide_banner', '-version'], check=True,
                                 capture_output=True, text=True).stdout.splitlines()[0].strip()
        listing = subprocess.run([binary, '-hide_banner', '-filters'], check=True,
                                 capture_output=True, text=True).stdout.splitlines()
        filters = sorted({line.split()[1] for line in listing
                          if len(line.split()) >= 3 and line[:1] == ' ' and '->' in line})
        _FFMPEG_CAPABILITIES = {'path': binary, 'version': version, 'filters': filters}
    return _FFMPEG_CAPABILITIES


def needs_complex_shaping(text):
    """True when text holds a script that renders wrongly without a shaper.

    Covers the Indic blocks, Arabic, Hebrew, Thai and Myanmar. Latin, Greek,
    Cyrillic and CJK draw correctly with a plain glyph-by-glyph path.
    """
    for char in text or '':
        code = ord(char)
        if (0x0590 <= code <= 0x08FF or 0x0900 <= code <= 0x0DFF or 0x0E00 <= code <= 0x0EFF
                or 0x1000 <= code <= 0x109F or 0xA8E0 <= code <= 0xA8FF):
            return True
    return False


def preview_host_record(font=None):
    """What a preview or audition was actually made with, for its audit.

    Records the real host, platform, ffmpeg build and the font's hash. It
    never claims the qualified worker: a preview is labelled not final.
    """
    import platform
    caps = ffmpeg_capabilities()
    font_path = pathlib.Path(font).resolve() if font else None
    return {
        'computeHost': platform.node(),
        'platform': sys.platform,
        'ffmpeg': caps['version'],
        'ffmpegShapingFilters': [name for name in ('ass', 'subtitles', 'drawtext') if name in caps['filters']],
        'font': ({'path': str(font_path), 'sha256': sha256_file(font_path)} if font_path else None),
        'finalApproved': False,
    }


def review_font(explicit=None):
    """A review-sheet font: --font, else the profile's fonts.review, else None."""
    value = explicit or profile_value('fonts', 'review')
    if not value:
        return None
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Review font does not exist: {path}')
    return path


def extract_exact_frame(source, frame_index, destination):
    """Decode one exact frame by index on the qualified GPU worker.

    This is the extractor the measured-deficit builder has always used. It
    writes a temporary file and publishes it only once a frame exists.
    """
    import subprocess
    require_media_worker('Media extraction')
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + '.partial.png')
    if temporary.exists():
        temporary.unlink()
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-i', str(source),
         '-map', '0:v:0', '-vf', f'select=eq(n\\,{frame_index})',
         '-frames:v', '1', '-vsync', '0', str(temporary)],
        check=True,
    )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise ValueError(f'Exact frame extraction produced no frame: {source} n={frame_index}')
    temporary.replace(destination)
