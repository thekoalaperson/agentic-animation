"""Small shared configuration, hashing, atomic-file and HTTP helpers for the
production tools.

Code, projects and credentials have separate roots. A profile supplies defaults,
never permission to spend, approve creative work, or publish an artifact.
"""
import base64, hashlib, json, mimetypes, os, pathlib, sys, urllib.error, urllib.request

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


def upload_ref(p):
    p = pathlib.Path(p)
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


def extract_exact_frame(source, frame_index, destination):
    """Decode one exact frame by index on the qualified GPU worker.

    This is the extractor the measured-deficit builder has always used. It
    writes a temporary file and publishes it only once a frame exists.
    """
    import shutil, subprocess
    if sys.platform != 'linux' or not shutil.which('nvidia-smi'):
        raise MediaWorkerRequired('Media extraction must run on the qualified GPU worker, not the Mac')
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
