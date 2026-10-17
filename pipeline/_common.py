"""Small shared configuration, hashing, atomic-file and HTTP helpers for the
production tools.

Code, projects and credentials have separate roots. A profile supplies defaults,
never permission to spend, approve creative work, or publish an artifact.
"""
import base64, hashlib, json, math, mimetypes, os, pathlib, sys, urllib.error, urllib.request

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

