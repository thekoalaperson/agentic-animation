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

