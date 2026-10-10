"""Create silent, stream-copied H3 production candidates on the GPU worker."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=180).stdout


def video_hash(path):
    return command(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
                    '-c:v', 'copy', '-f', 'hash', '-hash', 'sha256', '-']).strip()


def ingest(source, destination):
    if platform.system() != 'Linux':
        raise RuntimeError('Media ingestion runs on the qualified GPU worker, not the Mac')
    command(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    state_path = source / 'state.json'
    state = json.loads(state_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / 'ingest-receipt.json'
    rows = []
    for key, job in state['jobs'].items():
        if job.get('status') != 'complete':
            continue
        raw = source / job['output']
        if raw.parent != source or sha(raw) != job['sha256']:
            raise RuntimeError('Raw file path/hash mismatch: ' + key)
        if Path(key).name != key:
            raise RuntimeError('Invalid job ID')
        silent = destination / (key + '.mp4')
        if not silent.exists():
            command(['ffmpeg', '-v', 'error', '-n', '-i', str(raw), '-map',
                     '0:v:0', '-c:v', 'copy', '-an', str(silent)])
        raw_hash = video_hash(raw)
        if video_hash(silent) != raw_hash:
            raise RuntimeError('Silent picture payload differs: ' + key)
        probe = json.loads(command(['ffprobe', '-v', 'error', '-count_frames',
                                    '-show_streams', '-of', 'json', str(silent)]))
        streams = probe['streams']
        if len(streams) != 1 or streams[0]['codec_type'] != 'video':
            raise RuntimeError('Silent candidate contains unexpected streams')
        frames = int(streams[0]['nb_read_frames'])
        endpoint = destination / (key + '-endpoint.png')
        if not endpoint.exists():
            command(['ffmpeg', '-v', 'error', '-n', '-i', str(silent), '-vf',
                     f'select=eq(n\\,{frames-1})', '-frames:v', '1', str(endpoint)])
        rows.append(dict(id=key, rawSha256=job['sha256'],
                         silent=silent.name, silentSha256=sha(silent),
                         videoPayloadHash=raw_hash, frames=frames,
                         endpoint=endpoint.name, endpointSha256=sha(endpoint),
                         audioStreams=0, creativeReview='pending'))
    receipt = dict(sourceStateSha256=sha(state_path), items=rows,
                   policy='H3 picture only; raw retained separately as provenance')
    text = json.dumps(receipt, indent=2) + '\n'
    if receipt_path.exists() and receipt_path.read_text() != text:
        raise RuntimeError('Existing ingestion receipt differs; use a versioned destination')
    if not receipt_path.exists():
        with receipt_path.open('x') as stream:
            stream.write(text)
    print(json.dumps(dict(ingested=len(rows), receipt=str(receipt_path))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    ingest(args.source.resolve(), args.destination.resolve())
