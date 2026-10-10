"""Map already-measured picture runs to reusable HD ranges on the GPU worker."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from hd_range_cache import range_layout


def make_ranges(runs):
    """Keep shot identity while splitting long uninterrupted takes at600 frames."""
    rows, cursor, identities = [], 0, set()
    for run in runs:
        frames = run['frames']
        if (type(frames) is not int or frames <= 0
                or round(run['timelineStart'] * 24) != cursor
                or run['id'] in identities):
            raise ValueError('Picture runs must be unique and exactly contiguous')
        identities.add(run['id'])
        for offset in range(0, frames, 600):
            part = offset // 600
            rows.append({'id': run['id'] if frames <= 600 else f"{run['id']}-part{part:03d}",
                         'startFrame': cursor + offset, 'frames': min(600, frames-offset),
                         'sceneId': run.get('sceneId'),
                         'memberSegmentIds': run['memberSegmentIds'],
                         'parentRenderRunId': run['id'], 'parentRunOffsetFrames': offset})
        cursor += frames
    range_layout(cursor, 600, {'kind':'source-bound-hd-picture-ranges',
                             'totalFrames':cursor, 'ranges':rows})
    return rows, cursor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("render_runs", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Media probing/hashing stays on the GCP worker")
    runs = json.loads(a.render_runs.read_text())
    video_sha = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(a.source),
                                        "-map", "0:v:0", "-c", "copy", "-f", "hash",
                                        "-hash", "sha256", "-"], text=True).strip().split("=", 1)[1]
    rows, cursor = make_ranges(runs)
    doc = {"kind": "source-bound-hd-picture-ranges", "schemaVersion": 1,
           "sourceVideoSha256": video_sha, "totalFrames": cursor, "ranges": rows,
           "renderRunsSha256": hashlib.sha256(a.render_runs.read_bytes()).hexdigest()}
    range_layout(cursor, 600, doc)
    a.output.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"ranges": len(rows), "frames": cursor, "sourceVideoSha256": video_sha}), flush=True)


if __name__ == "__main__":
    main()
