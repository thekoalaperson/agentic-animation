#!/usr/bin/env python3
"""Source-bound dialogue sheets and blind speech-observation jobs.

No paid API calls. A sheet is evidence for review, never an acceptance record.
Without --preview this runs on the qualified GPU worker. --preview draws the
same sheets anywhere ffmpeg runs, names them *-preview-not-final, records the
real host and ffmpeg build, and writes no observation manifest. When the local
ffmpeg has no drawtext filter or no font is available, timestamps are not
burned in; the frame ordinals and times stay in each proof record.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common

WORKER_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
PREVIEW = "-preview-not-final"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Immutable review input changed: " + str(path))
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def select_jobs(plan, selected_ids, expected):
    """An explicit completed subset retains the original execution binding."""
    rows = plan["jobs"]
    known = {row["id"] for row in rows}
    if len(known) != len(rows):
        raise ValueError("Duplicate job IDs in execution manifest")
    if selected_ids is not None:
        requested = selected_ids.split(",")
        if len(set(requested)) != len(requested) or not set(requested) <= known:
            raise ValueError("Selected IDs are duplicate or absent from execution manifest")
        rows = [row for row in rows if row["id"] in requested]
    if len(rows) != expected or not 1 <= expected <= 160:
        raise ValueError("Exact bounded job count required")
    return rows


def timestamp_filter(font):
    """The drawtext stage that burns each frame's time, or '' without a font."""
    if font is None:
        return ""
    if any(c in str(font) for c in "':,[]\\;"):
        raise ValueError("Unsafe font path for an ffmpeg filter: " + str(font))
    return ("drawtext=fontfile=" + str(font) + ":"
            "text='%{pts\\:hms}':fontsize=17:fontcolor=white:box=1:boxcolor=black@0.75:x=4:y=4,")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--max-jobs", type=int, required=True)
    ap.add_argument("--ids", help="Explicit comma-separated completed subset; never changes source manifest")
    ap.add_argument("--preview", action="store_true",
                    help="Draw sheets anywhere ffmpeg runs; outputs are labelled preview-not-final")
    ap.add_argument("--font", type=Path, help="Timestamp font; otherwise the profile fonts.review, then the worker DejaVuSans")
    a = ap.parse_args(argv)
    host = None
    if a.preview:
        font = _common.review_font(a.font) or (WORKER_FONT if WORKER_FONT.is_file() else None)
        if "drawtext" not in _common.ffmpeg_capabilities()["filters"]:
            font = None
        host = _common.preview_host_record(font)
    else:
        if platform.system() != "Linux":
            raise SystemExit("Heavy media preparation requires the qualified GPU host")
        subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                       capture_output=True, check=True, timeout=30)
        font = _common.review_font(a.font) or WORKER_FONT
    suffix = PREVIEW if a.preview else ""
    root = a.repo_root.resolve()
    manifest = a.manifest.resolve()
    here = manifest.parent
    here.relative_to(root / "film")
    plan = json.loads(manifest.read_text())
    selected = select_jobs(plan, a.ids, a.max_jobs)
    review = here / "dense-review"
    review.mkdir(exist_ok=True)
    jobs = []
    for row in selected:
        source = (root / row["source"]).resolve()
        source.relative_to(root / "film")
        prompt = (root / row["prompt"]).resolve()
        prompt.relative_to(here)
        video = (root / row["outputStem"]).with_suffix(".mp4").resolve()
        video.relative_to(here)
        meta_path = video.with_suffix(".vid.json")
        meta = json.loads(meta_path.read_text())
        digest = sha(video)
        if (sha(source) != row["sourceSha256"] or sha(prompt) != row["promptSha256"]
                or meta.get("outputSha256") != digest
                or meta.get("sourceSha256") != row["sourceSha256"]
                or meta.get("executionManifestSha256") != sha(manifest)):
            raise ValueError("Source/output provenance changed: " + row["id"])
        jobs.append((row, video, digest, meta_path))

    def one(job):
        row, video, digest, meta_path = job
        probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames",
            "-show_streams", "-show_format", "-of", "json", str(video)],
            check=True, capture_output=True, text=True, timeout=120).stdout)
        v = next(s for s in probe["streams"] if s["codec_type"] == "video")
        if v["r_frame_rate"] != "24/1":
            raise ValueError("Review sampling requires known 24fps input")
        count = int(v["nb_read_frames"])
        frames = sorted(set(range(0, count, 12)) | {count - 1})
        if len(frames) > 36:
            raise ValueError("Review sheet would omit frames; increase its explicit grid")
        sheet = review / (row["id"] + "-2fps" + suffix + ".jpg")
        proof = sheet.with_suffix(".json")
        entry = dict(id=row["id"], source=str(video.relative_to(here)),
                     sourceSha256=digest, sidecarSha256=sha(meta_path),
                     frameOrdinals=frames, timestamps=[round(n/24, 6) for n in frames],
                     sheet=str(sheet.relative_to(here)), speaker=row["speaker"],
                     hindi=row["hindi"], probe=probe,
                     reviewScope="Every 12th frame plus final frame; not continuous playback or approval")
        if a.preview:
            entry.update(preview=True, finalApproved=False, burnedTimestamps=font is not None)
        if sheet.exists() != proof.exists():
            raise ValueError("Incomplete review pair; reconcile before overwriting")
        if sheet.exists():
            old = json.loads(proof.read_text())
            if old.get("sourceSha256") != digest or old.get("sheetSha256") != sha(sheet):
                raise ValueError("Existing review evidence changed")
            return old
        rows = (len(frames) + 4) // 5
        vf = ("select='not(mod(n,12))+eq(n," + str(count-1) + ")',scale=384:216,"
              + timestamp_filter(font) + "tile=5x" + str(rows))
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(video), "-vf", vf,
                        "-frames:v", "1", "-q:v", "2", str(sheet)], check=True, timeout=120)
        entry["sheetSha256"] = sha(sheet)
        save(proof, entry)
        return entry

    with ThreadPoolExecutor(max_workers=3) as pool:
        evidence = list(pool.map(one, jobs))
    if a.preview:
        _common.atomic_json(here / ("dense-review" + PREVIEW + ".json"), {
            "manifestSha256": sha(manifest), "episode": plan["episode"], "items": evidence,
            "acceptance": "not implied", "kind": "dialogue-sheets-preview-not-final", "hostRecord": host})
        print(json.dumps({"jobs": len(jobs), "sheets": len(evidence), "networkCalls": 0, "preview": True}))
        return
    save(here / "dense-review-index.json", {"manifestSha256": sha(manifest),
        "episode": plan["episode"], "items": evidence, "acceptance": "not implied"})
    blind = {"episode": plan["episode"], "purpose": "Blind native-Hindi completeness observation; advisory only",
        "jobs": [{"id": row["id"], "mode": "speech", "file": str(video.relative_to(here)),
                  "sha256": digest} for row, video, digest, _ in jobs]}
    save(here / "speech-observation-manifest.json", blind)
    print(json.dumps({"jobs": len(jobs), "sheets": len(evidence), "networkCalls": 0}))


if __name__ == "__main__":
    main()
