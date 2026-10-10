"""Exact-pixel, placement-independent HD range cache helpers (no model work)."""
import hashlib
import json
from pathlib import Path
import re


PIXEL_CONTRACT = "ffmpeg-sequential-rgb24-source-pixels-v1"
SHA = re.compile(r"^[a-f0-9]{64}$")


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def range_layout(frames, chunk_frames, plan=None):
    if plan is None:
        return [{"id": f"c{i:04d}", "startFrame": start, "frames": min(chunk_frames, frames-start)}
                for i, start in enumerate(range(0, frames, chunk_frames))]
    if plan.get("totalFrames") != frames or plan.get("kind") != "source-bound-hd-picture-ranges":
        raise ValueError("Range plan must bind this exact full picture frame count")
    rows, cursor, ids = [], 0, set()
    for row in plan.get("ranges", []):
        ident, start, count = row.get("id"), row.get("startFrame"), row.get("frames")
        if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident)
                or ident in ids or type(start) is not int or type(count) is not int
                or start != cursor or not 0 < count <= 600):
            raise ValueError("HD ranges must be unique, bounded and exactly contiguous")
        ids.add(ident)
        rows.append(dict(row))
        cursor += count
    if not rows or cursor != frames:
        raise ValueError("HD ranges do not cover every exact source ordinal once")
    return rows


def pixel_contract(cache_contract):
    return {key: value for key, value in cache_contract.items() if key != "sourceVideoSha256"}


def reusable_key(row):
    sha = row.get("sourceRgbSha256", "")
    if not SHA.fullmatch(sha) or row.get("sourcePixelContract") != PIXEL_CONTRACT:
        return None
    return sha, row.get("frames")


def load_reuse_index(roots, contract):
    result = {}
    expected = pixel_contract(contract)
    for raw_root in roots:
        root = Path(raw_root).resolve()
        configuration_path = root / "configuration.json"
        configuration = json.loads(configuration_path.read_text())
        audit_path = root / "audit.json"
        if audit_path.exists() and json.loads(audit_path.read_text()).get("regionalSourceRepairs"):
            raise ValueError("Prior cache used regional replacements; original raw chunks are not reusable")
        exclusions_path = root / "reuse-exclusions.json"
        exclusions = set()
        if exclusions_path.exists():
            exclusions_doc = json.loads(exclusions_path.read_text())
            if exclusions_doc.get("sourceVideoSha256") != configuration.get("sourceVideoSha256"):
                raise ValueError("Reuse exclusions do not bind this source picture")
            exclusions = set(exclusions_doc["chunks"])
        if any(configuration.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Reuse reconstruction/model contract differs: {root}")
        latest = {}
        for state in sorted(root.glob("state*.jsonl")):
            for line in state.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # An interrupted final JSONL append is not completion.
                if row.get("time", 0) >= latest.get(row.get("chunk"), {}).get("time", 0):
                    latest[row.get("chunk")] = row
        for row in latest.values():
            key = reusable_key(row)
            ident = row.get("chunk", "")
            if (row.get("status") != "complete" or key is None or ident in exclusions
                    or not re.fullmatch(r"[A-Za-z0-9_-]+", ident)
                    or any(row.get(k) != v for k, v in expected.items())
                    or row.get("sourceVideoSha256") != configuration.get("sourceVideoSha256")):
                continue
            output = root / "chunks" / f"{ident}.mp4"
            if output.is_file() and digest_file(output) == row.get("sha256"):
                result.setdefault(key, {"path": output, "state": row,
                                        "configurationSha256": digest_file(configuration_path)})
    return result


def layout_fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
