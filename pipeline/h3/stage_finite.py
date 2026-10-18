"""Stage a hash-bound authored H3 plan without any cloud or model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import image_dimensions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aspect_warning(item, field, source, tolerance=0.03):
    """Warn, never refuse, when a source's aspect differs from the planned output."""
    try:
        _, width, height = image_dimensions(source)
    except (OSError, ValueError):
        return None
    out_w, out_h = item.get("expectedWidth", 1344), item.get("expectedHeight", 768)
    if abs(width / height - out_w / out_h) <= tolerance:
        return None
    return dict(type="source-aspect", item=item.get("id"), field=field,
                source=[width, height], plannedOutput=[out_w, out_h],
                hint="pipeline/derive_still.py can crop, pad or scale the still first")


def stage(plan_path, target):
    plan = json.loads(plan_path.read_text())
    root = Path(plan.pop("sourceRoot")).resolve()
    target.mkdir(parents=True, exist_ok=True)
    (target / "stills").mkdir(exist_ok=True)
    warnings = []
    for item in plan["items"]:
        for source_field, name_field, hash_field in [
            ("source", "still", "sourceSha256"),
            ("lastSource", "last_frame_still", "lastFrameSha256")
        ]:
            if source_field not in item:
                continue
            source = (root / item[source_field]).resolve()
            assert source.is_relative_to(root)
            destination = target / "stills" / source.name
            digest = sha(source)
            if destination.exists():
                assert sha(destination) == digest, "Staged source name collision"
            else:
                shutil.copy2(source, destination)
            item[name_field] = source.name
            item[hash_field] = digest
            warning = aspect_warning(item, source_field, source)
            if warning:
                warnings.append(warning)
    path = target / "manifest.json"
    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == text, "Immutable staged manifest changed"
    else:
        path.write_text(text)
    summary = dict(manifest=str(path), jobs=len(plan["items"]),
                   sha256=sha(path), deadlineAt=plan["deadlineAt"])
    if warnings:
        summary["warnings"] = warnings
    print(json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("target", type=Path)
    args = ap.parse_args()
    stage(args.plan.resolve(), args.target.resolve())

