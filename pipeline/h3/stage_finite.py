"""Stage a hash-bound authored H3 plan without any cloud or model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import image_dimensions

if not __debug__:
    raise SystemExit("Optimized Python disables production safety gates")


class StageError(ValueError):
    """The authored plan cannot be staged without breaking its source binding."""


# H3 renders 17n+5 frames. The cited vLLM page now returns 404; the rule is
# corroborated by every frame count this repository has recorded and by
# third-party notes. A plan that breaks it would halt only after the GPU had
# rendered the take, so staging refuses it first.
H3_FRAME_STEP = 17
H3_FRAME_OFFSET = 5


def frame_rule_problem(frames):
    """Why ``frames`` cannot be an H3 expectedFrames value, or None."""
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= H3_FRAME_OFFSET:
        return f"expectedFrames must be a whole number above {H3_FRAME_OFFSET}, not {frames!r}"
    if (frames - H3_FRAME_OFFSET) % H3_FRAME_STEP:
        below = frames - (frames - H3_FRAME_OFFSET) % H3_FRAME_STEP
        return (f"expectedFrames {frames} is not 17n+5; the nearest H3 lengths are "
                f"{below} and {below + H3_FRAME_STEP}")
    return None


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
    for item in plan["items"]:
        if "expectedFrames" in item:
            problem = frame_rule_problem(item["expectedFrames"])
            if problem:
                raise StageError(f"{item.get('id')}: {problem}")
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
            if not source.is_relative_to(root):
                raise StageError(f"{item.get('id')}: {source_field} escapes sourceRoot: {source}")
            destination = target / "stills" / source.name
            digest = sha(source)
            if destination.exists():
                if sha(destination) != digest:
                    raise StageError(f"{item.get('id')}: staged still name collision at {destination}")
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
        if path.read_text() != text:
            raise StageError(f"immutable staged manifest changed: {path}")
    else:
        path.write_text(text)
    summary = dict(manifest=str(path), jobs=len(plan["items"]),
                   sha256=sha(path), deadlineAt=plan["deadlineAt"])
    if warnings:
        summary["warnings"] = warnings
    print(json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan", type=Path)
    ap.add_argument("target", type=Path)
    args = ap.parse_args()
    try:
        stage(args.plan.resolve(), args.target.resolve())
    except StageError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc

