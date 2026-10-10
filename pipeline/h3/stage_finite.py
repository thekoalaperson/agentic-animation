"""Stage a hash-bound authored H3 plan without any cloud or model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(plan_path, target):
    plan = json.loads(plan_path.read_text())
    root = Path(plan.pop("sourceRoot")).resolve()
    target.mkdir(parents=True, exist_ok=True)
    (target / "stills").mkdir(exist_ok=True)
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
    path = target / "manifest.json"
    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == text, "Immutable staged manifest changed"
    else:
        path.write_text(text)
    print(json.dumps(dict(manifest=str(path), jobs=len(plan["items"]),
                         sha256=sha(path), deadlineAt=plan["deadlineAt"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("target", type=Path)
    args = ap.parse_args()
    stage(args.plan.resolve(), args.target.resolve())

