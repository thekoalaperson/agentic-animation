"""H3 staging keeps its source binding even when Python runs optimized."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("stage_finite.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = load("stage_finite_target", SCRIPT)
CLIENT = load("finite_client_target", Path(__file__).parent / "box" / "finite_client.py")


class StageFiniteTest(unittest.TestCase):
    def escaping_plan(self, root: Path) -> Path:
        sources = root / "project"
        sources.mkdir()
        (root / "outside.png").write_bytes(b"not part of the project")
        plan = root / "plan.json"
        plan.write_text(json.dumps({
            "sourceRoot": str(sources), "deadlineAt": "2026-01-01T00:00:00Z",
            "items": [{"id": "demo-s001", "source": "../outside.png"}],
        }), encoding="utf-8")
        return plan

    def stage(self, root: Path, *flags):
        target = root / "staged"
        done = subprocess.run([sys.executable, *flags, str(SCRIPT), str(self.escaping_plan(root)),
                               str(target)], capture_output=True, text=True)
        copied = list((target / "stills").glob("*")) if (target / "stills").exists() else []
        return done, copied

    def test_escaping_source_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            done, copied = self.stage(Path(tmp))
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("demo-s001: source escapes sourceRoot", done.stderr)
        self.assertEqual(copied, [])

    def test_optimized_python_is_refused_before_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            done, copied = self.stage(Path(tmp), "-O")
            self.assertFalse((Path(tmp) / "staged").exists())
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("Optimized Python disables production safety gates", done.stderr)
        self.assertEqual(copied, [])

    def plan(self, root: Path, **item) -> Path:
        sources = root / "project"
        sources.mkdir(exist_ok=True)
        for name in ("demo-s001.png", "demo-s001-end.png"):
            (sources / name).write_bytes(name.encode())
        plan = root / "plan.json"
        plan.write_text(json.dumps({
            "sourceRoot": str(sources), "route": "owned-h100-h3", "generationApiAllowed": False,
            "deadlineAt": 4102444800,
            "items": [{"id": "demo-s001", "source": "demo-s001.png", "prompt": "a lamp on a table",
                       "seconds": 7, "short_edge": 768, "steps": 30, "seed": 1,
                       "expectedFrames": 175, **item}],
        }), encoding="utf-8")
        return plan

    def test_frame_counts_off_the_17n_plus_5_grid_are_refused_before_staging(self):
        for frames in (168, 176, "175", True):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaisesRegex(STAGE.StageError, "demo-s001: expectedFrames"):
                    STAGE.stage(self.plan(root, expectedFrames=frames), root / "staged")
                self.assertFalse((root / "staged").exists())
        self.assertIn("158 and 175", STAGE.frame_rule_problem(168))
        for frames in (73, 175, 277, 362):
            self.assertIsNone(STAGE.frame_rule_problem(frames))

    def test_last_source_stages_a_two_keyframe_item_the_client_accepts(self):
        # The client sends last_frame_still at frame_index -1. That body is
        # unexercised at every length until a one-job qualification run is recorded.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            STAGE.stage(self.plan(root, lastSource="demo-s001-end.png"), root / "staged")
            manifest = root / "staged" / "manifest.json"
            [item] = CLIENT.validate_manifest(manifest, 32)["items"]
            self.assertEqual(item["last_frame_still"], "demo-s001-end.png")
            self.assertEqual(item["lastFrameSha256"], STAGE.sha(root / "project/demo-s001-end.png"))
            self.assertEqual(item["still"], "demo-s001.png")


if __name__ == "__main__":
    unittest.main()
