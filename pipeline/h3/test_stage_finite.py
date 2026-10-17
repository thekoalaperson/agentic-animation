"""H3 staging keeps its source binding even when Python runs optimized."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("stage_finite.py")


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


if __name__ == "__main__":
    unittest.main()
