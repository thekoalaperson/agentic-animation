import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "project"


class ExampleProjectTest(unittest.TestCase):
    def test_readme_compile_runs_on_a_copy_of_the_example(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "animation-demo"
            shutil.copytree(EXAMPLE, project)
            output = project / "work" / "script-v1"
            result = subprocess.run(
                [sys.executable, str(REPO / "pipeline/edit/compile_production_script.py"),
                 str(project), str(project / "story/script.txt"), str(output)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertIsInstance(audit, dict)

    def test_authority_stub_stays_one_line(self):
        lines = (EXAMPLE / "story/GOAL.md").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
