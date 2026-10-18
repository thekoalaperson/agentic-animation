"""Tools whose gates are asserts refuse to run with asserts stripped."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

PIPELINE = Path(__file__).resolve().parents[1]
TOOLS = [
    "edit/audit_continuation_seams.py",
    "edit/derive_subtitle_pairs.py",
    "edit/prepare_episode_production.py",
    "open_media/build_stable_audio_review_brief.py",
]


class OptimizedPythonRefusedTest(unittest.TestCase):
    def test_each_tool_refuses_before_parsing_arguments(self):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            str(PIPELINE / part) for part in ("", "edit", "open_media")))
        for tool in TOOLS:
            with self.subTest(tool):
                done = subprocess.run([sys.executable, "-O", str(PIPELINE / tool), "--help"],
                                      capture_output=True, text=True, env=env, timeout=60)
                self.assertNotEqual(done.returncode, 0)
                self.assertIn("Optimized Python disables production safety gates", done.stderr)
                self.assertNotIn("usage:", done.stdout)


if __name__ == "__main__":
    unittest.main()
