"""Every command shows its --help without credentials, a profile or a GPU."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ("KIE_KEY", "ELEVENLABS_API_KEY", "AGENTIC_PROFILE", "AGENTIC_PROJECT_ROOT",
           "AGENTIC_ENV_FILE")


def entry_points() -> list[Path]:
    found = []
    for path in sorted((ROOT / "pipeline").rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        if "__name__" in path.read_text(encoding="utf-8") and "__main__" in path.read_text(encoding="utf-8"):
            found.append(path)
    return found


class CommandHelpTest(unittest.TestCase):
    def test_every_entry_point_answers_help(self):
        scripts = entry_points()
        self.assertGreater(len(scripts), 40)
        env = {key: value for key, value in os.environ.items() if key not in PRIVATE}
        env["PYTHONPATH"] = os.pathsep.join(
            str(ROOT / part) for part in (".", "pipeline", "pipeline/edit", "pipeline/h3",
                                          "pipeline/open_media"))
        with tempfile.TemporaryDirectory() as cwd:
            def run(script):
                try:
                    done = subprocess.run([sys.executable, str(script), "--help"], cwd=cwd,
                                          env=env, stdin=subprocess.DEVNULL,
                                          capture_output=True, text=True, timeout=120)
                except subprocess.TimeoutExpired:
                    return script, None, "timed out"
                return script, done.returncode, (done.stderr or done.stdout)[-300:]

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(run, scripts))
        failures = [f"{script.relative_to(ROOT)}: exit {code}: {tail}"
                    for script, code, tail in results if code != 0]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
