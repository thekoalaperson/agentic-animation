import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "project"
TIMELINE = REPO / "pipeline/edit/build_episode_timeline.py"


def load_timeline():
    spec = importlib.util.spec_from_file_location("example_project_timeline", TIMELINE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExampleProjectTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "animation-demo"
        shutil.copytree(EXAMPLE, self.project)
        self.scenes = [row["still_id"] for row in self.storyboard()["stills"]]

    def storyboard(self):
        return json.loads((EXAMPLE / "story/storyboard_plan.json").read_text(encoding="utf-8"))

    def write_timeline_inputs(self) -> Path:
        """Synthetic narration, picture and a config, made only in the temp copy."""
        project = self.project
        narration_manifest = project / "work/narration-hi-v1/manifest.json"
        write_json(narration_manifest, {"items": [
            {"id": f"{scene}-hi", "shotId": scene,
             "output": f"../../audio/narration-v1/{scene}-hi.mp3"}
            for scene in self.scenes]})
        state = {"manifestSha256": sha(narration_manifest), "items": {}}
        entries = {}
        for scene in self.scenes:
            audio = project / f"audio/narration-v1/{scene}-hi.mp3"
            picture = project / f"work/h3/{scene}.mp4"
            for path in (audio, picture):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{scene}{path.suffix}".encode())
            state["items"][f"{scene}-hi"] = {
                "status": "COMPLETED", "output": str(audio.resolve()),
                "sha256": sha(audio), "outputSha256": sha(audio)}
            entries[scene] = {"pictureRoute": "h3", "segments": [{
                "source": f"work/h3/{scene}.mp4", "in": 0, "out": 3,
                "role": "accepted-base", "verdict": "approved"}]}
        write_json(narration_manifest.with_suffix(".eleven-state.json"), state)
        write_json(project / "work/grok-dialogue-v1/execution-manifest.json", {"jobs": []})
        write_json(project / "work/h3-review-ledger-v1.json", {"entries": entries})
        config = project / "work/assembly-v1/timeline-inputs.json"
        write_json(config, {
            "version": 1,
            "storyboard": "story/storyboard_plan.json",
            "narrationManifest": "work/narration-hi-v1/manifest.json",
            "dialogueManifest": "work/grok-dialogue-v1/execution-manifest.json",
            "h3ReviewLedger": "work/h3-review-ledger-v1.json",
            "outputDirectory": "work/assembly-v1",
        })
        return config

    @staticmethod
    def fake_probe(timeline):
        def probe(path: Path):
            if path.suffix == ".mp3":
                return timeline.MediaProbe(2.0, None, 2.0, None, None, 0, 1)
            if path.suffix == ".mp4":
                return timeline.MediaProbe(3.0, 3.0, None, 72, 24.0, 1, 0)
            raise AssertionError(path)
        return probe

    def test_readme_compile_runs_on_a_copy_of_the_example(self):
        output = self.project / "work" / "script-v1"
        result = subprocess.run(
            [sys.executable, str(REPO / "pipeline/edit/compile_production_script.py"),
             str(self.project), str(self.project / "story/script.txt"), str(output)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        self.assertIsInstance(audit, dict)

    def test_storyboard_rows_carry_the_beat_shape_the_timeline_reads(self):
        for row in self.storyboard()["stills"]:
            self.assertIsInstance(row["beat"], dict, row["still_id"])
            self.assertIsInstance(row["beat"]["number"], int, row["still_id"])
            self.assertTrue(row["narration_hi"] and row["narration_en"], row["still_id"])

    def test_timeline_dry_run_runs_on_a_copy_of_the_example(self):
        self.write_timeline_inputs()
        result = subprocess.run(
            [sys.executable, str(TIMELINE), str(self.project), "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        readiness = json.loads(
            (self.project / "work/assembly-v1/timeline-readiness.json").read_text(encoding="utf-8"))
        self.assertTrue(readiness["dryRun"])
        self.assertFalse((self.project / "work/assembly-v1/edit-plan.json").exists())

    def test_measured_timeline_reads_the_example_beats(self):
        # The dry run returns before the event loop reads each row's beat, so
        # the full measured build runs here on synthetic media in the copy.
        timeline = load_timeline()
        config = self.write_timeline_inputs()
        ready, plan = timeline.build_timeline(
            self.project, config, probe=self.fake_probe(timeline))
        self.assertTrue(ready["ready"], ready)
        self.assertEqual([event["id"] for event in plan["events"]], self.scenes)
        self.assertEqual([event["narrationBreath"]["reason"] for event in plan["events"]],
                         ["beat-boundary", "beat-boundary"])

    def test_authority_stub_stays_one_line(self):
        lines = (EXAMPLE / "story/GOAL.md").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
