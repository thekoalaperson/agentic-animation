import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pipeline import observe_media_finite
from pipeline import prepare_dialogue_review as review
from pipeline.prepare_dialogue_review import select_jobs

SCRIPT = Path(review.__file__)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demo_project(root: Path, layout="private/demo-project"):
    """A tiny dialogue batch: one 2 s clip, speech-like tone from 0.6 s to 1.4 s."""
    project = root / layout
    work = project / "work/grok-dialogue-v1"
    (work / "prompts").mkdir(parents=True)
    (project / "art").mkdir()
    still = project / "art/demo-s001.png"
    still.write_bytes(b"demo still")
    prompt = work / "prompts/demo-s001.txt"
    prompt.write_text("demo-s001: a lamp on a table\n", encoding="utf-8")
    video = work / "demo-s001.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x36:r=24:d=2",
                    "-f", "lavfi", "-i",
                    "aevalsrc='if(between(t,0.6,1.4),0.5*sin(2*PI*300*t),0)':s=48000:d=2",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)],
                   check=True)
    manifest = work / "execution-manifest.json"
    manifest.write_text(json.dumps({"episode": "DEMO", "jobs": [{
        "id": "demo-s001", "speaker": "demo-teller", "hindi": "पंक्ति।",
        "source": "art/demo-s001.png", "sourceSha256": digest(still),
        "prompt": "work/grok-dialogue-v1/prompts/demo-s001.txt", "promptSha256": digest(prompt),
        "outputStem": "work/grok-dialogue-v1/demo-s001"}]}), encoding="utf-8")
    video.with_suffix(".vid.json").write_text(json.dumps({
        "outputSha256": digest(video), "sourceSha256": digest(still),
        "executionManifestSha256": digest(manifest)}), encoding="utf-8")
    return project, manifest, video


class SpeechWindowTest(unittest.TestCase):
    def test_window_measurement_runs_offline_in_any_project_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, manifest, video = demo_project(Path(tmp))
            done = subprocess.run([sys.executable, str(SCRIPT), str(manifest), "--project",
                                   str(project), "--max-jobs", "1", "--measure-speech-windows"],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            doc = json.loads((manifest.parent / "speech-windows.json").read_text())
            item = doc["items"][0]
            self.assertEqual(doc["manifestSha256"], digest(manifest))
            self.assertEqual(item["sourceSha256"], digest(video))
            self.assertFalse(item["fallbackToFullClip"])
            self.assertAlmostEqual(item["detected"]["start"], 0.6, delta=0.06)
            self.assertAlmostEqual(item["detected"]["end"], 1.4, delta=0.06)
            # The minimum handle widens, never narrows, what was heard.
            self.assertAlmostEqual(item["speechWindow"]["start"], item["detected"]["start"] - 0.25, places=5)
            self.assertAlmostEqual(item["speechWindow"]["end"], item["detected"]["end"] + 0.25, places=5)
            self.assertEqual(doc["admission"], "not implied")

    def test_changed_clip_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, manifest, video = demo_project(Path(tmp))
            with video.open("ab") as handle:
                handle.write(b"x")
            done = subprocess.run([sys.executable, str(SCRIPT), str(manifest), "--project",
                                   str(project), "--max-jobs", "1", "--measure-speech-windows"],
                                  capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("provenance changed", done.stderr)

    def test_window_rules(self):
        # Leading and trailing silence trimmed, then the handle is added back.
        self.assertEqual(review.speech_window(4.0, [(0.0, 1.0), (3.0, None)], 0.25),
                         ({"start": 1.0, "end": 3.0}, {"start": 0.75, "end": 3.25}, False))
        # A transcript span outside the detected window widens it.
        self.assertEqual(review.speech_window(4.0, [(0.0, 1.0)], 0.0, {"start": 0.4, "end": 2.0})[1],
                         {"start": 0.4, "end": 4.0})
        # Nothing heard: the whole clip, flagged.
        self.assertEqual(review.speech_window(2.0, [(0.0, 2.0)], 0.25),
                         (None, {"start": 0.0, "end": 2.0}, True))


class ObservationManifestTest(unittest.TestCase):
    def test_sheet_plan_has_the_shape_the_observation_lane_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, manifest, _ = demo_project(Path(tmp))
            here = manifest.parent
            sheet = here / "dense-review/demo-s001-2fps.jpg"
            sheet.parent.mkdir()
            sheet.write_bytes(b"demo sheet")
            prompt = project / "story/observe.txt"
            prompt.parent.mkdir()
            prompt.write_text("Describe what is visible in each frame.\n", encoding="utf-8")
            plan = json.loads(manifest.read_text())
            doc = review.observation_manifest(
                here, project, plan, [{"id": "demo-s001", "sheet": "dense-review/demo-s001-2fps.jpg",
                                       "sheetSha256": digest(sheet)}], prompt, 5)
            target = here / "sheet-observation-manifest.json"
            target.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            dry = observe_media_finite.run(target, 1, False)
            self.assertEqual((dry["jobs"], dry["networkCalls"], dry["financialAuthorized"]),
                             (1, 0, False))


class SubsetReviewTest(unittest.TestCase):
    def setUp(self):
        self.plan = {"jobs": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}

    def test_full_default(self):
        self.assertEqual(select_jobs(self.plan, None, 3), self.plan["jobs"])

    def test_explicit_subset_keeps_source_order_and_original(self):
        self.assertEqual(select_jobs(self.plan, "c,a", 2), [{"id": "a"}, {"id": "c"}])
        self.assertEqual(len(self.plan["jobs"]), 3)

    def test_fail_closed(self):
        for ids, count in [("a,a", 2), ("a,z", 2), ("a,b", 3), (None, 2), ("", 0)]:
            with self.subTest(ids=ids, count=count), self.assertRaises(ValueError):
                select_jobs(self.plan, ids, count)


if __name__ == "__main__":
    unittest.main()
