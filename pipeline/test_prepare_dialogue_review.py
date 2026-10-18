import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from pipeline import prepare_dialogue_review as review
from pipeline.prepare_dialogue_review import select_jobs

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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



class TimestampFilterTests(unittest.TestCase):
    def test_worker_filter_is_unchanged(self):
        self.assertEqual(review.timestamp_filter(review.WORKER_FONT),
                         "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                         "text='%{pts\\:hms}':fontsize=17:fontcolor=white:box=1:boxcolor=black@0.75:x=4:y=4,")

    def test_no_font_means_no_burned_timestamps(self):
        self.assertEqual(review.timestamp_filter(None), "")

    def test_unsafe_font_path_is_refused(self):
        with self.assertRaises(ValueError):
            review.timestamp_filter(Path("/fonts/a:b.ttf"))

    def test_worker_run_keeps_its_gate(self):
        with mock.patch.object(review.platform, "system", return_value="Darwin"), \
                self.assertRaisesRegex(SystemExit, "qualified GPU host"):
            review.main(["/nonexistent/manifest.json", "--repo-root", "/nonexistent", "--max-jobs", "1"])


@unittest.skipUnless(FFMPEG, "ffmpeg is not installed")
class PreviewSheetTests(unittest.TestCase):
    def test_preview_sheets_are_labelled_and_write_no_observation_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            here = root / "film/demo/dialogue"
            here.mkdir(parents=True)
            source = root / "film/demo/art/demo-s001.png"
            source.parent.mkdir(parents=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x90",
                            "-frames:v", "1", str(source)], check=True)
            prompt = here / "demo-s001.txt"
            prompt.write_text("demo-s001: a lamp on a table")
            video = here / "demo-s001.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=160x90:r=24:d=2",
                            "-pix_fmt", "yuv420p", str(video)], check=True)
            manifest = here / "execution-manifest.json"
            manifest.write_text(json.dumps({"episode": "demo", "jobs": [{
                "id": "demo-s001", "source": "film/demo/art/demo-s001.png", "sourceSha256": digest(source),
                "prompt": "film/demo/dialogue/demo-s001.txt", "promptSha256": digest(prompt),
                "outputStem": "film/demo/dialogue/demo-s001", "speaker": "demo-a", "hindi": "demo line"}]}))
            video.with_suffix(".vid.json").write_text(json.dumps({
                "outputSha256": digest(video), "sourceSha256": digest(source),
                "executionManifestSha256": digest(manifest)}))
            with mock.patch.object(review.platform, "system", return_value="Darwin"), mock.patch("sys.stdout"):
                review.main([str(manifest), "--repo-root", str(root), "--max-jobs", "1", "--preview"])
            self.assertTrue((here / "dense-review/demo-s001-2fps-preview-not-final.jpg").is_file())
            self.assertFalse((here / "dense-review/demo-s001-2fps.jpg").exists())
            self.assertFalse((here / "dense-review-index.json").exists())
            self.assertFalse((here / "speech-observation-manifest.json").exists())
            index = json.loads((here / "dense-review-preview-not-final.json").read_text())
            self.assertEqual(index["kind"], "dialogue-sheets-preview-not-final")
            self.assertIs(index["hostRecord"]["finalApproved"], False)
            entry = index["items"][0]
            self.assertIs(entry["finalApproved"], False)
            self.assertEqual(entry["frameOrdinals"], [0, 12, 24, 36, 47])
            self.assertIsInstance(entry["burnedTimestamps"], bool)


if __name__ == "__main__":
    unittest.main()
