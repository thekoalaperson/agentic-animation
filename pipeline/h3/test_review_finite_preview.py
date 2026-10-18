#!/usr/bin/env python3
"""The H3 sheet step runs apart from ingest, and its preview never admits."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import review_finite

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OrdinalTests(unittest.TestCase):
    def test_every_twelfth_frame_and_the_last(self):
        self.assertEqual(review_finite.sheet_ordinals(29), [0, 12, 24, 28])
        self.assertEqual(review_finite.sheet_ordinals(25), [0, 12, 24])

    def test_worker_review_keeps_its_gate(self):
        with mock.patch.object(review_finite.platform, "system", return_value="Darwin"), \
                self.assertRaisesRegex(RuntimeError, "qualified GPU worker"):
            review_finite.main(Path("/nonexistent"))


@unittest.skipUnless(FFMPEG, "ffmpeg is not installed")
class PreviewSheetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root / "inputs").mkdir()
        (self.root / "outputs").mkdir()
        raw = self.root / "outputs/demo-s001.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=160x90:r=24:d=1.25",
                        "-pix_fmt", "yuv420p", str(raw)], check=True)
        manifest = self.root / "inputs/manifest.json"
        manifest.write_text(json.dumps({"items": [
            {"id": "demo-s001", "sourceSha256": "a" * 64, "expectedFrames": 30},
            {"id": "demo-s002", "sourceSha256": "b" * 64, "expectedFrames": 30}]}))
        (self.root / "outputs/state.json").write_text(json.dumps({
            "manifestSha256": digest(manifest),
            "jobs": {"demo-s001": {"status": "complete", "output": "demo-s001.mp4",
                                   "sha256": digest(raw), "sourceSha256": "a" * 64}}}))
        self.raw = raw

    def tearDown(self):
        self.temp.cleanup()

    def test_preview_draws_labelled_sheets_and_admits_nothing(self):
        with mock.patch("sys.stdout"):
            audit = review_finite.preview_sheets(self.root)
        self.assertFalse((self.root / "local-audit.json").exists())
        self.assertFalse((self.root / "silent").exists())
        self.assertFalse((self.root / "review").exists())
        sheet = self.root / "review-preview-not-final/demo-s001-2fps-preview-not-final.jpg"
        self.assertTrue(sheet.is_file())
        self.assertIs(audit["finalApproved"], False)
        self.assertEqual(audit["kind"], "h3-review-sheets-preview-not-final")
        first, second = audit["outputs"]
        self.assertEqual((first["frames"], first["frameOrdinals"]), (30, [0, 12, 24, 29]))
        self.assertEqual(first["sheetSha256"], digest(sheet))
        self.assertEqual(second, {"id": "demo-s002", "status": "not-complete"})
        saved = json.loads((self.root / "review-preview-not-final/preview-sheet-audit.json").read_text())
        self.assertEqual(saved["outputs"], audit["outputs"])

    def test_changed_raw_output_is_refused(self):
        with self.raw.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Raw output or source changed"):
            review_finite.preview_sheets(self.root)


if __name__ == "__main__":
    unittest.main()
