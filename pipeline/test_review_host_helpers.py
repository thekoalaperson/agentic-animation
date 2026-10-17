"""Which work needs the worker, and what a preview records about its host."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import _common


class WorkerGateTests(unittest.TestCase):
    def test_off_linux_is_refused_with_the_reason(self):
        with mock.patch.object(_common.sys, "platform", "darwin"):
            with self.assertRaisesRegex(_common.MediaWorkerRequired, "H3 render must run on the qualified GPU worker"):
                _common.require_media_worker("H3 render")

    def test_linux_without_a_gpu_is_refused(self):
        with mock.patch.object(_common.sys, "platform", "linux"), \
                mock.patch("shutil.which", return_value=None):
            with self.assertRaises(_common.MediaWorkerRequired):
                _common.require_media_worker("Upscale")

    def test_nvidia_smi_that_lists_no_gpu_is_refused(self):
        import subprocess
        failure = subprocess.CalledProcessError(9, ["nvidia-smi", "-L"])
        with mock.patch.object(_common.sys, "platform", "linux"), \
                mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch("subprocess.run", side_effect=failure):
            with self.assertRaisesRegex(_common.MediaWorkerRequired, "found no usable GPU"):
                _common.require_media_worker("Upscale")
        with mock.patch.object(_common.sys, "platform", "linux"), \
                mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "GPU 0: demo\n", "")):
            self.assertEqual(_common.require_media_worker("Upscale"), "GPU 0: demo")


class ReviewFontTests(unittest.TestCase):
    def test_explicit_then_profile_then_none(self):
        with tempfile.TemporaryDirectory() as directory:
            font = Path(directory) / "demo.ttf"
            font.write_bytes(b"demo font")
            profile = Path(directory) / "profile.json"
            profile.write_text(json.dumps({"fonts": {"review": str(font)}}))
            with mock.patch.dict(os.environ, {"AGENTIC_PROFILE": ""}):
                self.assertIsNone(_common.review_font())
                self.assertEqual(_common.review_font(font), font.resolve())
            with mock.patch.dict(os.environ, {"AGENTIC_PROFILE": str(profile)}):
                self.assertEqual(_common.review_font(), font.resolve())
            with self.assertRaises(ValueError):
                _common.review_font(Path(directory) / "missing.ttf")


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
class HostRecordTests(unittest.TestCase):
    def test_record_names_the_real_host_and_never_approves(self):
        with tempfile.TemporaryDirectory() as directory:
            font = Path(directory) / "demo.ttf"
            font.write_bytes(b"demo font")
            record = _common.preview_host_record(font)
        self.assertIs(record["finalApproved"], False)
        self.assertTrue(record["ffmpeg"].startswith("ffmpeg version"))
        self.assertEqual(record["platform"], _common.sys.platform)
        self.assertEqual(record["font"]["sha256"], hashlib.sha256(b"demo font").hexdigest())
        self.assertTrue(set(record["ffmpegShapingFilters"]) <= {"ass", "subtitles", "drawtext"})
        self.assertIn("scale", _common.ffmpeg_capabilities()["filters"])


if __name__ == "__main__":
    unittest.main()
