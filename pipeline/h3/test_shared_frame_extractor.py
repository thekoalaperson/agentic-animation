#!/usr/bin/env python3
"""The deficit builder's exact-frame extractor now lives in _common and keeps
its worker rule and its refusal type."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _common
import build_measured_deficit_manifest as deficit


class SharedExtractorTests(unittest.TestCase):
    def test_off_worker_extraction_is_still_a_blocked_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(_common.sys, "platform", "darwin"):
                with self.assertRaisesRegex(deficit.DeficitPlanBlocked, "qualified GPU worker"):
                    deficit._default_extract_frame(Path(temporary) / "in.mp4", 5,
                                                   Path(temporary) / "out.png")


if __name__ == "__main__":
    unittest.main()
