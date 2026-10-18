#!/usr/bin/env python3
"""Aspect warnings before motion spend: dialogue dry run and H3 staging warn,
never refuse, when a still's header geometry is far from the planned output."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIALOGUE = load("dialogue_geometry_target", HERE / "kie_dialogue_finite.py")
STAGE = load("stage_geometry_target", HERE / "h3" / "stage_finite.py")


class GeometryWarningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def still(self, name, size):
        path = self.base / name
        Image.new("RGB", size, "gray").save(path)
        return path

    def test_dialogue_warns_only_when_far_from_sixteen_nine(self):
        rows = [dict(id="demo-s001", sourcePath=self.still("wide.png", (1280, 720))),
                dict(id="demo-s002", sourcePath=self.still("square.png", (512, 512))),
                dict(id="demo-s003", sourcePath=self.still("near.png", (1300, 720)))]
        unreadable = self.base / "unreadable.png"
        unreadable.write_bytes(b"not an image")
        rows.append(dict(id="demo-s004", sourcePath=unreadable))
        warnings = DIALOGUE.source_geometry_warnings(rows)
        self.assertEqual([w["jobId"] for w in warnings], ["demo-s002"])
        self.assertIn("derive_still.py", warnings[0]["hint"])
        # A version 3 manifest freezes its own ratio; the warning compares against that.
        tall = DIALOGUE.source_geometry_warnings(rows, "9:16")
        self.assertEqual([w["jobId"] for w in tall], ["demo-s001", "demo-s002", "demo-s003"])
        self.assertEqual({w["sentAspectRatio"] for w in tall}, {"9:16"})

    def test_warnings_do_not_enter_the_hashed_execution_request(self):
        plan = {"manifestSha256": "a" * 64, "model": "grok-imagine-video-1-5-preview",
                "resolution": "720p", "aspectRatio": "16:9",
                "geometryWarnings": [{"type": "source-aspect"}]}
        row = dict(id="demo-s001", sourceSha256="b" * 64, promptSha256="c" * 64,
                   durationSeconds=7, outputStem="motion/demo-s001", referenceSha256s=[])
        bare = {key: value for key, value in plan.items() if key != "geometryWarnings"}
        self.assertEqual(DIALOGUE.execution_request(plan, row), DIALOGUE.execution_request(bare, row))

    def test_stage_prints_warnings_without_changing_the_manifest(self):
        root = self.base / "project"
        root.mkdir()
        Image.new("RGB", (768, 1344), "gray").save(root / "tall.png")
        Image.new("RGB", (1344, 768), "gray").save(root / "wide.png")
        plan = dict(sourceRoot=str(root), deadlineAt=1, items=[
            dict(id="demo-a", source="tall.png"),
            dict(id="demo-b", source="wide.png"),
            dict(id="demo-c", source="tall.png", expectedWidth=768, expectedHeight=1344),
        ])
        plan_path = self.base / "plan.json"
        plan_path.write_text(json.dumps(plan))
        target = self.base / "staged"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            STAGE.stage(plan_path, target)
        summary = json.loads(buffer.getvalue())
        self.assertEqual([w["item"] for w in summary["warnings"]], ["demo-a"])
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertNotIn("warnings", manifest)
        self.assertEqual(set(manifest["items"][0]), {"id", "source", "still", "sourceSha256"})


if __name__ == "__main__":
    unittest.main()
