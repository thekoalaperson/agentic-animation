"""Synthetic seam-contract regressions; no media, Pillow, subprocess or cloud use.

All images, decoded arrays, probes and command execution are fake. These tests
exercise source/selection metadata and the reported boundary, not perceptual QA.
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


class SyntheticSheet:
    def paste(self, *_args):
        pass

    def save(self, path, **_kwargs):
        Path(path).write_bytes(b"SYNTHETIC GRID, NOT MEDIA")


fake_pil = SimpleNamespace(
    Image=SimpleNamespace(new=lambda *_args: SyntheticSheet(),
                          fromarray=lambda value: value),
    ImageDraw=SimpleNamespace(Draw=lambda _sheet: SimpleNamespace(text=lambda *_a, **_k: None)),
)
spec = importlib.util.spec_from_file_location(
    "continuation_seams_under_test", Path(__file__).with_name("audit_continuation_seams.py")
)
seams = importlib.util.module_from_spec(spec)
with mock.patch.dict("sys.modules", {"PIL": fake_pil}):
    spec.loader.exec_module(seams)


class ContinuationSeamContractTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ep = Path(tmp.name).resolve() / "episode"
        self.ep.mkdir()
        (self.ep / "work").mkdir()
        self.parent = self.ep / "work/parent.bin"
        self.child = self.ep / "work/child.bin"
        self.parent.write_bytes(b"SYNTHETIC PARENT")
        self.child.write_bytes(b"SYNTHETIC CHILD")
        self.review = self.ep / "work/review.json"
        self.out = self.ep / "work/seams"
        self.row = {
            "id": "synthetic-s001", "sceneId": "synthetic-s001",
            "verdict": "sampled-compatible-used-window-candidate",
            "sourceBinding": {
                "parentSilentSourcePath": "work/parent.bin",
                "parentSilentSourceSha256": seams.sha(self.parent),
                "parentAnchorFrame": 174,
                "silentPath": "work/child.bin",
                "silentSha256": seams.sha(self.child),
            },
            "plannedUse": {
                "firstFrame": 1, "lastFrameInclusive": 3, "frames": 3,
                "nativeFps": 24, "editorialSpeed": 1, "seconds": 3/24,
                "dropConditioningFrame0": True,
                "contiguousParentExtensionOnly": True,
            },
        }
        self.commands = []
        self.frame_calls = []

    def command(self, args):
        self.commands.append(args)
        if args[0] == "ffmpeg" and args[-1] != "-":
            Path(args[-1]).write_bytes(b"SYNTHETIC PREVIEW, NOT MEDIA")
        return b""

    def decoded(self, path, start, end):
        self.frame_calls.append((path, start, end))
        # Parent174 == child0; child1 is the next synthetic value.
        values = np.arange(start, end) + (174 if path == self.child else 0)
        return np.broadcast_to(values[:, None, None, None], (end-start, 2, 2, 3)).astype(np.uint8)

    def execute(self, rows=None, system="Linux"):
        self.review.write_text(json.dumps({"items": rows if rows is not None else [self.row]}))
        args = SimpleNamespace(episode=self.ep, review=self.review, output=self.out)
        with mock.patch.object(seams.argparse.ArgumentParser, "parse_args", return_value=args), \
                mock.patch.object(seams.platform, "system", return_value=system), \
                mock.patch.object(seams, "run", side_effect=self.command), \
                mock.patch.object(seams, "probe", return_value={
                    "width": 1344, "height": 768, "nb_frames": "175", "avg_frame_rate": "24/1"}), \
                mock.patch.object(seams, "frames", side_effect=self.decoded), \
                mock.patch("builtins.print"):
            seams.main()
        return json.loads((self.out / "seam-audit.json").read_text())

    def test_actual_join_drops_frame_zero_and_includes_three_used_frames(self):
        result = self.execute()
        self.assertEqual(self.frame_calls, [(self.parent, 163, 175), (self.child, 0, 4)])
        item = result["items"][0]
        self.assertEqual(item["conditioningFrameDifference"], 0)
        self.assertEqual(item["boundaryMeanAbsoluteRgbDifference"], 1)
        self.assertEqual(len(item["nativeAdjacentDifferences"]), 14)
        preview = next(cmd for cmd in self.commands if "-filter_complex" in cmd)
        graph = preview[preview.index("-filter_complex")+1]
        self.assertIn("trim=start_frame=1:end_frame=4", graph)
        self.assertFalse(result["creativeApproval"])
        self.assertFalse(result["continuousPlaybackPerformed"])

    def test_long_used_window_keeps_prefix_audit_bounded(self):
        self.row["plannedUse"].update(lastFrameInclusive=83, frames=83, seconds=83/24)
        self.execute()
        self.assertEqual(self.frame_calls[-1], (self.child, 0, 13))
        preview = next(cmd for cmd in self.commands if "-filter_complex" in cmd)
        self.assertIn("trim=start_frame=1:end_frame=25", preview[preview.index("-filter_complex")+1])

    def test_mac_is_refused_before_any_command_or_decode(self):
        with self.assertRaises((AssertionError, RuntimeError)):
            self.execute(system="Darwin")
        self.assertEqual(self.commands, [])
        self.assertEqual(self.frame_calls, [])

    def test_stale_source_hash_is_refused_before_decode(self):
        self.row["sourceBinding"]["parentSilentSourceSha256"] = "0" * 64
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            self.execute()
        self.assertFalse(any(cmd[0] == "ffmpeg" for cmd in self.commands))

    def test_parent_endpoint_must_be_actual_last_frame(self):
        self.row["sourceBinding"]["parentAnchorFrame"] = 173
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            self.execute()

    def test_inconsistent_declared_used_frame_count_is_refused(self):
        self.row["plannedUse"]["frames"] = 100
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            self.execute()

    def test_conflicting_frame_zero_policy_is_refused(self):
        self.row["plannedUse"]["dropConditioningFrame0"] = False
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            self.execute()

    def test_conflicting_native_fps_is_refused(self):
        self.row["plannedUse"]["nativeFps"] = 30
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            self.execute()

    def test_noncontiguous_policy_is_refused(self):
        self.row["plannedUse"]["contiguousParentExtensionOnly"] = False
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            self.execute()

    def test_duplicate_ids_are_refused_instead_of_overwriting_evidence(self):
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            self.execute([self.row, copy.deepcopy(self.row)])

    def test_full_decode_claim_has_fatal_decode_error_policy(self):
        self.execute()
        decodes = [cmd for cmd in self.commands if cmd[0] == "ffmpeg" and cmd[-1] == "-"]
        self.assertEqual(len(decodes), 2)
        for cmd in decodes:
            self.assertIn("-xerror", cmd, "Recoverable FFmpeg decode errors must not yield fullSourcesDecode=True")

    def test_tool_source_changed_during_run_cannot_receive_new_hash(self):
        tool_source = self.ep / "work/synthetic-tool.py"
        tool_source.write_bytes(b"SYNTHETIC LOADED CODE")
        command = self.command

        def mutate_tool_during_fake_decode(args):
            if args[0] == "ffmpeg" and args[-1] == "-":
                tool_source.write_bytes(b"DIFFERENT SYNTHETIC CODE")
            return command(args)

        with mock.patch.object(seams, "__file__", str(tool_source)), \
                mock.patch.object(self, "command", side_effect=mutate_tool_during_fake_decode):
            with self.assertRaises((AssertionError, RuntimeError, ValueError)):
                self.execute()


if __name__ == "__main__":
    unittest.main()
