"""Synthetic contract tests; no production media, FFmpeg, network or cloud use.

Run only this module, not the older command-line calibration script named
test_designed_sfx.py. All decode/save/process entry points are replaced with
small synthetic fixtures when exercising render().
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "render_designed_sfx_under_test", Path(__file__).with_name("render_designed_sfx.py")
)
sfx = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sfx)

REJECTED = (RuntimeError, ValueError, AssertionError, KeyError, TypeError)


class DesignedSfxContractTests(unittest.TestCase):
    def test_nested_eight_axis_emotions_are_not_lost(self):
        values = {key: index * 10 for index, key in enumerate(
            ('awe', 'dread', 'devotion', 'grief', 'hope', 'scale', 'kinetic', 'intimacy'))}
        self.assertEqual(sfx.emotion_values({'shot': 1, 'emotions': values}), values)
        self.assertEqual(sfx.emotion_values({'kineticEnergy': 20})['kinetic'], 20)
        self.assertEqual(sfx.emotion_values({'emotions': {'kineticEnergy': 30}})['kinetic'], 30)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.ep = self.base / "episode"
        self.ep.mkdir()
        self.root = self.ep / "work/designed-sfx"
        self.root.mkdir(parents=True)
        self.clock = {
            "episode": "TEST",
            "kind": "measured-speech-clock-not-renderable-edit-plan",
            "renderable": False,
            "runtimeSeconds": 10,
            "events": [
                {"shot": 1, "start": 0, "duration": 2, "end": 2},
                {"shot": 2, "start": 2, "duration": 8, "end": 10},
            ],
        }
        self.wave = {
            "episode": "TEST",
            "shots": [{"shot": n, "awe": 30, "kineticEnergy": 20} for n in (1, 2)],
        }
        self.design = {
            "version": 2,
            "episode": "TEST",
            "authority": "Synthetic test authority only",
            "generationApiAllowed": False,
            "workingAudioOnly": True,
            "finalApproved": False,
            "sourceBatches": {},
            "cues": [self.cue()],
        }
        self.refresh_clock()
        self.add_batch("a", b"SYNTHETIC SOURCE A")
        self.add_batch("b", b"SYNTHETIC SOURCE B")

    def write(self, relative, data):
        path = self.ep / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
        return {"path": relative, "sha256": sfx.sha(path)}

    def refresh_clock(self):
        self.design["clock"] = self.write("work/clock.json", self.clock)
        self.wave["clockSha256"] = self.design["clock"]["sha256"]
        self.design["wave"] = self.write("work/wave.json", self.wave)

    def add_batch(self, key, payload):
        relative = f"work/palette-{key}"
        path = self.ep / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "source.bin").write_bytes(payload)
        brief = {
            "model": "stabilityai/stable-audio-3-small-sfx",
            "generationApiAllowed": False,
            "jobs": [{"id": "shared-id", "output": "source.bin", "seconds": 1,
                      "prompt": f"Synthetic sound {key}"}],
        }
        brief_spec = self.write(relative + "/probe-brief.json", brief)
        audit = {"technicalPass": True, "briefSha256": brief_spec["sha256"],
                 "outputs": [{"id": "shared-id", "sha256": sfx.sha(path / "source.bin"),
                              "verified": True}]}
        self.design["sourceBatches"][key] = {
            "brief": brief_spec,
            "audit": self.write(relative + "/local-audit.json", audit),
        }

    def rewrite_batch(self, key, *, brief=None, audit=None):
        spec = self.design["sourceBatches"][key]
        if brief is not None:
            spec["brief"] = self.write(spec["brief"]["path"], brief)
        if audit is None:
            audit = sfx.read(self.ep / spec["audit"]["path"])
        audit["briefSha256"] = spec["brief"]["sha256"]
        spec["audit"] = self.write(spec["audit"]["path"], audit)

    @staticmethod
    def cue(**changes):
        result = {
            "id": "cue-a", "sourceBatch": "a", "job": "shared-id", "shot": 1,
            "offset": 0.25, "semitones": 0, "speed": 1, "maxSeconds": 1,
            "activeRmsDb": -24, "fadeIn": 0.05, "fadeOut": 0.1,
        }
        result.update(changes)
        return result

    def load(self):
        return sfx.load_contract(self.ep, copy.deepcopy(self.design))

    def make_final(self):
        self.clock["kind"] = "renderable-edit-plan"
        self.clock["renderable"] = True
        self.refresh_clock()
        self.design["workingAudioOnly"] = False
        self.design["picture"] = self.write("work/picture.json", [
            {"id": "p1", "timelineStart": 0, "duration": 2, "sceneId": "test-s001"},
            {"id": "p2", "timelineStart": 2, "duration": 8, "sceneId": "test-s002"},
        ])

    def render_synthetic(self, *, during_decode=None, output_root=None):
        root = output_root or self.root
        root.mkdir(parents=True, exist_ok=True)
        (root / "design.json").write_text(json.dumps(self.design) + "\n")
        decoded = []
        saved = []

        def fake_decode(path, filters=None):
            decoded.append(Path(path))
            if during_decode:
                during_decode()
            # Exactly one second. Arrays are synthetic and never read from media.
            return np.full((sfx.SR, 2), 0.02, dtype=np.float32)

        def fake_save(path, samples):
            saved.append(Path(path))
            Path(path).write_bytes(b"SYNTHETIC OUTPUT ONLY")

        def fake_run(command, **kwargs):
            self.assertEqual(list(command), ["ffmpeg", "-version"])
            return SimpleNamespace(stdout="ffmpeg SYNTHETIC-NOT-EXECUTED\n")

        with mock.patch.object(sfx.sys, "platform", "linux"), \
                mock.patch.object(sfx.shutil, "which", return_value="/synthetic/nvidia-smi"), \
                mock.patch.object(sfx, "decode", side_effect=fake_decode), \
                mock.patch.object(sfx, "save", side_effect=fake_save), \
                mock.patch.object(sfx, "run", side_effect=fake_run), \
                mock.patch.object(sfx.subprocess, "run", side_effect=AssertionError("External process forbidden in unit test")), \
                mock.patch("builtins.print"):
            sfx.render(self.ep, root)
        return sfx.read(root / "mix-audit.json"), decoded, saved

    def test_scene_offset_maps_to_current_clock(self):
        self.design["cues"] = [self.cue(shot=2, offset=0.75)]
        _, plan, picture, _, contract = self.load()
        working = copy.deepcopy(self.design)
        sfx.load_contract(self.ep, working)
        self.assertEqual(working["cues"][0]["at"], 2.75)
        self.assertEqual(plan["runtimeSeconds"], 10)
        self.assertEqual(picture, [])
        self.assertTrue(contract["workingAudioOnly"])
        self.assertFalse(contract["finalApproved"])
        self.assertIsNone(contract["picturePlanSha256"])

    def test_same_job_id_in_two_palettes_keeps_exact_hash_and_prompt(self):
        self.design["cues"].append(self.cue(id="cue-b", sourceBatch="b", shot=2))
        result, decoded, _ = self.render_synthetic()
        self.assertEqual([p.parent.name for p in decoded], ["palette-a", "palette-b"])
        self.assertNotEqual(result["cues"][0]["sourceSha256"], result["cues"][1]["sourceSha256"])
        self.assertEqual([c["sourcePrompt"] for c in result["cues"]], ["Synthetic sound a", "Synthetic sound b"])
        self.assertEqual([c["at"] for c in result["cues"]], [0.25, 2.25])

    def test_stale_absolute_start_is_refused(self):
        self.design["cues"][0]["at"] = 500
        with self.assertRaises(REJECTED):
            self.load()

    def test_outside_scene_offset_is_refused(self):
        for offset in (-0.1, 2, float("nan"), float("inf")):
            with self.subTest(offset=offset):
                self.design["cues"][0]["offset"] = offset
                with self.assertRaises(REJECTED):
                    self.load()

    def test_hash_changes_in_clock_wave_brief_or_audit_are_refused(self):
        specs = [self.design["clock"], self.design["wave"],
                 self.design["sourceBatches"]["a"]["brief"],
                 self.design["sourceBatches"]["b"]["audit"]]
        for spec in specs:
            path = self.ep / spec["path"]
            original = path.read_bytes()
            with self.subTest(path=spec["path"]):
                try:
                    path.write_bytes(original + b" ")
                    with self.assertRaises(REJECTED):
                        self.load()
                finally:
                    path.write_bytes(original)

    def test_stale_wave_clock_reference_is_refused_even_when_wave_hash_matches(self):
        self.wave["clockSha256"] = "0" * 64
        self.design["wave"] = self.write("work/wave.json", self.wave)
        with self.assertRaises(REJECTED):
            self.load()

    def test_cross_palette_brief_audit_swap_is_refused(self):
        self.design["sourceBatches"]["a"]["audit"] = self.design["sourceBatches"]["b"]["audit"]
        with self.assertRaises(REJECTED):
            self.load()

    def test_changed_source_bytes_are_refused_before_decode(self):
        (self.ep / "work/palette-a/source.bin").write_bytes(b"CHANGED")
        with self.assertRaises(REJECTED):
            self.render_synthetic()

    def test_paid_generation_brief_is_refused(self):
        brief = sfx.read(self.ep / self.design["sourceBatches"]["a"]["brief"]["path"])
        brief["generationApiAllowed"] = True
        self.rewrite_batch("a", brief=brief)
        with self.assertRaises(REJECTED):
            self.load()

    def test_bound_file_escape_and_symlink_escape_are_refused(self):
        outside = self.base / "outside.json"
        outside.write_text("{}")
        link = self.ep / "link.json"
        link.symlink_to(outside)
        for path in ("../outside.json", "link.json", str(outside)):
            with self.subTest(path=path), self.assertRaises(REJECTED):
                sfx.bound_file(self.ep, {"path": path, "sha256": sfx.sha(outside)})

    def test_working_clock_cannot_claim_final_approval(self):
        self.design["finalApproved"] = True
        with self.assertRaises(REJECTED):
            self.load()

    def test_working_mode_requires_nonrenderable_measured_kind(self):
        self.clock["kind"] = "renderable-edit-plan"
        self.refresh_clock()
        with self.assertRaises(REJECTED):
            self.load()

    def test_final_mode_rejects_explicit_nonrenderable_clock(self):
        self.design["workingAudioOnly"] = False
        with self.assertRaises(REJECTED):
            self.load()

    def test_valid_final_plan_with_hash_bound_picture_is_supported(self):
        self.make_final()
        _, _, picture, _, contract = self.load()
        self.assertEqual(len(picture), 2)
        self.assertFalse(contract["workingAudioOnly"])
        self.assertEqual(contract["picturePlanSha256"], self.design["picture"]["sha256"])

    def test_explicit_cross_scene_tail_requires_reason(self):
        self.design["cues"][0].update(offset=1.5, maxSeconds=1)
        with self.assertRaises(REJECTED):
            self.load()
        self.design["cues"][0]["tailReason"] = "Continuous wind deliberately bridges the cut."
        self.load()

    def test_missing_max_seconds_does_not_silently_authorize_cross_scene_tail(self):
        self.design["cues"][0].update(offset=1.5)
        self.design["cues"][0].pop("maxSeconds")
        with self.assertRaises(REJECTED):
            self.render_synthetic()

    def test_unknown_contract_version_is_not_treated_as_legacy(self):
        legacy = self.make_legacy()
        legacy["version"] = 3
        with self.assertRaises(REJECTED):
            sfx.load_contract(self.ep, legacy)

    def test_measured_clock_cannot_be_promoted_by_deleting_renderable_flag(self):
        self.make_final()
        self.clock["kind"] = "measured-speech-clock-not-renderable-edit-plan"
        self.clock.pop("renderable")
        self.refresh_clock()
        self.design["finalApproved"] = True
        with self.assertRaises(REJECTED):
            self.load()

    def test_final_picture_cannot_be_empty(self):
        self.make_final()
        self.design["picture"] = self.write("work/picture.json", [])
        with self.assertRaises(REJECTED):
            self.load()

    def test_final_picture_cannot_come_from_an_unrelated_clock_range(self):
        self.make_final()
        self.design["picture"] = self.write("work/picture.json", [
            {"id": "unrelated", "timelineStart": 100, "duration": 10},
        ])
        with self.assertRaises(REJECTED):
            self.load()

    def test_duplicate_scene_events_are_refused_not_last_one_wins(self):
        # The duplicate has enough room for this cue: failure must not merely
        # come from the unrelated cross-scene-tail check.
        self.clock["events"].append({"shot": 1, "start": 4, "duration": 4, "end": 8})
        self.refresh_clock()
        with self.assertRaises(REJECTED):
            self.load()

    def test_nonfinite_scene_start_is_refused_at_contract_load(self):
        self.clock["events"][0]["start"] = float("nan")
        self.refresh_clock()
        with self.assertRaises(REJECTED):
            self.load()

    def test_duplicate_job_ids_inside_one_brief_are_refused(self):
        spec = self.design["sourceBatches"]["a"]
        brief = sfx.read(self.ep / spec["brief"]["path"])
        brief["jobs"].append(dict(brief["jobs"][0], prompt="Ambiguous duplicate"))
        self.rewrite_batch("a", brief=brief)
        with self.assertRaises(REJECTED):
            self.load()

    def test_unsafe_cue_id_cannot_escape_edited_directory(self):
        self.design["cues"][0]["id"] = "../escaped"
        with self.assertRaises(REJECTED):
            self.render_synthetic()
        self.assertFalse((self.root / "escaped.wav").exists())

    def test_working_only_result_cannot_be_written_to_deliverables(self):
        with self.assertRaises(REJECTED):
            self.render_synthetic(output_root=self.ep / "deliverables/sfx")

    def test_changed_design_during_render_cannot_receive_a_new_hash_as_if_used(self):
        def mutate():
            path = self.root / "design.json"
            changed = sfx.read(path)
            changed["cues"][0]["activeRmsDb"] = -80
            path.write_text(json.dumps(changed))

        with self.assertRaises(REJECTED):
            self.render_synthetic(during_decode=mutate)

    def test_changed_clock_during_render_is_refused_before_receipt(self):
        def mutate():
            path = self.ep / self.design["clock"]["path"]
            changed = sfx.read(path)
            changed["events"][1]["start"] = 7
            path.write_text(json.dumps(changed))

        with self.assertRaises(REJECTED):
            self.render_synthetic(during_decode=mutate)

    def test_nonpositive_or_nonfinite_max_seconds_are_refused_before_decode(self):
        for maximum in (0, -1, float("nan"), float("inf")):
            with self.subTest(maximum=maximum):
                self.design["cues"][0].update(maxSeconds=maximum, tailReason="Authored bridge")
                with self.assertRaises(REJECTED):
                    self.load()

    def test_blank_cross_scene_reason_is_not_an_editorial_reason(self):
        self.design["cues"][0].update(offset=1.5, maxSeconds=1, tailReason="   ")
        with self.assertRaises(REJECTED):
            self.load()

    def test_selected_output_explicitly_marked_unverified_is_refused(self):
        spec = self.design["sourceBatches"]["a"]
        audit = sfx.read(self.ep / spec["audit"]["path"])
        audit["outputs"][0]["verified"] = False
        self.rewrite_batch("a", audit=audit)
        with self.assertRaises(REJECTED):
            self.render_synthetic()

    def test_receipt_cannot_invent_stable_audio_route_for_another_model(self):
        spec = self.design["sourceBatches"]["a"]
        brief = sfx.read(self.ep / spec["brief"]["path"])
        brief["model"] = "synthetic-other-open-audio-model"
        self.rewrite_batch("a", brief=brief)
        try:
            result, _, _ = self.render_synthetic()
        except REJECTED:
            return  # Rejecting an unsupported model is also an honest outcome.
        self.assertNotIn("StableAudio3", result["route"])

    def test_working_receipt_does_not_claim_a_human_or_final_watch(self):
        result, _, _ = self.render_synthetic()
        self.assertFalse(result["finalApproved"])
        self.assertTrue(result["workingAudioOnly"])
        self.assertEqual(result["cues"][0]["picture"], [])
        self.assertIn("final mixed review pending", result["review"])
        self.assertIn("Not individual human take verdicts", result["review"])

    def test_gpu_host_guard_runs_before_read_or_decode(self):
        with mock.patch.object(sfx.sys, "platform", "darwin"), \
                mock.patch.object(sfx, "read") as read, \
                mock.patch.object(sfx, "decode") as decode:
            with self.assertRaises(RuntimeError):
                sfx.render(self.ep, self.root)
            read.assert_not_called()
            decode.assert_not_called()

    def make_legacy(self):
        spec = self.design["sourceBatches"]["a"]
        self.write("work/assembly-v1/edit-plan.json", {"episode": "TEST", "runtimeSeconds": 10})
        self.write("work/assembly-v1/picture-segments.json", [{"id": "p1"}])
        self.write("work/dramatic-wave-v1/dramatic-wave.json", self.wave)
        return {"version": 1, "sourceBatch": "work/palette-a", "briefSha256": spec["brief"]["sha256"]}

    def test_v1_load_path_remains_compatible(self):
        legacy = self.make_legacy()
        batches, plan, picture, _, contract = sfx.load_contract(self.ep, legacy)
        self.assertEqual(set(batches), {"legacy"})
        self.assertEqual(plan["runtimeSeconds"], 10)
        self.assertEqual(picture, [{"id": "p1"}])
        self.assertIn("sourceBriefSha256", contract)


if __name__ == "__main__":
    unittest.main()
