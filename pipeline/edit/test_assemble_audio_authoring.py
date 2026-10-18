"""Authored audio choices in the long-form assembler: silence, voice gain, music shaping."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.edit import assemble_longform as assembly


class AudioAuthoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.ep = Path(self.temporary.name) / "demo-film"
        self.work = self.ep / "work/assembly-v1"
        self.work.mkdir(parents=True)
        self.written = {}

    def tearDown(self):
        self.temporary.cleanup()

    def capture(self, path, samples):
        self.written[Path(path).name] = samples.copy()
        Path(path).write_bytes(b"captured")

    def events(self, kinds=("narration", "dialogue", "narration", "narration")):
        return [{"shot": i + 1, "id": f"demo-s{i + 1:03d}", "kind": kind,
                 "start": float(i), "end": float(i + 1), "duration": 1.0}
                for i, kind in enumerate(kinds)]

    def wave(self):
        return {"shots": [{"shot": i + 1, "music_energy": float(i)} for i in range(4)]}

    def test_silence_placements_decode_nothing_and_keep_one_conform_row_each(self):
        plan = {"runtimeSeconds": 4.0, "events": self.events(), "musicBreaths": [],
                "scorePlacements": [{"movement": "hush", "silence": True, "timelineStart": 0.0,
                                     "timelineDuration": 4.0, "reason": "The film opens unscored."}]}
        with patch.object(assembly, "SR", 24), patch.object(assembly, "write_audio", self.capture), \
                patch.object(assembly, "audio", side_effect=AssertionError("nothing to decode")):
            assembly.build_music(self.ep, self.work, plan, self.wave())
        self.assertFalse(np.any(self.written["music-wave-48k24.wav"]))
        conform = json.loads((self.work / "music-conform.json").read_text())
        self.assertEqual(conform, [{"movement": "hush", "silence": True, "timelineStart": 0.0,
                                    "timelineDuration": 4.0, "reason": "The film opens unscored.",
                                    "repriseJoins": []}])
        # A silence places no take, so the wave audit records no fit for it.
        audit = json.loads((self.work / "music-wave-audit.json").read_text())
        self.assertEqual(audit["scoreFits"], [])

    def shaped(self, shaping=None, breaths=()):
        plan = {"runtimeSeconds": 4.0, "events": self.events(), "musicBreaths": list(breaths)}
        if shaping is not None:
            plan["musicShaping"] = shaping
        with patch.object(assembly, "SR", 24):
            return assembly.wave_envelope(plan, self.wave(), 96)

    def test_absent_or_default_shaping_reproduces_todays_envelope(self):
        absent, absent_audit = self.shaped()
        explicit, explicit_audit = self.shaped({"mode": "wave", **assembly.MUSIC_SHAPING_DEFAULTS})
        np.testing.assert_array_equal(absent, explicit)
        self.assertNotIn("musicShaping", absent_audit)
        self.assertEqual({k: v for k, v in explicit_audit.items() if k != "musicShaping"}, absent_audit)
        # Today's literal constants, so a default can never drift from them.
        raw = [0.14 + 0.82 * (i / 3) ** 1.55 for i in range(4)]
        raw[1] *= 0.78
        self.assertEqual([row["envelopeTarget"] for row in absent_audit["eventTargets"]], raw)

    def test_authored_shape_and_per_event_override_change_only_what_they_name(self):
        _, audit = self.shaped({"floor": 0.3, "range": 0.6, "dialogueScale": 1.0, "smoothingSeconds": 0,
                                "events": {"demo-s004": {"exponent": 1.0, "reason": "Let the end rise evenly."}}})
        targets = [row["envelopeTarget"] for row in audit["eventTargets"]]
        self.assertAlmostEqual(targets[0], 0.3)
        self.assertAlmostEqual(targets[1], 0.3 + 0.6 * (1 / 3) ** 1.55)  # no dip under the line
        self.assertAlmostEqual(targets[3], 0.9)
        self.assertEqual(audit["musicShaping"]["events"]["demo-s004"]["reason"], "Let the end rise evenly.")

    def test_take_dynamics_needs_no_wave_and_keeps_breaths(self):
        plan = {"runtimeSeconds": 4.0, "events": self.events(), "scorePlacements": [],
                "musicShaping": {"mode": "take-dynamics"},
                "musicBreaths": [{"shots": [3, 3], "level": 0, "reason": "A held breath."}]}
        with patch.object(assembly, "SR", 24):
            env, audit = assembly.wave_envelope(plan, None, 96)
        self.assertTrue(np.all(env[48:72] == 0))
        self.assertAlmostEqual(float(env[10]), 1.0)
        self.assertTrue(audit["method"].startswith("take dynamics"))
        self.assertEqual([row["shot"] for row in audit["eventTargets"]], [1, 2, 3, 4])
        with patch.object(assembly, "SR", 24), patch.object(assembly, "write_audio", self.capture):
            assembly.build_music(self.ep, self.work, plan, None)
            with self.assertRaisesRegex(RuntimeError, "dramatic wave"):
                assembly.build_music(self.ep, self.work, {**plan, "musicShaping": {"mode": "wave"}}, None)

    def test_shaping_refusals_name_the_value(self):
        ids = [e["id"] for e in self.events()]
        cases = {
            "unknown": {"ceiling": 1},
            "wave or take-dynamics": {"mode": "loud"},
            "exceed 1.0": {"floor": 0.5, "range": 0.6},
            "exponent": {"exponent": 9},
            "names no event": {"events": {"demo-s099": {"floor": 0.2, "reason": "x"}}},
            "needs a reason": {"events": {"demo-s001": {"floor": 0.2}}},
            "may set": {"events": {"demo-s001": {"smoothingSeconds": 1, "reason": "x"}}},
        }
        for message, shaping in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                assembly.resolve_music_shaping(shaping, ids)

    def voice_levels(self, samples, **event_extra):
        source = self.ep / "work/narration/demo-s001.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"demo narration")
        event = {"id": "demo-s001", "shot": 1, "kind": "narration", "start": 0.0, "end": 1.0,
                 "duration": 1.0, "audio": "work/narration/demo-s001.wav",
                 "audioSha256": assembly.sha(source), **event_extra}
        plan = {"runtimeSeconds": 1.0, "events": [event]}
        with patch.object(assembly, "SR", 100), patch.object(assembly, "write_audio", self.capture), \
                patch.object(assembly, "audio", return_value=samples.copy()):
            assembly.build_voice(self.ep, self.work, plan)
        return json.loads((self.work / "voice-levels.json").read_text())[0], self.written["voice-hi-48k24.wav"]

    def test_voice_gain_applies_after_the_shared_target_and_keeps_the_peak_ceiling(self):
        whisper = np.full((100, 2), 0.01, np.float32)
        plain, plain_stem = self.voice_levels(whisper)
        self.assertEqual(set(plain), {"event", "rms", "peak", "gain"})
        quieter, quieter_stem = self.voice_levels(whisper, voiceGainDb=-12)
        self.assertAlmostEqual(quieter["gain"], plain["gain"] * 10 ** (-12 / 20), places=6)
        self.assertFalse(quieter["authoredGainLimitedByPeak"])
        self.assertLess(float(np.max(quieter_stem)), float(np.max(plain_stem)))
        shout = np.full((100, 2), 0.05, np.float32)
        shout[50] = 0.9  # already near the ceiling, so +6 dB can only reach the cap
        capped, stem = self.voice_levels(shout, voiceGainDb=6)
        self.assertTrue(capped["authoredGainLimitedByPeak"])
        self.assertLessEqual(float(np.max(np.abs(stem))), 0.92 + 1e-6)
        with self.assertRaisesRegex(ValueError, "-12..6"):
            self.voice_levels(whisper, voiceGainDb=9)


if __name__ == "__main__":
    unittest.main()
