"""Keep a working watch distinct from delivery admission."""
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

import mix_working_context_review as working


def write_test_wav(path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(np.rint(samples * 32768).astype("<i2").tobytes())


class WorkingReviewTests(unittest.TestCase):
    def test_local_protection_is_bounded_and_covers_edge_syllables(self):
        plan = {"runtimeSeconds": 20, "events": [{"id": "s1", "start": 2, "end": 10}]}
        def audit(level, kind="narration"):
            return {"intelligibility": {"events": [{"event": "s1", "kind": kind, "p10VoiceToBackgroundDb": level}]}}
        row = working.speech_protection_notches(plan, audit(-5.5))[0]
        self.assertEqual(row["scoreGainDb"], -8)
        self.assertEqual(row["startSeconds"], 1.85)
        self.assertEqual(row["endSeconds"], 10.22)
        self.assertEqual(working.speech_protection_notches(plan, audit(4)), [])
        self.assertEqual(working.speech_protection_notches(plan, audit(-5, "dialogue")), [])

    def test_music_presence_does_not_change_sfx_or_default_final_contract(self):
        baseline = working.mix.default_settings()
        richer = working.working_settings("music-presence-v2")
        self.assertEqual(richer["musicGainDb"], 4.0)
        self.assertEqual(richer["musicDuck"]["ratio"], 3.0)
        self.assertEqual(richer["sfxDuck"], baseline["sfxDuck"])
        self.assertEqual(working.mix.default_settings(), baseline)
        self.assertEqual(working.working_settings("conservative-v1"), baseline)
        with self.assertRaises(working.mix.MixError):
            working.working_settings("anything")

    def test_output_is_work_only_and_not_canonical_final(self):
        with tempfile.TemporaryDirectory() as raw:
            episode = Path(raw).resolve()
            expected = episode / "work/render-review-v1"
            self.assertEqual(working.preview_output(episode, "work/render-review-v1"), expected)
            for bad in ("deliverables", "work", "work/assembly-v1", "work/final-delivery-v1", "../outside"):
                with self.assertRaises(working.mix.MixError):
                    working.preview_output(episode, bad)

    def test_final_or_wrong_language_inventory_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            for inv in ({"kind": "stable-audio-absolute-sfx-final-mix-input"},
                        {"kind": "demo-absolute-sfx-context-review-inventory", "episode": "DEMO", "language": "en"}):
                with self.assertRaises(working.mix.MixError):
                    working.source_paths(Path(raw), inv)

    def test_hard_silence_measures_buses_and_premaster_not_only_metadata(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = {key: root / f"{key}.wav" for key in
                     ("voiceStem", "selectedScoreStem", "contextPreviewSfxStem")}
            voice = np.ones((4800, 2), dtype="float32") * .125
            silence = np.zeros_like(voice)
            write_test_wav(paths["voiceStem"], voice)
            for key in ("selectedScoreStem", "contextPreviewSfxStem"):
                write_test_wav(paths[key], silence)
            premaster = root / "mix.wav"
            write_test_wav(premaster, voice)
            design = {"hardSfxSilenceWindows": [{"sceneId": "s110", "start": 0, "end": .1}]}
            result = working.check_vacuums(paths, premaster, design)
            self.assertEqual(result[0]["mixMinusVoicePeak"], 0)
            write_test_wav(premaster, voice + .001)
            with self.assertRaises(working.mix.MixError):
                working.check_vacuums(paths, premaster, design)
            write_test_wav(premaster, voice)
            write_test_wav(paths["selectedScoreStem"], silence + .001)
            with self.assertRaises(working.mix.MixError):
                working.check_vacuums(paths, premaster, design)


if __name__ == "__main__":
    unittest.main()
