"""Metadata/command/synthetic-array tests only; never invoke media or GPU tools."""
import copy
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rebuild_delivery_audio as adapter


def fixture():
    events = [dict(id=f"demo-s{i + 1:03}", start=i, end=i + 1, duration=1,
                   subtitleAudioAt=0, subtitleAudioDuration=.8,
                   subtitleTextHi=f"पूर्ण वाक्य {i + 1}।", kind="narration") for i in range(3)]
    plan = dict(language="hi", deliveryFps=24, runtimeSeconds=3, events=events)
    spec = dict(kind="authored-motion-short", language="hi", fps=24,
                bodyFrames=48, ctaFrames=156, totalFrames=204, beats=[])
    for index, event in enumerate((events[0], events[2])):
        spec["beats"].append(dict(id=event["id"], sourceStartFrame=event["start"] * 24,
                                  startFrame=index * 24, frames=24,
                                  spokenTextHi=event["subtitleTextHi"], audioDuration=.8))
    return plan, spec


class ContractTests(unittest.TestCase):
    def config(self):
        return dict(schemaVersion=1, kind="source-bound-audio-only-delivery-revision", fps=24,
                    language="hi", backgroundDerivation="unity-voice-linear-premaster-minus-unchanged-voice",
                    outputRoot="work/audio-v1", inputs={k:dict(path=f"work/{k}", sha256="a" * 64) for k in adapter.INPUTS})

    def test_future_premaster_requires_exact_external_binding(self):
        c = self.config()
        c["inputs"]["bodyPremaster"]["sha256"] = None
        with self.assertRaisesRegex(ValueError, "exact SHA"):
            adapter.resolved_config(c)
        resolved = adapter.resolved_config(c, "b" * 64)
        self.assertEqual(resolved["inputs"]["bodyPremaster"]["sha256"], "b" * 64)
        self.assertIsNone(c["inputs"]["bodyPremaster"]["sha256"])

    def test_cannot_override_bound_input_or_accept_missing_extra_inputs(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            adapter.resolved_config(self.config(), "b" * 64)
        for key in ("missing", "extra", "invalidHash", "nonlinear"):
            c = self.config()
            if key == "missing":
                del c["inputs"]["voice"]
            elif key == "extra":
                c["inputs"]["unknown"] = dict(path="work/x", sha256="a" * 64)
            elif key == "invalidHash":
                c["inputs"]["old720"]["sha256"] = "not a hash"
            else:
                c["backgroundDerivation"] = "nonlinear-voice"
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.resolved_config(c)

    def test_paths_cannot_escape_or_target_delivery_and_broad_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp)
            for value in ("../escape", "/tmp/absolute", "deliverables/a.mp4", ".", "work"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    adapter.inside(ep, value, work=True)
            self.assertEqual(adapter.inside(ep, "work/new/output", work=True), (ep / "work/new/output").resolve())

    def test_symlink_cannot_escape_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp) / "episode"
            ep.mkdir()
            (ep / "work").mkdir()
            (ep / "work/escaped").symlink_to(Path(tmp), target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                adapter.inside(ep, "work/escaped/private-input.wav", work=True)

    def test_nonfinite_or_noninteger_clock_is_rejected(self):
        for value in (float("nan"), float("inf"), True, "1.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.frame(value, "test frame")
        for value in (1.5, True, 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.integer(value, "test ordinal")

    def test_complete_ordered_scenes_keep_both_clocks(self):
        p, s = fixture()
        self.assertEqual(adapter.validate_plans(p, s), dict(bodyFrames=72, fullFrames=228, shortFrames=204, ctaFrames=156))

    def test_rejects_gaps_omitted_opener_reorder_trim_text_and_timing_changes(self):
        for change in ("gap", "opener", "reverse", "source", "trim", "text", "duration", "fractional", "cta"):
            p, s = fixture()
            if change == "gap": p["events"][1]["start"] = 1.1
            if change == "opener": s["beats"][0]["id"] = "demo-s002"
            if change == "reverse": s["beats"].reverse()
            if change == "source": s["beats"][1]["sourceStartFrame"] = 47
            if change == "trim": s["beats"][1]["frames"] = 23
            if change == "text": s["beats"][1]["spokenTextHi"] = "बदला हुआ पाठ"
            if change == "duration": s["beats"][1]["audioDuration"] = .7
            if change == "fractional": s["beats"][1]["startFrame"] = 24.5
            if change == "cta": s["ctaFrames"] = 155
            with self.subTest(change=change), self.assertRaises(ValueError):
                adapter.validate_plans(p, s)

    def test_no_processing_on_mac(self):
        with patch.object(adapter.platform, "system", return_value="Darwin"), patch.object(adapter.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "never Mac"):
                adapter.execute(Path("/unused"), Path("/unused/config"), {})
            run.assert_not_called()

    def test_gpu_requires_nonempty_device(self):
        with patch.object(adapter.platform, "system", return_value="Linux"), patch.object(adapter.subprocess, "run") as run:
            run.return_value.stdout = ""
            with self.assertRaisesRegex(ValueError, "No GPU"):
                adapter.require_gpu()


class PacketTests(unittest.TestCase):
    def test_priming_is_measured_from_packets_not_zero_stream_start(self):
        old = [dict(pts=0, dts=0, duration=1024), dict(pts=1024, dts=1024, duration=976)]
        encoded = [dict(pts=-1024, dts=-1024, duration=1024), dict(pts=0, dts=0, duration=976)]
        a = adapter.packet_interval(old, "1/48000")
        b = adapter.packet_interval(encoded, "1/48000")
        shift = a[0] - b[0]
        self.assertEqual(shift, Fraction(1024, 48000))
        self.assertEqual(a[1], b[1] + shift)

    def test_full_aac_retains_fractional_millisecond_pcm_endpoint(self):
        cmd = adapter.encode_full_aac("full.wav", "new.m4a", "loudnorm")
        self.assertEqual(cmd[cmd.index("-af") + 1], "loudnorm,aresample=48000,asetpts=N/SR/TB")
        self.assertEqual(cmd[cmd.index("-movie_timescale") + 1], "48000")
        # Representative packet ends: an old track and a rounded intermediate encode.
        old_end, rounded_end, priming, pcm_samples = 58911024, 58909968, 1024, 58910000
        self.assertEqual(old_end - (rounded_end + priming), 32)
        self.assertEqual(pcm_samples + priming, old_end)
        for flag in ("-t", "-shortest", "-frames:a"):
            self.assertNotIn(flag, cmd)

    def test_copy_mux_never_reencodes_picture_or_uses_shortest_or_duration_cut(self):
        cmd = adapter.clock_preserving_mux("old.mp4", "new.m4a", "out.mp4", "1/12288", "1/48000", Fraction(1024, 48000))
        self.assertIn("-copyts", cmd)
        self.assertEqual(cmd[cmd.index("-c") + 1], "copy")
        self.assertEqual(cmd[cmd.index("-itsoffset") + 1], "0.021333333333")
        self.assertEqual(cmd[cmd.index("-video_track_timescale") + 1], "12288")
        self.assertEqual(cmd[cmd.index("-movie_timescale") + 1], "1536000")
        for flag in ("-shortest", "-t", "-frames:v", "-y", "-vf"):
            self.assertNotIn(flag, cmd)

    def test_rejects_malformed_packet_clock(self):
        for rows in ([], [dict(pts=0, dts=0, duration=0)], [dict(pts=0, duration=1024)]):
            with self.assertRaises(ValueError):
                adapter.packet_interval(rows, "1/48000")


class SyntheticSampleTests(unittest.TestCase):
    def test_only_background_fades_voice_and_cta_samples_stay_exact(self):
        import numpy as np
        _, spec = fixture()
        voice = np.arange(72000 * 2, dtype=np.float32).reshape((-1, 2)) / 1000000
        voice = np.tile(voice, (2, 1))
        premix = voice + np.float32(.3)
        output, dry = adapter.short_audio(voice, premix, spec)
        for row in adapter.short_slices(spec):
            lo, at, n = row["sourceStartSample"], row["outputStartSample"], row["samples"]
            np.testing.assert_array_equal(dry[at:at+n], voice[lo:lo+n])
            np.testing.assert_array_equal(output[at], voice[lo])
            np.testing.assert_array_equal(output[at+n-1], voice[lo+n-1])
        np.testing.assert_array_equal(output[96000:], 0)
        np.testing.assert_array_equal(dry[96000:], 0)
        np.testing.assert_allclose(output[10000], voice[10000] + .3, atol=1e-7)

    def test_unbroken_source_join_has_no_artificial_duck(self):
        _, s = fixture()
        s["beats"][1]["sourceStartFrame"] = 24
        rows = adapter.short_slices(s)
        self.assertEqual(rows[0]["fadeOutSamples"], 0)
        self.assertEqual(rows[1]["fadeInSamples"], 0)
        self.assertEqual(rows[-1]["fadeOutSamples"], 28800)


if __name__ == "__main__":
    unittest.main()
