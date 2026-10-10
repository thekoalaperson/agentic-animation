import unittest
from unittest.mock import patch
import numpy as np
from pipeline.edit import assemble_longform as music


class ExplicitScoreSections(unittest.TestCase):
    def setUp(self):
        self.source = np.arange(100, dtype=np.float32).reshape(50, 2)
        self.row = dict(sourceStart=1, sourceDuration=2, timelineDuration=2,
                        fadeInSeconds=0, fadeOutSeconds=0)

    def test_uses_exact_requested_recording_window(self):
        with patch.object(music, 'SR', 10), patch.object(music, 'extend_score') as auto:
            out, joins, receipt = music.conform_score_section(self.source, self.row)
        np.testing.assert_array_equal(out, self.source[10:30])
        auto.assert_not_called()
        self.assertEqual(joins, [])
        self.assertEqual(receipt['sourceStartSample'], 10)

    def test_never_pads_or_loops_a_short_explicit_cut(self):
        for changed in [dict(sourceStart=4), dict(timelineDuration=3), dict(sourceStart=-1),
                        dict(sourceDuration=float('nan')), dict(gainDb=7),
                        dict(fadeInSeconds=1.5, fadeOutSeconds=1.5)]:
            with self.subTest(changed=changed), patch.object(music, 'SR', 10):
                with self.assertRaises(RuntimeError):
                    music.conform_score_section(self.source, self.row | changed)

    def test_partial_cut_contract_rejected(self):
        with patch.object(music, 'SR', 10), self.assertRaises(RuntimeError):
            music.conform_score_section(self.source, dict(sourceStart=1, timelineDuration=2))

    def test_gain_and_independent_fades_keep_source_immutable(self):
        original = self.source.copy()
        with patch.object(music, 'SR', 10):
            out, _, _ = music.conform_score_section(self.source, self.row | dict(gainDb=-6, fadeInSeconds=.5))
        np.testing.assert_array_equal(self.source, original)
        self.assertEqual(float(out[0,0]), 0)
        np.testing.assert_allclose(out[-1], original[29] * 10**(-6/20), rtol=1e-6)

    def test_legacy_untrimmed_path_remains_available(self):
        with patch.object(music, 'SR', 10), patch.object(music, 'extend_score', return_value=(self.source.copy(), [])) as auto:
            music.conform_score_section(self.source, dict(timelineDuration=5))
        auto.assert_called_once()

    def test_post_gain_bus_rejects_pcm_clipping(self):
        with patch.object(music, 'SR', 10):
            out, _, _ = music.conform_score_section(np.full((50,2), .63, np.float32), self.row | dict(gainDb=6))
        with self.assertRaises(RuntimeError):
            music.validate_score_bus(out * .96)

    def test_overlaps_and_nonfinite_bus_rejected(self):
        for values in [np.full((10,2), .6)+np.full((10,2), .6), np.array([[float('nan'),0]])]:
            with self.assertRaises(RuntimeError):
                music.validate_score_bus(values)

    def test_safe_bus_is_unchanged(self):
        values = np.array([[.9,-.2]],np.float32)
        original = values.copy()
        self.assertAlmostEqual(music.validate_score_bus(values), .9, places=6)
        np.testing.assert_array_equal(values, original)


if __name__ == '__main__':
    unittest.main()
