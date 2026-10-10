"""Pure command tests; no media, cloud or device operation."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class MasterClockTests(unittest.TestCase):
    def command(self, enabled, movie_clock=True):
        fake = types.SimpleNamespace(body_mux_command=lambda p, a, o, f, d:
            ['ffmpeg', '-i', p, '-i', a, '-c:v', 'copy', '-af', f, '-t', str(d), o])
        spec = importlib.util.spec_from_file_location('isolated_master_clock',
            Path(__file__).with_name('master_review_audio.py'))
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'mix_absolute_sfx_longform': fake}):
            spec.loader.exec_module(module)
        return module.master_mux_command('picture.mp4', 'voice-mix.wav', 'result.mp4',
            'normalizer,qualified-limiter', 1208 / 24, enabled, movie_clock)

    def test_opt_in_uses_emitted_sample_clock_after_all_processing(self):
        command = self.command(True)
        self.assertEqual(command[command.index('-af') + 1],
            'normalizer,qualified-limiter,aformat=channel_layouts=stereo,aresample=48000,asetpts=N/SR/TB')
        self.assertEqual(command[command.index('-movie_timescale') + 1], '48000')
        self.assertEqual(command[command.index('-c:v') + 1], 'copy')
        self.assertEqual(command[-1], 'result.mp4')
        self.assertNotIn('-shortest', command)

    def test_legacy_configuration_is_unchanged(self):
        command = self.command(False)
        self.assertEqual(command[command.index('-af') + 1], 'normalizer,qualified-limiter')
        self.assertNotIn('-movie_timescale', command)

    def test_old_muxer_opt_out_keeps_audio_sample_clock_and_picture_copy(self):
        command=self.command(True,False)
        self.assertNotIn('-movie_timescale',command)
        self.assertTrue(command[command.index('-af')+1].endswith('aresample=48000,asetpts=N/SR/TB'))
        self.assertEqual(command[command.index('-c:v')+1],'copy')


if __name__ == '__main__':
    unittest.main()
