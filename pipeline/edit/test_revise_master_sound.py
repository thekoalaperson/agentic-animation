"""The master revision's loudness choices and optional score echo."""
import shutil
import subprocess
import unittest

import mix_absolute_sfx_longform as longform
import revise_master_sound as revise


class ReviseMasterSoundTest(unittest.TestCase):
    def test_defaults_keep_todays_echo_and_targets(self):
        self.assertEqual(revise.music_space({}),
                         "asplit=2[md][ms];[ms]aecho=0.8:0.45:80|160|320:0.10|0.06|0.025,volume=0.22[mr];"
                         "[md][mr]amix=inputs=2:normalize=0[space];")
        self.assertEqual(revise.mastering_settings({"musicGainDb": 2}),
                         {"targetIntegratedLufs": -14.0, "targetTruePeakDbtp": -1.5, "targetLra": 11.0,
                          "refuseDynamicMastering": False})

    def test_authored_targets_use_the_mixer_bounds(self):
        chosen = revise.mastering_settings({"targetIntegratedLufs": -23, "targetLra": "measured"})
        self.assertEqual((chosen["targetIntegratedLufs"], chosen["targetLra"]), (-23, "measured"))
        with self.assertRaisesRegex(longform.MixError, "-27..-12"):
            revise.mastering_settings({"targetIntegratedLufs": -9})
        with self.assertRaisesRegex(RuntimeError, "boolean"):
            revise.music_space({"musicReflections": "off"})

    def test_dry_score_graph_parses_in_ffmpeg(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is unavailable")
        for settings in ({}, {"musicReflections": False}):
            graph = ("[1:a]volume=0dB," + revise.music_space(settings)
                     + "[space][0:a]sidechaincompress=threshold=0.02:ratio=4:attack=10:release=300[music];"
                       "[2:a][0:a]sidechaincompress=threshold=0.035:ratio=4:attack=8:release=360[sfx];"
                       "[0:a][music][sfx]amix=inputs=3:normalize=0:dropout_transition=0[out]")
            command = [ffmpeg, "-v", "error"]
            for _ in range(3):
                command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            command += ["-filter_complex", graph, "-map", "[out]", "-t", "0.05", "-f", "null", "-"]
            with self.subTest(settings):
                result = subprocess.run(command, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
