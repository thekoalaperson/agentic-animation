"""Small real-media regression for the approved-audio HD assembly contract."""
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

if __package__:
    from . import hd_audio_remux as remux
else:
    import hd_audio_remux as remux


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
class HDRemuxTest(unittest.TestCase):
    def test_offset_frames_and_audio_tail_preserved(self):
        with tempfile.TemporaryDirectory(prefix="hd-remux-test-") as directory:
            root = Path(directory)
            base, source, picture, final = [root / f"{n}.mp4" for n in
                                            ("base", "source", "picture", "final")]

            def ffmpeg(*args):
                subprocess.run(["ffmpeg", "-y", "-v", "error", *map(str, args)], check=True)

            # Audio intentionally outlasts picture; preserving its final packet
            # catches accidental -shortest truncation as well as A/V offset loss.
            ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x72:rate=24:duration=2",
                   "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2.1",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", base)
            ffmpeg("-i", base, "-map", "0:v:0", "-c", "copy", "-an", picture)
            ffmpeg("-copyts", "-itsoffset", "0.02099609375", "-i", picture,
                   "-i", base, "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
                   "-movie_timescale", "1536000", "-video_track_timescale", "12288", source)
            result = remux.remux_approved_audio(picture, source, final)
            self.assertEqual(result["relativeStartErrorSeconds"], 0)
            self.assertEqual(result["sourceVideoStartSeconds"], float(Fraction(258, 12288)))
            self.assertTrue(result["approvedAudioPacketClocksUnchanged"])
            self.assertEqual(remux.audio_packet_clock(source), remux.audio_packet_clock(final))
            self.assertEqual(remux.packet_hash(picture, "0:v:0"), remux.packet_hash(final, "0:v:0"))
            stream = next(s for s in remux.streams(final) if s["codec_type"] == "video")
            self.assertEqual(int(stream["nb_frames"]), 48)
            decode = subprocess.run([
                "ffmpeg", "-v", "error", "-xerror", "-reinit_filter", "0", "-i", str(final),
                "-fps_mode", "passthrough", "-enc_time_base:v", "-1", "-f", "null", "-"
            ], check=True, capture_output=True, text=True)
            self.assertEqual(decode.stderr.strip(), "")


if __name__ == "__main__":
    unittest.main()
