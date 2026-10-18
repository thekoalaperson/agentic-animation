"""Render real dips and dissolves from tiny synthetic sources when ffmpeg can."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import assemble_longform as assembly

WIDTH, HEIGHT = 128, 72


def _ffmpeg_can_render() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except OSError:
        return False
    return " xfade " in filters and "libx264" in encoders


@unittest.skipUnless(_ffmpeg_can_render(), "needs ffmpeg with xfade and libx264")
class RenderedTransitionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.sources = {}
        for name, colour in (("a", "red"), ("b", "blue")):
            path = self.root / f"{name}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=256x144:r=24:d=3",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
            self.sources[name] = path

    def tearDown(self):
        self.temp.cleanup()

    def item(self, name, start_frame, frames, **extra):
        return {"id": f"p-{name}", "source": str(self.sources[name]), "sourceSha256": name * 64,
                "sourceStart": start_frame / 24, "sourceDuration": frames / 24, "speed": 1.0,
                "frames": frames, **extra}

    @staticmethod
    def frames(path: Path) -> np.ndarray:
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                                       "-pix_fmt", "rgb24", "-"])
        return np.frombuffer(raw, np.uint8).reshape(-1, HEIGHT, WIDTH, 3).astype(int)

    def test_dissolve_overlaps_unused_frames_and_keeps_the_frame_count(self):
        cut = [assembly.render_segment(self.item("a", 12, 24), self.cache, WIDTH, HEIGHT),
               assembly.render_segment(self.item("b", 12, 24), self.cache, WIDTH, HEIGHT)]
        hard = np.concatenate([self.frames(path) for path in cut])
        run = {"id": "p-a-dissolve-p-b", "frames": 48, "dissolves": [12], "sources": [
            self.item("a", 12, 24 + 6, shownFrames=24), self.item("b", 6, 6 + 24, shownFrames=24)]}
        soft = self.frames(assembly.render_dissolve_run(run, self.cache, WIDTH, HEIGHT))
        self.assertEqual(soft.shape, hard.shape)
        centre = soft[:, HEIGHT // 2, WIDTH // 2]
        # Outside the overlap each picture is what the cut shows; inside it both are mixed.
        np.testing.assert_allclose(centre[:16], hard[:16, HEIGHT // 2, WIDTH // 2], atol=6)
        np.testing.assert_allclose(centre[32:], hard[32:, HEIGHT // 2, WIDTH // 2], atol=6)
        self.assertTrue(np.all(centre[20:28, 0] > 40) and np.all(centre[20:28, 2] > 40))
        # The HD range cache keys on decoded RGB, so the dissolve range never reuses the cut's pixels.
        overlap = slice(18, 30)
        self.assertNotEqual(hashlib.sha256(soft[overlap].astype(np.uint8).tobytes()).hexdigest(),
                            hashlib.sha256(hard[overlap].astype(np.uint8).tobytes()).hexdigest())

    def test_dip_fades_inside_the_frames_already_shown(self):
        out = self.frames(assembly.render_segment(
            self.item("a", 0, 24, fadeOut={"frames": 6, "colour": "black"}), self.cache, WIDTH, HEIGHT))
        into = self.frames(assembly.render_segment(
            self.item("b", 0, 24, fadeIn={"frames": 6, "colour": "white"}), self.cache, WIDTH, HEIGHT))
        self.assertEqual((len(out), len(into)), (24, 24))
        self.assertLess(out[-1].mean(), 4)  # last frame reaches black
        self.assertGreater(out[-2, :, :, 0].mean(), 20)  # and the frames before it ramp down
        self.assertGreater(out[0, :, :, 0].mean(), 200)  # untouched before the fade
        self.assertGreater(into[0].mean(), 240)  # first frame starts from white
        self.assertGreater(into[-1, :, :, 2].mean(), 200)


if __name__ == "__main__":
    unittest.main()
