#!/usr/bin/env python3
"""Tests for frame-evaluated MP4 portrait reframing."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.edit.shorts_motion_crop import CropSpecError, build_clip_filter


class MotionCropTest(unittest.TestCase):
    def test_static_center_stays_static(self) -> None:
        value = build_clip_filter(2.0, {"cx0": 0.25, "cx1": 0.25}, 54, 96)
        self.assertNotIn("t/", value)
        self.assertIn("0.250000000", value)

    def test_dynamic_center_is_frame_evaluated(self) -> None:
        value = build_clip_filter(2.0, {"cx0": 0.2, "cx1": 0.8}, 54, 96)
        self.assertIn("t/2.000000000", value)
        self.assertIn("clip(", value)

    def test_unsafe_centers_are_rejected(self) -> None:
        with self.assertRaises(CropSpecError):
            build_clip_filter(2.0, {"cx0": -0.01}, 54, 96)
        with self.assertRaises(CropSpecError):
            build_clip_filter(2.0, {"cx1": 1.01}, 54, 96)

    def test_wider_panel_is_authored_and_padded_not_stretched(self) -> None:
        value = build_clip_filter(2, {'layout': 'wide-panel', 'panelAspect': 4/3}, 1080, 1920)
        self.assertIn('scale=1080:810', value)
        self.assertIn('pad=1080:1920:0:600', value)
        self.assertIn('iw/1.333333333', value)
        square = build_clip_filter(2, {'layout': 'wide-panel', 'panelAspect': 1,
                                       'panelTop': .21875}, 1080, 1920)
        self.assertIn('scale=1080:1080', square)
        self.assertIn('pad=1080:1920:0:420', square)

    def test_unsafe_panel_geometry_is_rejected(self) -> None:
        for values in ({'layout': 'guessed'}, {'layout': 'wide-panel', 'panelAspect': True},
                       {'layout': 'wide-panel', 'panelAspect': .5},
                       {'layout': 'wide-panel', 'panelTop': .9}):
            with self.assertRaises(CropSpecError):
                build_clip_filter(2, values, 1080, 1920)

    def test_actual_wider_panel_preserves_both_source_sides_and_clock(self) -> None:
        vf = build_clip_filter(1, {'layout': 'wide-panel', 'panelAspect': 4/3}, 108, 192)
        result = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
            'color=red:s=160x120:d=1:r=24', '-vf',
            'drawbox=x=80:y=0:w=80:h=120:c=blue:t=fill,' + vf,
            '-frames:v', '24', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
            check=True, capture_output=True)
        frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(24, 192, 108, 3)
        for frame in (frames[0], frames[-1]):
            self.assertGreater(int(frame[100, 10, 0]), int(frame[100, 10, 2]) + 100)
            self.assertGreater(int(frame[100, 95, 2]), int(frame[100, 95, 0]) + 100)
            self.assertLess(frame[10].max(), 25)

    def test_actual_mp4_moves_from_left_to_right(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "halves.mp4"
            output = root / "portrait.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                "color=black:s=160x90:d=2:r=24", "-vf",
                "drawbox=x=0:y=0:w=80:h=90:c=red:t=fill,"
                "drawbox=x=80:y=0:w=80:h=90:c=blue:t=fill",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ], check=True)
            vf = build_clip_filter(2.0, {"cx0": 0.2, "cx1": 0.8,
                                         "cy": 0.5, "z0": 1.0}, 54, 96)
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", vf,
                "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
            ], check=True)

            def frame(at: float) -> np.ndarray:
                proc = subprocess.run([
                    "ffmpeg", "-v", "error", "-ss", f"{at:.3f}", "-i", str(output),
                    "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
                ], check=True, capture_output=True)
                return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(96, 54, 3)

            first = frame(0.0).mean(axis=(0, 1))
            last = frame(1.9).mean(axis=(0, 1))
            self.assertGreater(first[0], first[2] + 100)
            self.assertGreater(last[2], last[0] + 100)


if __name__ == "__main__":
    unittest.main()
