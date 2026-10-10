"""Pure metadata and command tests: no video, ffmpeg, GPU or cloud operation."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_finishing_probe as gpu
import package_hindi_delivery as delivery


def fixture():
    stream = dict(codec_name="h264", pix_fmt="yuv420p", r_frame_rate="24/1",
                  avg_frame_rate="24/1", time_base="1/12288", nb_frames="3")
    # Decode order is not display order; B frames must remain legal.
    rows = [dict(pts=258, dts=-766, duration=512), dict(pts=1282, dts=-254, duration=512),
            dict(pts=770, dts=258, duration=512)]
    return stream, rows


class PacketTests(unittest.TestCase):
    def test_exact_pts_and_mp4_count_not_claimed_as_decoded_frames(self):
        s, rows = fixture()
        report = gpu.validate_packet_clock(s, rows, 3)
        self.assertEqual(report["frames"], 3)
        self.assertEqual(report["firstSeconds"], .020996)
        self.assertFalse(report["decodedFrameCountVerified"])
        self.assertIn("not decoded-frame", report["basis"])

    def test_absent_header_count_uses_packet_count_pending_decode(self):
        s, rows = fixture()
        del s["nb_frames"]
        self.assertIn("decoded count pending", gpu.validate_packet_clock(s, rows, 3)["frameCountBasis"])

    def test_rejects_incompatible_counts_formats_and_clocks(self):
        for fault in ("header", "expected", "codec", "depth", "cadence", "duplicate", "dts", "duration", "missing", "bool"):
            s, rows = fixture()
            expected = 4 if fault == "expected" else 3
            if fault == "header": s["nb_frames"] = "4"
            if fault == "codec": s["codec_name"] = "hevc"
            if fault == "depth": s["pix_fmt"] = "yuv420p10le"
            if fault == "cadence": rows[2]["pts"] += 1
            if fault == "duplicate": rows[2]["pts"] = rows[0]["pts"]
            if fault == "dts": rows[2]["dts"] = rows[1]["dts"]
            if fault == "duration": rows[0]["duration"] = 511
            if fault == "missing": del rows[0]["pts"]
            if fault == "bool": rows[0]["pts"] = True
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                gpu.validate_packet_clock(s, rows, expected)

    def test_container_probe_has_no_frame_decode_switch(self):
        with patch.object(gpu.subprocess, "check_output", return_value=b'{}') as run:
            gpu.container("sample.mp4")
            self.assertNotIn("-count_frames", run.call_args[0][0])
            self.assertNotIn("-show_frames", run.call_args[0][0])


class DecodeTests(unittest.TestCase):
    def test_explicit_retained_worker_decoder_and_passthrough_without_picture_encode(self):
        cmd = gpu.decode_command("sample.mp4")
        self.assertNotIn("-hwaccel", cmd)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "h264")
        self.assertEqual(cmd[cmd.index("-threads") + 1], "3")
        self.assertIn("wrapped_avframe", cmd)
        self.assertEqual(cmd[cmd.index("-vsync") + 1], "0")
        self.assertIn("-xerror", cmd)
        self.assertIn("explode", cmd)
        for flag in ("-count_frames", "-show_frames", "-t", "-frames:v", "-r", "-shortest", "-y"):
            self.assertNotIn(flag, cmd)

    def test_actual_final_count_is_required(self):
        progress = "frame=2\ndup_frames=0\ndrop_frames=0\nprogress=continue\nframe=3\ndup_frames=0\ndrop_frames=0\nprogress=end\n"
        self.assertEqual(gpu.parse_decode_progress(progress, 3)["frame"], "3")
        for changed in (progress.replace("frame=3", "frame=4"), progress.replace("progress=end", "progress=continue"),
                        progress.replace("drop_frames=0", "drop_frames=1"), progress.replace("dup_frames=0\n", "")):
            with self.assertRaises(ValueError):
                gpu.parse_decode_progress(changed, 3)

    def test_refuses_mac_before_subprocess_or_file_access(self):
        with patch.object(gpu.platform, "system", return_value="Darwin"), patch.object(gpu.subprocess, "check_output") as run:
            with self.assertRaisesRegex(ValueError, "Linux GPU"):
                gpu.decode_h264_worker(Path("/not-an-input.mp4"), 3)
            run.assert_not_called()

    def test_zero_exit_does_not_hide_a_decode_error_or_truncation(self):
        from subprocess import CompletedProcess
        valid = "frame=3\ndup_frames=0\ndrop_frames=0\nprogress=end\n"
        for stderr, stdout in (("decoder error", valid), ("", valid.replace("frame=3", "frame=2"))):
            with patch.object(gpu.platform, "system", return_value="Linux"), \
                 patch.object(gpu.subprocess, "check_output", return_value="NVIDIA L4\n"), \
                 patch.object(gpu.shutil, "which", return_value="/mock/ffmpeg"), \
                 patch.object(gpu, "sha256", return_value="a" * 64), \
                 patch.object(gpu.subprocess, "run", return_value=CompletedProcess([], 0, stdout, stderr)):
                with self.assertRaises(ValueError):
                    gpu.decode_h264_worker(Path("sample.mp4"), 3)


class PackagingOptInTests(unittest.TestCase):
    def test_one_gpu_decode_no_cpu_picture_scans_and_stable_report(self):
        video, rows = fixture()
        video.update(codec_type="video", width=1280, height=720, duration="0.125")
        audio = dict(codec_type="audio", codec_name="aac", channels=2, sample_rate="48000", duration="0.146")
        doc = dict(streams=[video, audio], format={"duration": "0.146"})
        clock = gpu.validate_packet_clock(video, rows, 3)
        expected = dict(width=1280, height=720, fps="24/1", frames=3, durationSeconds=.125,
            videoCodec="h264", pixelFormat="yuv420p", audioCodec="aac", audioChannels=2,
            audioSampleRate=48000, maxAvDeltaSeconds=.05,
            loudness=dict(targetIntegratedLufs=-14, toleranceLu=.5, maxTruePeakDbtp=-1))
        receipt = dict(sourceSha256="a" * 64, actualDecodedFrameCount=3, decoder="h264",
            hardwareAcceleration="none; explicit CPU decode on qualified GPU worker", hardwareOutputFormat="yuv420p", errorLogEmpty=True,
            errorLogSha256="b" * 64, duplicatedFrames=0, droppedFrames=0, toolSha256="c" * 64,
            ffmpegBinarySha256="d" * 64, passed=True, elapsedSeconds=1, command=["first-temp-path"],
            progressSha256="e" * 64)
        timing = dict(firstPtsSeconds=0, lastEndSeconds=.146, packetCount=8,
                      maxContinuityErrorSeconds=0, timelineSha256="f" * 64)
        with patch.object(delivery, "GPU_FINISHING_PROBE", True), \
             patch.object(gpu, "inspect_h264_mp4", return_value=(doc, clock)), \
             patch.object(gpu, "decode_h264_worker", return_value=receipt) as decode, \
             patch.object(delivery, "_run", side_effect=AssertionError("CPU media scan was called")), \
             patch.object(delivery, "_frame_timing_report", side_effect=AssertionError("Decoded CPU timestamp scan was called")), \
             patch.object(delivery, "_audio_packet_timing_report", return_value=timing), \
             patch.object(delivery, "_measure_loudness", return_value=dict(integratedLufs=-14, truePeakDbtp=-1.5)), \
             patch.object(delivery, "_video_payload_sha256", return_value="a" * 64):
            first = delivery._validate_media_file_uncached(Path("first.mp4"), "long-hi", expected)
            decode.assert_called_once_with(Path("first.mp4"), 3)
            receipt.update(elapsedSeconds=2, command=["second-package-path"], progressSha256="a" * 64)
            second = delivery._validate_media_file_uncached(Path("second.mp4"), "long-hi", expected)
        self.assertEqual(first, second)
        self.assertIn("not decoded-frame", first["frameTimingBasis"])
        self.assertEqual(first["gpuDecode"]["actualDecodedFrameCount"], 3)


if __name__ == "__main__":
    unittest.main()
