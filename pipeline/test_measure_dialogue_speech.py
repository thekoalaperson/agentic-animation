"""Transcript evidence for dialogue clips, with the Whisper call replaced offline."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pipeline import measure_dialogue_speech as measure
from pipeline import prepare_dialogue_review as review
from pipeline.audit_sfx_speech import SimpleSegment
from pipeline.test_prepare_dialogue_review import demo_project, digest


class CompareLineTest(unittest.TestCase):
    def test_dandas_and_punctuation_are_not_differences(self):
        result = measure.compare_line("यह, एक पंक्ति है।", "यह एक पंक्ति है")
        self.assertTrue(result["exactAfterNormalization"])
        self.assertEqual((result["missingWords"], result["extraWords"]), ([], []))

    def test_missing_and_extra_words_are_listed(self):
        result = measure.compare_line("यह एक पंक्ति है।", "यह पंक्ति है और")
        self.assertEqual(result["missingWords"], ["एक"])
        self.assertEqual(result["extraWords"], ["और"])
        self.assertFalse(result["sameIgnoringNuktaAndJoiners"])

    def test_nukta_and_joiner_spellings_are_reported_not_called_changes(self):
        result = measure.compare_line("ज़मीन", "जमीन")
        self.assertFalse(result["exactAfterNormalization"])
        self.assertTrue(result["sameIgnoringNuktaAndJoiners"])
        self.assertEqual(result["nuktaOrJoinerDifferences"], [["ज़मीन", "जमीन"]])


class MeasureTest(unittest.TestCase):
    def test_report_is_advisory_and_feeds_the_speech_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, manifest, video = demo_project(Path(tmp))
            plan = json.loads(manifest.read_text())
            jobs = [(row, v, d) for row, v, d, _ in review.verified_jobs(
                plan, manifest.resolve(), project.resolve(), manifest.resolve().parent, plan["jobs"])]

            def transcribe(path):
                info = type("Info", (), {"language": "hi"})()
                return [SimpleSegment({"text": " पंक्ति", "start": 0.5, "end": 1.6,
                                       "avg_logprob": -0.3, "no_speech_prob": 0.1})], info

            output = project / "work/transcripts.json"
            report = measure.measure(jobs, output, model_name="fake", language="hi",
                                     manifest_sha256=digest(manifest), transcribe=transcribe)
            item = report["items"][0]
            self.assertTrue(report["advisoryOnly"])
            self.assertEqual(item["transcriptSpan"], {"start": 0.5, "end": 1.6})
            self.assertTrue(item["comparison"]["exactAfterNormalization"])
            self.assertEqual(item["measured"]["fps"], "24/1")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                measure.measure(jobs, output, model_name="fake", language="hi",
                                manifest_sha256=digest(manifest), transcribe=transcribe)
            done = subprocess.run([sys.executable, str(Path(review.__file__)), str(manifest),
                                   "--project", str(project), "--max-jobs", "1",
                                   "--measure-speech-windows", "--minimum-handle", "0",
                                   "--transcript-evidence", str(output)],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            windows = json.loads((manifest.parent / "speech-windows.json").read_text())
            window = windows["items"][0]["speechWindow"]
            self.assertLessEqual(window["start"], 0.5)
            self.assertGreaterEqual(window["end"], 1.6)
            self.assertEqual(windows["transcriptEvidenceSha256"], digest(output))


if __name__ == "__main__":
    unittest.main()
