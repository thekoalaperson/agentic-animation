from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipeline.audit_sfx_speech import audit, credible_rows


class SpeechScreenTests(unittest.TestCase):
    def test_thresholds_are_conservative_and_exact(self):
        segments = [
            SimpleNamespace(text=" words ", start=0, end=.3, avg_logprob=-.8, no_speech_prob=.45),
            SimpleNamespace(text="too short", start=0, end=.29, avg_logprob=-.2, no_speech_prob=.1),
            SimpleNamespace(text="too uncertain", start=0, end=1, avg_logprob=-.81, no_speech_prob=.1),
            SimpleNamespace(text="likely effect", start=0, end=1, avg_logprob=-.2, no_speech_prob=.46),
        ]
        self.assertEqual([row["text"] for row in credible_rows(segments)], ["words"])

    def test_audit_hashes_and_quarantines_without_media_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources"
            source.mkdir()
            (source / "a.wav").write_bytes(b"RIFF-a")
            (source / "b.wav").write_bytes(b"RIFF-b")
            output = root / "work" / "audit.json"

            def fake(path):
                segments = [] if path.name == "a.wav" else [
                    SimpleNamespace(text="hello", start=.1, end=.9, avg_logprob=-.2, no_speech_prob=.1)
                ]
                return segments, SimpleNamespace(language="en", language_probability=.9)

            result = audit(source, output, "stub", fake)
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["quarantined"], 1)
            self.assertEqual([row["status"] for row in result["results"]], [
                "no-credible-speech-detected", "quarantine"
            ])
            self.assertEqual(json.loads(output.read_text())["quarantined"], 1)
            self.assertEqual(sorted(path.name for path in source.iterdir()), ["a.wav", "b.wav"])

    def test_report_cannot_enter_deliverables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources"
            source.mkdir()
            (source / "a.wav").write_bytes(b"RIFF")
            with self.assertRaisesRegex(ValueError, "never deliverables"):
                audit(source, root / "deliverables" / "audit.json", "stub", lambda _: ([], object()))


if __name__ == "__main__":
    unittest.main()
