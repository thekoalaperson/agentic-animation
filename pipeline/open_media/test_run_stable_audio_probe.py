"""Refusals the self-hosted SFX runner makes before it loads a model."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import run_stable_audio_probe as probe

MODEL = "stabilityai/stable-audio-3-small-sfx"


class StableAudioPlanTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, **job):
        return {"model": MODEL, "generationApiAllowed": False,
                "jobs": [{"id": "demo-hum", "prompt": "demo-s003: a lamp hums", "seconds": 4,
                          "seed": 1, "output": "outputs/demo-hum.wav", **job}]}

    def test_qualified_jobs_pass_and_carry_their_settings(self):
        self.assertEqual(probe.validate_plan(self.plan(), self.root, 2), MODEL)
        self.assertEqual(probe.job_settings(self.plan()["jobs"][0], MODEL), (8, 1.0))
        self.assertEqual(probe.job_settings(self.plan(steps=24, cfgScale=3)["jobs"][0], MODEL), (24, 3.0))

    def test_unknown_model_bad_settings_and_unqualified_length_are_refused(self):
        cases = {
            "No qualified local route": {**self.plan(), "model": "demo-other-model"},
            "steps": self.plan(steps=0),
            "cfgScale": self.plan(cfgScale="high"),
            "qualification run": self.plan(seconds=20),
        }
        for message, plan in cases.items():
            with self.subTest(message), self.assertRaisesRegex(RuntimeError, message):
                probe.validate_plan(plan, self.root, 2)

    def test_a_retained_qualification_run_raises_the_length_it_proves(self):
        audit = self.root / "qualification/batch-audit.json"
        audit.parent.mkdir()
        audit.write_text(json.dumps({"runtime": {"model": MODEL}, "outputs": [
            {"status": "complete", "seconds": 20, "job": {"prompt": "demo: a long room tone"}}]}))
        evidence = {"path": "qualification/batch-audit.json",
                    "sha256": hashlib.sha256(audit.read_bytes()).hexdigest()}
        plan = {**self.plan(seconds=20), "lengthQualification": evidence}
        probe.validate_plan(plan, self.root, 2)
        with self.assertRaisesRegex(RuntimeError, "qualification run"):
            probe.validate_plan({**self.plan(seconds=25), "lengthQualification": evidence}, self.root, 2)
        with self.assertRaisesRegex(RuntimeError, "hash bound"):
            probe.validate_plan({**plan, "lengthQualification": {**evidence, "sha256": "0" * 64}}, self.root, 2)


if __name__ == "__main__":
    unittest.main()
