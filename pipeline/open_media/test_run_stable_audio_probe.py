"""Refusals the self-hosted SFX runner makes before it loads a model."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

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

    def run_audit(self, plan, folder):
        """Write a batch audit through the runner's own row and audit code, with no GPU."""
        root = self.root / folder
        root.mkdir()
        probe.validate_plan(plan, root, 2)
        runtime = {"model": MODEL, "maximumJobSeconds": probe.maximum_job_seconds(plan, root, MODEL)}
        if plan.get("lengthQualificationRun") is True:
            runtime["lengthQualificationRun"] = True
        rows = []
        for job in plan["jobs"]:
            samples = np.zeros((int(job["seconds"] * 100), 2), dtype=np.float32)
            rows.append(probe.output_row(plan, job, MODEL, "demo-identity", "0" * 64, root / job["output"],
                                         samples, 100, 8, 1.0, 0.1, 0))
        probe.write_batch_audit(root, runtime, rows, 1.0)
        return root / "batch-audit.json", rows, runtime

    def evidence(self, audit):
        copy = self.root / "qualification/batch-audit.json"
        copy.parent.mkdir(exist_ok=True)
        copy.write_bytes(audit.read_bytes())
        return {"path": "qualification/batch-audit.json",
                "sha256": hashlib.sha256(copy.read_bytes()).hexdigest()}

    def test_a_qualification_run_admits_two_longer_jobs_and_marks_them_unqualified(self):
        plan = {**self.plan(seconds=20), "lengthQualificationRun": True}
        _audit, rows, runtime = self.run_audit(plan, "first")
        self.assertIs(rows[0]["lengthQualified"], False)
        self.assertEqual(rows[0]["seconds"], 20)
        self.assertEqual(runtime["maximumJobSeconds"], 20)
        three = {**plan, "jobs": [{**plan["jobs"][0], "id": f"demo-{i}", "output": f"outputs/demo-{i}.wav"}
                                  for i in range(3)]}
        cases = {
            "at most two jobs": {**three, "authorization": "demo-approval", "productionApproval": False},
            "true or false": {**plan, "lengthQualificationRun": "yes"},
            "positive finite": {**self.plan(seconds=float("inf")), "lengthQualificationRun": True},
            "cannot also cite": {**plan, "lengthQualification": {"path": "x", "sha256": "0" * 64}},
        }
        for message, bad in cases.items():
            with self.subTest(message), self.assertRaisesRegex(RuntimeError, message):
                probe.validate_plan(bad, self.root, 100)

    def test_an_ordinary_batch_carries_no_qualification_keys(self):
        _audit, rows, runtime = self.run_audit(self.plan(), "ordinary")
        self.assertNotIn("lengthQualified", rows[0])
        self.assertNotIn("lengthQualificationRun", runtime)
        self.assertEqual(runtime["maximumJobSeconds"], 10)
        self.assertEqual(set(rows[0]), {
            "id", "status", "identity", "sha256", "path", "job", "seconds", "sampleRate", "channels",
            "peak", "rms", "overFullScaleSamples", "steps", "cfgScale", "qualifiedSettings",
            "inferenceAndSaveSeconds", "peakGpuMemoryBytes", "normalization", "creativeVerdict"})

    def test_a_retained_qualification_run_raises_the_length_it_proves(self):
        audit, _rows, _runtime = self.run_audit({**self.plan(seconds=20), "lengthQualificationRun": True}, "first")
        evidence = self.evidence(audit)
        plan = {**self.plan(seconds=20), "lengthQualification": evidence}
        probe.validate_plan(plan, self.root, 2)
        self.assertEqual(probe.maximum_job_seconds(plan, self.root, MODEL), 20)
        with self.assertRaisesRegex(RuntimeError, "qualification run"):
            probe.validate_plan({**self.plan(seconds=25), "lengthQualification": evidence}, self.root, 2)
        with self.assertRaisesRegex(RuntimeError, "hash bound"):
            probe.validate_plan({**plan, "lengthQualification": {**evidence, "sha256": "0" * 64}}, self.root, 2)

    def test_an_ordinary_audit_or_another_model_does_not_qualify_a_length(self):
        ordinary, _rows, _runtime = self.run_audit(self.plan(), "ordinary")
        with self.assertRaisesRegex(RuntimeError, "length-qualification run of this model"):
            probe.validate_plan({**self.plan(seconds=4), "lengthQualification": self.evidence(ordinary)},
                                self.root, 2)
        other = self.root / "other.json"
        other.write_text(json.dumps({"runtime": {"model": "demo-other-model", "lengthQualificationRun": True},
                                     "outputs": [{"status": "complete", "seconds": 20,
                                                  "job": {"prompt": "demo: a room tone", "seconds": 20}}]}))
        with self.assertRaisesRegex(RuntimeError, "length-qualification run of this model"):
            probe.validate_plan({**self.plan(seconds=20), "lengthQualification": self.evidence(other)},
                                self.root, 2)

if __name__ == "__main__":
    unittest.main()
