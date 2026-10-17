"""Offline tests for kie_dialogue_finite.py --freeze; no model calls or media."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from pipeline import kie_dialogue_finite as RUNNER


PRICE = dict(credits_per_second=5, price_verified_at="2026-01-01T00:00:00Z",
             price_evidence="offline test fixture; not a provider quote")


class DialogueFreezeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ep = Path(self.tmp.name).resolve() / "demo-project"
        (self.ep / "story").mkdir(parents=True)
        (self.ep / "story/GOAL.md").write_text("demo direction\n", encoding="utf-8")
        (self.ep / "story/production-overrides-v1.json").write_text('{"items": {}}\n')
        self.batch = self.ep / "work/dialogue-v1"
        items = []
        for number in (1, 2, 3):
            job_id = f"demo-s00{number}"
            still = self.ep / f"art/stills/{job_id}.png"
            still.parent.mkdir(parents=True, exist_ok=True)
            still.write_bytes(f"offline still {number}".encode())
            prompt = self.batch / f"prompts/{job_id}.txt"
            prompt.parent.mkdir(parents=True, exist_ok=True)
            prompt.write_text(f"demo-s00{number}: a lamp on a table. Line: नमस्ते {number}।\n",
                              encoding="utf-8")
            # The shape prepare_episode_production.py writes: absolute still, text.
            items.append({"id": job_id, "speaker": "Demo speaker", "still": str(still),
                          "text": f"नमस्ते {number}।", "english": "Hello."})
        self.draft = self.ep / "work/production-intake-v1/dialogue-draft.json"
        self.draft.parent.mkdir(parents=True)
        self.draft.write_text(json.dumps({"episode": "DEMO", "items": items}, ensure_ascii=False),
                              encoding="utf-8")
        self.durations = {"demo-s001": 6, "demo-s002": 4, "demo-s003": 9}
        RUNNER.EPISODE = self.ep

    def freeze(self, target: Path, **extra):
        RUNNER.EPISODE = self.ep
        options = dict(PRICE, durations=self.durations, no_source_audit=True)
        options.update(extra)
        return RUNNER.freeze(self.draft, target, **options)

    def test_freeze_writes_a_manifest_the_validator_accepts(self):
        target = self.batch / "execution-manifest.json"
        result = self.freeze(target)
        self.assertEqual((result["mode"], result["jobs"], result["generatedSeconds"]),
                         ("DRY_RUN", 3, 19))
        self.assertFalse(result["financialAuthorized"])
        doc = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(set(doc), RUNNER.ROOT_FIELDS)
        self.assertFalse(doc["financialApproval"])
        self.assertEqual(doc["maximumCredits"], 19 * 5)
        self.assertEqual(doc["jobs"][0]["source"], "art/stills/demo-s001.png")
        self.assertEqual(doc["jobs"][0]["outputStem"], "work/dialogue-v1/motion/demo-s001")
        self.assertEqual((self.batch / "draft-manifest.json").read_bytes(), self.draft.read_bytes())
        self.assertFalse(json.loads((self.batch / "source-audit.json").read_text())["newAngleAuditPerformed"])
        before = target.read_bytes()
        with self.assertRaisesRegex(RUNNER.PlanError, "overwrite a frozen manifest"):
            self.freeze(target)
        self.assertEqual(target.read_bytes(), before)

    def test_command_line_freeze(self):
        durations = self.ep / "work/durations.json"
        durations.write_text(json.dumps(self.durations))
        target = self.batch / "execution-manifest.json"
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            code = RUNNER.main([str(target), "--freeze", str(self.draft), "--project", str(self.ep),
                                "--credits-per-second", "5", "--price-verified-at",
                                PRICE["price_verified_at"], "--price-evidence",
                                PRICE["price_evidence"], "--durations", str(durations),
                                "--no-source-audit"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(shown.getvalue())["generatedSeconds"], 19)

    def test_every_missing_input_is_named_in_one_refusal(self):
        (self.batch / "prompts/demo-s002.txt").unlink()
        with self.assertRaises(RUNNER.PlanError) as caught:
            self.freeze(self.batch / "execution-manifest.json",
                        durations={"demo-s001": 6, "demo-s002": 4})
        message = str(caught.exception)
        self.assertIn("demo-s002: prompt not written yet", message)
        self.assertIn("demo-s003: durationSeconds missing", message)
        self.assertFalse((self.batch / "execution-manifest.json").exists())

    def test_audit_absence_must_be_declared_and_price_must_be_supplied(self):
        with self.assertRaisesRegex(RUNNER.PlanError, "--no-source-audit"):
            self.freeze(self.batch / "execution-manifest.json", no_source_audit=False)
        with self.assertRaisesRegex(RUNNER.PlanError, "credits-per-second"):
            self.freeze(self.batch / "execution-manifest.json", credits_per_second=None)

    def test_pilot_then_main_batch_never_holds_a_job_twice(self):
        pilot = self.ep / "work/dialogue-pilot-v1/execution-manifest.json"
        for job_id in ("demo-s002",):
            prompt = pilot.parent / f"prompts/{job_id}.txt"
            prompt.parent.mkdir(parents=True)
            prompt.write_text((self.batch / f"prompts/{job_id}.txt").read_text(encoding="utf-8"),
                              encoding="utf-8")
        with self.assertRaisesRegex(RUNNER.PlanError, "selection-reason"):
            self.freeze(pilot, only=["demo-s002"])
        self.freeze(pilot, only=["demo-s002"], selection_reason="riskiest staging first")
        main = self.batch / "execution-manifest.json"
        result = self.freeze(main, omit_from=[pilot], selection_reason="takes the pilot already holds")
        self.assertEqual(result["jobs"], 2)
        doc = json.loads(main.read_text(encoding="utf-8"))
        self.assertEqual([job["id"] for job in doc["jobs"]], ["demo-s001", "demo-s003"])
        self.assertEqual(doc["siblingManifests"][0]["path"], "work/dialogue-pilot-v1/execution-manifest.json")
        note = json.loads((self.batch / "selection-note.json").read_text())
        self.assertEqual(note["omittedBecauseHeldBy"], {doc["siblingManifests"][0]["path"]: ["demo-s002"]})

        # A hand-edited manifest that also holds the pilot's job is refused.
        held = json.loads(pilot.read_text(encoding="utf-8"))["jobs"][0]
        held.update(prompt="work/dialogue-v1/prompts/demo-s002.txt",
                    outputStem="work/dialogue-v1/motion/demo-s002")
        doc["jobs"].append(held)
        doc.update(jobCount=3, maximumGeneratedSeconds=19, maximumCredits=95)
        overlap = self.batch / "overlap-manifest.json"
        overlap.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        RUNNER.configure(overlap, self.ep)
        with self.assertRaisesRegex(RUNNER.PlanError, "paid for once"):
            RUNNER.load_plan(overlap, 19)

    def test_frozen_manifest_drafts_an_approval_record_of_the_same_scope(self):
        from pipeline import paid_approval
        target = self.batch / "execution-manifest.json"
        self.freeze(target)
        record = paid_approval.draft_record("grok-dialogue", target,
                                            self.ep / "story/approvals/dialogue-v1.json", self.ep)
        self.assertEqual(record["manifests"][0]["finiteExecution"],
                         {"jobs": 3, "generatedSeconds": 19, "credits": 95})


if __name__ == "__main__":
    unittest.main()
