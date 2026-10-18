import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from pipeline.paid_approval import PaidApprovalError, sha256_file, validate_paid_approval


def write_approval(record: Path, *, episode, lane: str, manifest: Path,
                   scope: dict, project_root: Path | None = None) -> Path:
    """Write an offline APPROVED version-2 record for tests; never a real grant."""
    root = (project_root or record.parent).resolve()
    manifest = manifest.resolve()
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({
        "schemaVersion": 2, "kind": "paid-generation-authorization",
        "episode": episode, "status": "APPROVED", "authorizationId": "offline-test",
        "authorizedBy": "Offline test fixture", "authorizedAtUtc": "2026-01-01T00:00:00Z",
        "authorizationText": ["Offline fixture; authorizes nothing outside this test."],
        "budgetPolicy": "explicit-finite-limits",
        "projectRoot": os.path.relpath(root, record.parent.resolve()),
        "topUpAuthorized": False, "automaticRetryAuthorized": False,
        "manifests": [{"lane": lane, "manifestPath": manifest.relative_to(root).as_posix(),
                       "manifestSha256": sha256_file(manifest), "finiteExecution": scope}],
    }), encoding="utf-8")
    return record


def grant_retries(record: Path, *, jobs: int, credits: float) -> Path:
    """Turn an offline test record into a version-3 retry grant; never a real grant."""
    document = json.loads(record.read_text(encoding="utf-8"))
    document.update(schemaVersion=3, automaticRetryAuthorized=True,
                    retryBudget={"jobs": jobs, "credits": credits})
    record.write_text(json.dumps(document), encoding="utf-8")
    return record


class PaidApprovalTest(unittest.TestCase):
    def test_version_three_grants_retries_only_where_the_lane_asks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text('{"financialApproval": false}')
            scope = {"jobs": 2, "generatedSeconds": 14, "credits": 70}
            record = write_approval(root / "approval.json", episode="DEMO", lane="grok-dialogue",
                                    manifest=manifest, scope=scope)
            arguments = dict(episode="DEMO", lane="grok-dialogue", manifest_path=manifest,
                             manifest_sha256=sha256_file(manifest), finite_execution=scope)
            version_two = validate_paid_approval(record, **arguments)
            self.assertEqual(set(version_two), {"path", "sha256", "authorizationId", "authorizedAtUtc",
                                                "budgetPolicy", "lane", "manifestSha256",
                                                "finiteExecution"})
            self.assertEqual(validate_paid_approval(record, **arguments, allow_retry_budget=True),
                             {**version_two})
            document = json.loads(record.read_text())
            for change, message in (({"automaticRetryAuthorized": True}, "automaticRetryAuthorized"),
                                    ({"retryBudget": {"jobs": 1, "credits": 5}}, "root schema")):
                record.write_text(json.dumps({**document, **change}))
                with self.subTest(version2=change), self.assertRaisesRegex(PaidApprovalError, message):
                    validate_paid_approval(record, **arguments, allow_retry_budget=True)
            record.write_text(json.dumps(document))
            grant_retries(record, jobs=1, credits=5)
            with self.assertRaisesRegex(PaidApprovalError, "does not retry"):
                validate_paid_approval(record, **arguments)
            granted = validate_paid_approval(record, **arguments, allow_retry_budget=True)
            self.assertEqual((granted["schemaVersion"], granted["retryBudget"]),
                             (3, {"jobs": 1, "credits": 5}))
            self.assertEqual(granted["sha256"], sha256_file(record))
            three = json.loads(record.read_text())
            for change, message in (({"automaticRetryAuthorized": False}, "must be true"),
                                    ({"retryBudget": {"jobs": 0, "credits": 5}}, "retryBudget"),
                                    ({"retryBudget": {"jobs": 1}}, "retryBudget"),
                                    ({"retryBudget": {"jobs": True, "credits": 5}}, "retryBudget"),
                                    ({"topUpAuthorized": True}, "topUpAuthorized"),
                                    ({"status": "DRAFT"}, "status"),
                                    ({"extra": 1}, "root schema")):
                record.write_text(json.dumps({**three, **change}))
                with self.subTest(version3=change), self.assertRaisesRegex(PaidApprovalError, message):
                    validate_paid_approval(record, **arguments, allow_retry_budget=True)
            record.write_text(json.dumps(three))
            manifest.write_text('{"financialApproval": true}')
            with self.assertRaisesRegex(PaidApprovalError, "manifest hash is stale"):
                validate_paid_approval(record, **{**arguments, "manifest_sha256": sha256_file(manifest)},
                                       allow_retry_budget=True)

    def test_admission_requires_exact_current_bytes_and_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text('{"financialApproval": false}')
            admission = root / "approval.json"
            scope = {"requests": 9, "credits": 108, "sourceSeconds": 1360}
            document = {
                "schemaVersion": 2, "kind": "paid-generation-authorization",
                "episode": "DEMO", "status": "APPROVED", "authorizationId": "test",
                "authorizedBy": "Test operator", "authorizedAtUtc": "2026-09-05T01:05:12Z",
                "authorizationText": ["Approve only this finite batch within the listed limits."],
                "budgetPolicy": "explicit-finite-limits", "projectRoot": ".",
                "topUpAuthorized": False, "automaticRetryAuthorized": False,
                "manifests": [{"lane": "suno-score", "manifestPath": "manifest.json",
                    "manifestSha256": sha256_file(manifest), "finiteExecution": scope}],
            }
            arguments = dict(episode="DEMO", lane="suno-score", manifest_path=manifest,
                             manifest_sha256=sha256_file(manifest), finite_execution=scope)
            admission.write_text(json.dumps(document))
            result = validate_paid_approval(admission, **arguments)
            self.assertEqual(result["sha256"], sha256_file(admission))
            self.assertFalse(json.loads(manifest.read_text())["financialApproval"])
            with self.assertRaisesRegex(PaidApprovalError, "scope differs"):
                validate_paid_approval(admission, **{**arguments, "finite_execution": {**scope, "requests": 10}})
            altered = copy.deepcopy(document)
            altered["manifests"][0]["manifestSha256"] = "0" * 64
            admission.write_text(json.dumps(altered))
            with self.assertRaisesRegex(PaidApprovalError, "hash is stale"):
                validate_paid_approval(admission, **arguments)
            admission.write_text(json.dumps(document))
            manifest.write_text('{"financialApproval": true}')
            with self.assertRaisesRegex(PaidApprovalError, "bytes do not match"):
                validate_paid_approval(admission, **arguments)

    def test_eleven_resume_state_binds_external_authorization(self):
        from pipeline import eleven_batch_finite as eleven
        from pipeline.test_eleven_batch_finite import FiniteElevenBatchTest
        with tempfile.TemporaryDirectory() as temporary:
            helper = FiniteElevenBatchTest()
            manifest = helper.manifest(Path(temporary), financial_approval=False)
            plan = eleven.load_plan(manifest, helper.cap(manifest))
            approval = {"sha256": "a" * 64, "authorizationId": "approved"}
            plan.update(approvalAdmission=approval, approvalAdmissionSha256=approval["sha256"],
                        approvalAuthorizationId=approval["authorizationId"])
            state_path = helper.state_for(manifest)
            first = eleven._load_or_create_state(plan, state_path)
            self.assertEqual(first, eleven._load_or_create_state(plan, state_path))
            plan["approvalAdmissionSha256"] = "b" * 64
            with self.assertRaisesRegex(eleven.PlanError, "approvalAdmissionSha256"):
                eleven._load_or_create_state(plan, state_path)


class PaidApprovalCommandTest(unittest.TestCase):
    def test_describe_lists_every_runner_scope(self):
        from pipeline import paid_approval
        from pipeline import eleven_batch_finite, kie_music_finite, observe_media_finite
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(paid_approval.main(["describe"]), 0)
        lanes = json.loads(out.getvalue())["lanes"]
        self.assertEqual(set(lanes["suno-score"]["finiteExecution"]),
                         {"requests", "credits", "sourceSeconds"})
        for module in (eleven_batch_finite, kie_music_finite, observe_media_finite):
            self.assertIn(module.APPROVAL_LANE, lanes)

    def test_described_keys_are_each_runners_own_scope(self):
        import importlib.util
        from pipeline import paid_approval
        from pipeline import eleven_batch_finite, kie_music_finite, observe_media_finite
        spec = importlib.util.spec_from_file_location(
            "dialogue_scope_target", Path(paid_approval.__file__).with_name("kie_dialogue_finite.py"))
        dialogue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dialogue)
        # Each runner's approval_scope reads only these plan fields.
        plans = {
            dialogue: {"jobs": [], "totalSeconds": 0, "maximumCredits": 0},
            eleven_batch_finite: {"items": [], "aggregateCharacters": 0},
            kie_music_finite: {"requestCount": 1, "expectedCredits": 1,
                               "cues": [{"payload": {"duration": 1}}]},
            observe_media_finite: {"jobs": [], "maxCredits": 1},
        }
        self.assertEqual({module.APPROVAL_LANE for module in plans}, set(paid_approval.LANES))
        for module, plan in plans.items():
            with self.subTest(lane=module.APPROVAL_LANE):
                self.assertEqual(set(paid_approval.LANES[module.APPROVAL_LANE]["finiteExecution"]),
                                 set(module.approval_scope(plan)))

    def test_draft_is_refused_until_a_person_grants_it_and_check_names_stale_scope(self):
        from pipeline import paid_approval
        from pipeline import eleven_batch_finite as eleven
        from pipeline.test_eleven_batch_finite import FiniteElevenBatchTest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            work = root / "work/narration-v1"
            work.mkdir(parents=True)
            manifest = FiniteElevenBatchTest().manifest(work)
            record = root / "story/approvals/narration.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(paid_approval.main(
                    ["draft", eleven.APPROVAL_LANE, str(manifest), "--out", str(record),
                     "--project", str(root)]), 0)
            drafted = json.loads(record.read_text())
            self.assertEqual(drafted["status"], "DRAFT")
            self.assertEqual([drafted[k] for k in paid_approval.GRANT_FIELDS], ["", "", [], ""])
            with self.assertRaisesRegex(PaidApprovalError, "status must be 'APPROVED'"):
                paid_approval.check_record(record, manifest, eleven.APPROVAL_LANE, root)
            with self.assertRaisesRegex(PaidApprovalError, "overwrite"):
                paid_approval.draft_record(eleven.APPROVAL_LANE, manifest, record, root)
            drafted.update(status="APPROVED", authorizedBy="Offline test fixture",
                           authorizedAtUtc="2026-01-01T00:00:00Z", authorizationId="offline",
                           authorizationText=["Offline fixture wording."])
            record.write_text(json.dumps(drafted))
            self.assertEqual(paid_approval.check_record(
                record, manifest, eleven.APPROVAL_LANE, root)["lane"], eleven.APPROVAL_LANE)
            drafted["manifests"][0]["finiteExecution"]["characters"] += 1
            record.write_text(json.dumps(drafted))
            with self.assertRaisesRegex(PaidApprovalError, "finiteExecution.characters"):
                paid_approval.check_record(record, manifest, eleven.APPROVAL_LANE, root)

    def test_dialogue_draft_matches_the_runner_scope(self):
        from pipeline import paid_approval
        from pipeline.test_kie_dialogue_finite import DialogueFiniteTest
        fixture = DialogueFiniteTest("test_dry_run_is_offline_and_accepts_seven_seconds")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        record = fixture.ep / "story/approvals/draft.json"
        drafted = paid_approval.draft_record("grok-dialogue", fixture.manifest, record, fixture.ep)
        self.assertEqual(drafted["manifests"][0]["finiteExecution"],
                         {"jobs": 1, "generatedSeconds": 7, "credits": 35})
        self.assertEqual(drafted["manifests"][0]["manifestPath"],
                         "work/dialogue/execution-manifest.json")


if __name__ == "__main__":
    unittest.main()
