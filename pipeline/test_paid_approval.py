import copy
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.paid_approval import PaidApprovalError, sha256_file, validate_paid_approval


class PaidApprovalTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
