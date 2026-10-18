"""Offline tests for the observation lane's spending gate; no model calls."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pipeline import observe_media_finite as observe
from pipeline.test_paid_approval import write_approval


class ObservationApprovalTest(unittest.TestCase):
    def plan(self, root: Path, *, jobs=2, max_credits=10, boolean=True) -> Path:
        still = root / "art/demo-s001.png"
        still.parent.mkdir(parents=True)
        still.write_bytes(b"offline-still")
        digest = hashlib.sha256(still.read_bytes()).hexdigest()
        path = root / "work/observe/plan.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "episode": "DEMO", "projectRoot": "../..", "financialApproval": boolean,
            "maxCredits": max_credits,
            "jobs": [{"id": f"demo-look-{n}", "prompt": "Describe what is visible.",
                      "media": [{"path": "art/demo-s001.png", "sha256": digest,
                                 "url": "https://offline.invalid/demo-s001.png"}]}
                     for n in range(1, jobs + 1)],
        }), encoding="utf-8")
        return path

    def approve(self, plan: Path, root: Path) -> Path:
        doc = json.loads(plan.read_text())
        return write_approval(root / "story/observe-approval.json", episode="DEMO",
                              lane=observe.APPROVAL_LANE, manifest=plan,
                              scope=observe.approval_scope(doc), project_root=root)

    @staticmethod
    def response(credits):
        body = json.dumps({"choices": [{"message": {"content": "a lamp"}}],
                           "credits_consumed": credits}).encode()
        handle = mock.MagicMock()
        handle.__enter__.return_value = io.BytesIO(body)
        return handle

    def test_plan_boolean_alone_never_authorizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan = self.plan(root)
            dry = observe.run(plan, 2, False)
            self.assertEqual((dry["networkCalls"], dry["financialAuthorized"]), (0, False))
            with mock.patch.object(observe.urllib.request, "urlopen") as opened:
                with self.assertRaisesRegex(ValueError, "requires a current approval record"):
                    observe.run(plan, 2, True)
            opened.assert_not_called()

    def test_record_authorizes_and_reported_credits_stop_at_the_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan = self.plan(root, jobs=2, max_credits=4)
            approval = self.approve(plan, root)
            with mock.patch.object(observe.urllib.request, "urlopen",
                                   return_value=self.response(5)) as opened, \
                 mock.patch.object(observe, "env_key", return_value="offline"), \
                 mock.patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "reached maxCredits"):
                    observe.run(plan, 2, True, approval)
            self.assertEqual(opened.call_count, 1)
            first = json.loads((plan.parent / "plan-observations/demo-look-1.json").read_text())
            self.assertEqual(first["status"], "COMPLETED")
            self.assertFalse((plan.parent / "plan-observations/demo-look-2.json").exists())

    def test_plan_without_a_credit_ceiling_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan = self.plan(root)
            doc = json.loads(plan.read_text())
            del doc["maxCredits"]
            plan.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "maxCredits"):
                observe.run(plan, 2, True, root / "missing-approval.json")


if __name__ == "__main__":
    unittest.main()
