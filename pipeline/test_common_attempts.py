"""Shared evidence helpers: numbered attempts and the project upload log."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pipeline import _common


class AttemptRecordTest(unittest.TestCase):
    def test_each_attempt_is_a_new_file_and_earlier_ones_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            first = _common.write_attempt_json(folder, "credit-preflight", {"availableCredits": 90})
            second = _common.write_attempt_json(folder, "credit-preflight", {"availableCredits": 60})
            self.assertEqual((first.name, second.name),
                             ("credit-preflight-attempt-001.json", "credit-preflight-attempt-002.json"))
            self.assertEqual(json.loads(first.read_text())["availableCredits"], 90)
            records = _common.attempt_records(folder, "credit-preflight")
            self.assertEqual([r["attempt"] for r in records], [1, 2])
            self.assertEqual([r["record"]["availableCredits"] for r in records], [90, 60])
            self.assertEqual(_common.attempt_records(folder, "execution-receipt"), [])



class UploadLogTest(unittest.TestCase):
    def test_each_upload_appends_hash_host_name_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            still = Path(tmp) / "demo-s001.png"
            still.write_bytes(b"offline still")
            log = Path(tmp) / "work/upload-log.jsonl"
            reply = {"data": {"downloadUrl": "https://files.invalid/demo-s001.png"}}
            with mock.patch.object(_common, "http_json", return_value=reply), \
                 mock.patch.object(_common, "env_key", return_value="offline"):
                self.assertEqual(_common.upload_ref(still, log), "https://files.invalid/demo-s001.png")
                _common.upload_ref(still, log)
            lines = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(set(lines[0]), {"sha256", "providerHost", "remoteName", "uploadedAtUtc"})
            self.assertEqual(lines[0]["providerHost"], "kieai.redpandaai.co")
            self.assertTrue(lines[0]["remoteName"].endswith("_demo-s001.png"))


if __name__ == "__main__":
    unittest.main()
