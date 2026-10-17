"""Attempt-numbered evidence files accumulate instead of replacing each other."""
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
