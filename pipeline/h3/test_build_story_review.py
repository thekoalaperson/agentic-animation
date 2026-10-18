import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import build_story_review as review


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StoryReviewEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name).resolve() / "demo-project"
        run = self.project / "work/h3-demo"
        (run / "silent").mkdir(parents=True)
        (run / "review").mkdir()
        silent = run / "silent/demo-s001-a.mp4"
        silent.write_bytes(b"silent picture")
        (run / "review/demo-s001-a-2fps.jpg").write_bytes(b"sheet")
        self.write(run / "local-audit.json", {
            "frameSampling": "every 12th frame plus the last",
            "outputs": [
                {"id": "demo-s001-a", "status": "technical-pass", "sheet": "/worker/review/demo-s001-a-2fps.jpg",
                 "silent": {"file": "/worker/silent/demo-s001-a.mp4", "sha256": sha(silent)}},
                {"id": "demo-s099-a", "status": "technical-pass",
                 "silent": {"file": "/worker/silent/demo-s099-a.mp4", "sha256": "0" * 64}},
                {"id": "demo-s002-a", "status": "not-complete"},
                "a malformed row is skipped",
            ]})
        dialogue = self.project / "work/grok-dialogue-v1"
        (dialogue / "motion").mkdir(parents=True)
        (dialogue / "dense-review").mkdir()
        take = dialogue / "motion/demo-s002.mp4"
        take.write_bytes(b"dialogue take")
        (dialogue / "dense-review/demo-s002-2fps.jpg").write_bytes(b"sheet")
        self.write(dialogue / "dense-review-index.json", {"items": [{
            "id": "demo-s002", "source": "motion/demo-s002.mp4", "sourceSha256": sha(take),
            "sheet": "dense-review/demo-s002-2fps.jpg", "reviewScope": "every 12th frame; not playback"}]})
        self.data = {"shots": [
            {"id": "demo-s001", "shot": "1", "speaker": "Narrator", "candidates": []},
            {"id": "demo-s002", "shot": "2", "speaker": "Character One", "candidates": []}]}

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_worker_evidence_joins_its_row_with_what_it_checked(self):
        unplaced = review.add_evidence(
            self.data, self.project, [self.project / "work/h3-demo/local-audit.json"],
            [self.project / "work/grok-dialogue-v1/dense-review-index.json"])
        self.assertEqual(unplaced, ["demo-s099-a"])
        h3, = self.data["shots"][0]["candidates"]
        self.assertEqual((h3["path"], h3["hashVerified"], h3["checked"]),
                         ("work/h3-demo/silent/demo-s001-a.mp4", True, "every 12th frame plus the last"))
        self.assertTrue(h3["sheet"].is_file())
        take, = self.data["shots"][1]["candidates"]
        self.assertEqual((take["kind"], take["checkedBy"]), ("grok-dialogue", "prepare_dialogue_review.py"))
        self.assertEqual(take["creativeApproval"], "not established by this check")

    def test_evidence_for_a_listed_take_is_joined_not_duplicated(self):
        self.data["shots"][0]["candidates"].append({"path": "work/h3-demo/silent/demo-s001-a.mp4",
                                                    "kind": "h3-base", "hashVerified": True})
        review.add_evidence(self.data, self.project, [self.project / "work/h3-demo/local-audit.json"])
        listed, = self.data["shots"][0]["candidates"]
        self.assertEqual(listed["checkedBy"], "review_finite.py")


if __name__ == "__main__":
    unittest.main()
