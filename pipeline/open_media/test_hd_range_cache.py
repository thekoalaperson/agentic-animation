import json
from pathlib import Path
import tempfile
import unittest
from hd_range_cache import PIXEL_CONTRACT, digest_file, range_layout, load_reuse_index


class RangeCacheTests(unittest.TestCase):
    def test_explicit_picture_ranges_survive_timeline_shifts(self):
        doc = {"kind": "source-bound-hd-picture-ranges", "totalFrames": 8,
               "ranges": [{"id": "a", "startFrame": 0, "frames": 3},
                          {"id": "b", "startFrame": 3, "frames": 5}]}
        self.assertEqual(len(range_layout(8, 600, doc)), 2)
        for mutate in (lambda x: x[1].update(startFrame=4), lambda x: x[1].update(id="a"),
                       lambda x: x[1].update(id="../outside"), lambda x: x[1].update(frames=601)):
            bad = json.loads(json.dumps(doc)); mutate(bad["ranges"])
            with self.assertRaises(ValueError): range_layout(8, 600, bad)

    def test_exact_pixels_model_and_output_are_required_not_ordinal_identity(self):
        old = {"sourceVideoSha256": "a" * 64, "modelSha256": "b" * 64,
               "reconstruction": "same-neural-math", "sourceResolution": [1280,720]}
        new = {**old, "sourceVideoSha256": "c" * 64}
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); (root / "chunks").mkdir()
            output = root / "chunks/p123.mp4"; output.write_bytes(b"unchanged-qualified-output")
            (root / "configuration.json").write_text(json.dumps(old))
            record = {**old, "chunk": "p123", "status": "complete", "time": 1,
                      "startFrame": 999, "frames": 3, "sourceRgbSha256": "d" * 64,
                      "sourcePixelContract": PIXEL_CONTRACT, "sha256": digest_file(output)}
            state = root / "state-shard-00.jsonl"; state.write_text(json.dumps(record) + "\n")
            index = load_reuse_index([root], new)
            self.assertIn(("d" * 64, 3), index)
            self.assertNotIn(("e" * 64, 3), index)
            self.assertNotIn(("d" * 64, 4), index)
            self.assertEqual(index[("d" * 64, 3)]["state"]["startFrame"], 999)
            (root / "reuse-exclusions.json").write_text(json.dumps({"sourceVideoSha256":old["sourceVideoSha256"],"chunks":["p123"]}))
            self.assertEqual(load_reuse_index([root], new), {})
            (root / "reuse-exclusions.json").unlink()
            (root / "audit.json").write_text(json.dumps({"regionalSourceRepairs":[{"chunk":"p123"}]}))
            with self.assertRaises(ValueError): load_reuse_index([root], new)
            (root / "audit.json").unlink()
            with self.assertRaises(ValueError): load_reuse_index([root], {**new, "modelSha256": "f" * 64})
            output.write_bytes(b"mutated")
            self.assertEqual(load_reuse_index([root], new), {})

    def test_no_pixel_hash_or_incomplete_state_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); (root / "chunks").mkdir()
            contract = {"sourceVideoSha256": "a" * 64, "modelSha256": "b" * 64}
            (root / "configuration.json").write_text(json.dumps(contract))
            output = root / "chunks/p1.mp4"; output.write_bytes(b"pixels")
            row = {**contract, "chunk": "p1", "status": "complete", "sha256": digest_file(output), "frames": 3}
            state = root / "state.jsonl"; state.write_text(json.dumps(row) + "\n{interrupted")
            self.assertEqual(load_reuse_index([root], contract), {})
            row.update(sourceRgbSha256="d"*64,sourcePixelContract=PIXEL_CONTRACT,status="running")
            state.write_text(json.dumps(row) + "\n")
            self.assertEqual(load_reuse_index([root], contract), {})


if __name__ == "__main__": unittest.main()
