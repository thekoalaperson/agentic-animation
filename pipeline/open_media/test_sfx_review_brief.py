"""The SFX review brief takes its inventories as arguments; the CLAP audit reads its prompt wrapper."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import audit_sfx_clap as clap
import build_stable_audio_review_brief as brief


class ReviewBriefTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.ep = Path(self.temporary.name).resolve() / "demo-film"
        self.root = self.ep / "work/sfx-review-v1"
        self.root.mkdir(parents=True)
        manifest = self.ep / "work/demo-sounds/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"families": [
            {"id": f"demo-{name}", "title": name, "category": "room", "shots": [1]}
            for name in ("lamp", "door")]}))
        (self.root / "authored-prompts.json").write_text(json.dumps([
            {"id": "demo-lamp", "seconds": 4, "intent": "demo-s003: a lamp on a table",
             "prompts": ["a lamp hums", "a lamp buzzes"], "directions": ["low", "high"]}]))

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            brief.build(self.ep, self.root, *args)
        return json.loads((self.root / "probe-brief.json").read_text())

    def test_manifests_and_families_come_from_arguments(self):
        written = self.build(["work/demo-sounds/manifest.json"], ["demo-lamp"])
        self.assertEqual(list(written["sourceManifests"]), ["work/demo-sounds/manifest.json"])
        self.assertEqual([job["family"] for job in written["jobs"]], ["demo-lamp", "demo-lamp"])
        with self.assertRaises(AssertionError):
            self.build(["work/demo-sounds/manifest.json"])  # the door family has no authored prompts
        for args in (([],), (["work/missing/manifest.json"],), (["work/demo-sounds/manifest.json"], ["demo-sea"])):
            with self.subTest(args), self.assertRaises(SystemExit):
                self.build(*args)

    def test_clap_prompt_wrapper_defaults_to_the_earlier_briefs_and_can_be_named(self):
        wrapped = "Isolated cinematic sound effect. TrackType: SFX. a door creaks. No music. Dry, clean, mono"
        self.assertEqual(clap.semantic_text(wrapped), "a door creaks.")
        own = {"cutAt": [" | "], "remove": ["SFX: "], "negativeWord": "without"}
        self.assertEqual(clap.semantic_text("SFX: a lamp hums without voices | close", own), "a lamp hums")
        self.assertEqual(clap.semantic_text("a lamp hums. No voices", {}), "a lamp hums. No voices")


if __name__ == "__main__":
    unittest.main()
