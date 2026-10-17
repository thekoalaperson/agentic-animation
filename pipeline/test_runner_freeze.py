"""Offline tests for the narration and score --freeze modes; no network or media."""
import json
from pathlib import Path
import tempfile
import unittest

from pipeline import eleven_batch_finite as narration
from pipeline import kie_music_finite as score


SETTINGS = {"stability": 0.4, "similarity_boost": 0.7, "style": 0.2,
            "use_speaker_boost": False, "speed": 1.0}


class NarrationFreezeTest(unittest.TestCase):
    def test_prepare_shaped_plan_freezes_and_binds_text_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            (project / "story").mkdir()
            (project / "story/GOAL.md").write_text("demo direction\n", encoding="utf-8")
            (project / "story/demo-s002.txt").write_text("दूसरा वाक्य।\n", encoding="utf-8")
            draft = project / "work/production-intake-v1/narration-plan.json"
            draft.parent.mkdir(parents=True)
            draft.write_text(json.dumps({
                "episode": "DEMO", "model": "eleven_multilingual_v2", "voice": "demoVoice_01",
                "items": [
                    {"id": "demo-s001-hi", "shotId": "demo-s001", "language": "hi",
                     "text": "पहला वाक्य।", "role": "narration",
                     "output": "audio/narration-v1/demo-s001-hi.mp3"},
                    {"id": "demo-s002-hi", "textFile": "story/demo-s002.txt",
                     "output": "audio/narration-v1/demo-s002-hi.mp3"}]},
                ensure_ascii=False), encoding="utf-8")
            manifest = project / "work/narration-v1/manifest.json"
            result = narration.freeze(draft, manifest, project, voice_settings=SETTINGS,
                                      authority_path=project / "story/GOAL.md")
            self.assertEqual(result["mode"], "DRY_RUN")
            self.assertFalse(result["financialAuthorized"])
            expected = len("पहला वाक्य।") + len("दूसरा वाक्य।")
            self.assertEqual(result["aggregateCharacters"], expected)
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertFalse(doc["financialApproval"])
            self.assertEqual(doc["voiceSettings"], SETTINGS)
            self.assertIn("textSha256", doc["items"][1])
            self.assertEqual(result["items"][0]["output"],
                             str(project / "audio/narration-v1/demo-s001-hi.mp3"))
            with self.assertRaisesRegex(narration.PlanError, "overwrite a frozen manifest"):
                narration.freeze(draft, manifest, project, voice_settings=SETTINGS)
            (project / "story/demo-s002.txt").write_text("बदला।\n", encoding="utf-8")
            with self.assertRaisesRegex(narration.PlanError, "textFile changed"):
                narration.load_plan(manifest, expected)
            (project / "story/demo-s002.txt").write_text("दूसरा वाक्य।\n", encoding="utf-8")
            (project / "story/GOAL.md").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(narration.PlanError, "authority changed"):
                narration.load_plan(manifest, expected)

    def test_voice_settings_and_voice_are_never_supplied_by_the_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            draft = project / "work/plan.json"
            draft.parent.mkdir(parents=True)
            draft.write_text(json.dumps({"items": [{"id": "a", "text": "पाठ।", "output": "a.mp3"}]},
                                        ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(narration.PlanError, "voiceSettings keys"):
                narration.freeze(draft, project / "work/m.json", project,
                                 voice_settings={"speed": 1.0}, voice_id="demoVoice")
            with self.assertRaisesRegex(narration.PlanError, "no cast voice"):
                narration.freeze(draft, project / "work/m.json", project, voice_settings=SETTINGS)


class ScoreFreezeTest(unittest.TestCase):
    def test_draft_freezes_to_the_cues_load_plan_expects(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            (project / "story").mkdir()
            authority = project / "story/GOAL.md"
            authority.write_text("demo direction\n", encoding="utf-8")
            draft = project / "work/score-drafts/demo-draft.json"
            draft.parent.mkdir(parents=True)
            draft.write_text(json.dumps({
                "episode": "DEMO",
                "commonPayload": {"model": "V6", "customMode": True, "instrumental": True,
                                  "prompt": "", "callBackUrl": "https://localhost.invalid/cb"},
                "cues": [{"id": f"demo-m{n}", "shots": [n, n], "emotionalArc": "demo arc",
                          "payload": {"title": f"Demo {n}", "duration": 30,
                                      "negativeTags": "demo", "style": "demo cue"}}
                         for n in (1, 2)]}), encoding="utf-8")
            manifest = project / "work/score-v1/execution-manifest.json"
            with self.assertRaisesRegex(score.PlanError, "credits-per-request"):
                score.freeze(draft, manifest, credits_per_request=None,
                             price_verified_at="2026-01-01T00:00:00Z",
                             price_evidence="offline fixture", authority_path=authority)
            result = score.freeze(draft, manifest, credits_per_request=12,
                                  price_verified_at="2026-01-01T00:00:00Z",
                                  price_evidence="offline fixture; not a provider quote",
                                  authority_path=authority)
            self.assertEqual((result["requests"], result["expectedCredits"], result["sourceSeconds"]),
                             (2, 24, 60))
            self.assertFalse(result["financialAuthorized"])
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertFalse(doc["financialApproval"])
            self.assertEqual(doc["sourceDraft"], "creative-draft.json")
            self.assertEqual(doc["authority"], "../../story/GOAL.md")
            self.assertEqual(doc["cues"][0]["payload"]["model"], "V6")
            with self.assertRaisesRegex(score.PlanError, "overwrite a frozen manifest"):
                score.freeze(draft, manifest, credits_per_request=12,
                             price_verified_at="2026-01-01T00:00:00Z",
                             price_evidence="offline fixture", authority_path=authority)


if __name__ == "__main__":
    unittest.main()
