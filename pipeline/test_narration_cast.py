"""Version 2 narration manifests: a cast of voices in one manifest and one approval."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pipeline import eleven_batch_finite as finite
from pipeline.test_paid_approval import write_approval


SETTINGS = {"stability": 0.3, "similarity_boost": 0.6, "style": 0.1,
            "use_speaker_boost": False, "speed": 1.05}
OTHER = {"stability": 0.7, "similarity_boost": 0.5, "style": 0.4,
         "use_speaker_boost": True, "speed": 0.8}


def write(path: Path, doc: dict) -> Path:
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def cast_manifest(root: Path, **changes) -> Path:
    items = [
        {"id": "demo-s001", "text": "पहली पंक्ति।", "output": "audio/demo-s001.mp3"},
        {"id": "demo-s002", "speaker": "demo-teller", "text": "दूसरी पंक्ति।",
         "output": "audio/demo-s002.mp3", "seed": 7},
        {"id": "demo-s003", "speaker": "demo-child", "text": "तीसरी।",
         "output": "audio/demo-s003.mp3", "voiceSettings": OTHER},
    ]
    doc = {
        "version": 2, "episode": "DEMO", "financialApproval": False, "language": "hi",
        "outputFormat": "mp3_44100_192",
        "voiceId": "demoRootVoice", "modelId": "eleven_multilingual_v2", "voiceSettings": SETTINGS,
        "cast": {"demo-teller": {"voiceId": "demoTellerVoice", "modelId": "eleven_v4",
                                 "voiceSettings": SETTINGS},
                 "demo-child": {"voiceId": "demoChildVoice", "modelId": "eleven_multilingual_v2",
                                "voiceSettings": SETTINGS}},
        "pronunciationDictionaryLocators": [
            {"pronunciationDictionaryId": "demoDict", "versionId": "demoDictV1"}],
        "items": items,
    }
    doc.update(changes)
    doc["aggregateCharacters"] = sum(len(row["text"]) for row in doc["items"])
    return write(root / "narration.json", doc)


def cap(manifest: Path) -> int:
    return json.loads(manifest.read_text())["aggregateCharacters"]


class VersionOneUnchangedTest(unittest.TestCase):
    def manifest(self, root: Path, **changes) -> Path:
        doc = {"version": 1, "episode": "DEMO", "financialApproval": False, "language": "hi",
               "voiceId": "demoVoice", "modelId": "eleven_multilingual_v2",
               "outputFormat": "mp3_44100_192", "voiceSettings": SETTINGS,
               "aggregateCharacters": len("एक पंक्ति।"),
               "items": [{"id": "a", "text": "एक पंक्ति।", "output": "a.mp3"}]}
        doc.update(changes)
        return write(root / "narration.json", doc)

    def test_config_body_and_url_keep_their_version_one_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = finite.load_plan(self.manifest(Path(tmp)), len("एक पंक्ति।"))
        # Recorded state binds configSha256, so this dict must never change.
        self.assertEqual(plan["config"], {"language": "hi", "voiceId": "demoVoice",
                                          "modelId": "eleven_multilingual_v2",
                                          "outputFormat": "mp3_44100_192",
                                          "voiceSettings": SETTINGS})
        self.assertEqual(plan["configSha256"], finite._canonical_sha(plan["config"]))
        item = plan["items"][0]
        self.assertEqual(set(item), {"id", "text", "textFile", "output", "characters"})
        self.assertEqual(finite.request_body(plan, item),
                         {"text": "एक पंक्ति।", "model_id": "eleven_multilingual_v2",
                          "voice_settings": SETTINGS})
        self.assertEqual(finite.request_url(plan, item),
                         finite.API_ROOT + "/demoVoice?output_format=mp3_44100_192")

    def test_version_one_refuses_version_two_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(finite.PlanError, "version 2"):
                finite.load_plan(self.manifest(root, cast={}), len("एक पंक्ति।"))
            items = [{"id": "a", "text": "एक पंक्ति।", "output": "a.mp3", "speaker": "x"}]
            with self.assertRaisesRegex(finite.PlanError, "unsupported per-item fields"):
                finite.load_plan(self.manifest(root, items=items), len("एक पंक्ति।"))
            with self.assertRaisesRegex(finite.PlanError, "Hindi-only"):
                finite.load_plan(self.manifest(root, language="sa"), len("एक पंक्ति।"))


class CastManifestTest(unittest.TestCase):
    def test_each_item_is_sent_with_its_own_frozen_voice(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = finite.load_plan(cast_manifest(Path(tmp)), cap(cast_manifest(Path(tmp))))
        root, teller, child = plan["items"]
        self.assertEqual(finite.request_url(plan, root).split("?")[0], finite.API_ROOT + "/demoRootVoice")
        self.assertEqual(finite.request_url(plan, teller).split("?")[0], finite.API_ROOT + "/demoTellerVoice")
        locators = [{"pronunciation_dictionary_id": "demoDict", "version_id": "demoDictV1"}]
        # multilingual_v2 is documented not to take language_code; eleven_v4 does.
        self.assertEqual(finite.request_body(plan, root),
                         {"text": "पहली पंक्ति।", "model_id": "eleven_multilingual_v2",
                          "voice_settings": SETTINGS,
                          "pronunciation_dictionary_locators": locators})
        self.assertEqual(finite.request_body(plan, teller),
                         {"text": "दूसरी पंक्ति।", "model_id": "eleven_v4",
                          "voice_settings": SETTINGS, "language_code": "hi", "seed": 7,
                          "pronunciation_dictionary_locators": locators})
        self.assertEqual(finite.request_body(plan, child)["voice_settings"], OTHER)
        self.assertEqual([row["speaker"] for row in plan["voices"]],
                         [None, "demo-child", "demo-teller"])
        self.assertEqual(finite.approval_scope(plan), {"items": 3, "characters": plan["aggregateCharacters"]})

    def test_refusals_name_the_problem(self):
        cases = [
            ({"items": [{"id": "x", "speaker": "nobody", "text": "पाठ।", "output": "x.mp3"}]},
             "not in the cast"),
            ({"cast": {"demo-teller": {"voiceId": "v", "modelId": "eleven_unknown",
                                       "voiceSettings": SETTINGS}}}, "dated allowlist"),
            ({"cast": {"demo-teller": {"voiceId": "v", "modelId": "eleven_v4"}}},
             "exactly voiceId, modelId and voiceSettings"),
            ({"language": "Hindi"}, "ISO 639"),
            ({"pronunciationDictionaryLocators": [{"pronunciationDictionaryId": "d"}]},
             "versionId"),
            ({"previousText": True}, "does not know root fields"),
            ({"items": [{"id": "x", "speaker": "demo-teller", "text": "पाठ।", "output": "x.mp3",
                         "seed": -1}]}, "seed must be"),
        ]
        for changes, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                manifest = cast_manifest(Path(tmp), **changes)
                with self.assertRaisesRegex(finite.PlanError, message):
                    finite.load_plan(manifest, cap(manifest))
        with tempfile.TemporaryDirectory() as tmp:
            doc = json.loads(cast_manifest(Path(tmp)).read_text())
            for key in ("voiceId", "modelId", "voiceSettings"):
                doc.pop(key)
            manifest = write(Path(tmp) / "narration.json", doc)
            with self.assertRaisesRegex(finite.PlanError, "no root voice"):
                finite.load_plan(manifest, cap(manifest))

    def test_a_language_other_than_hindi_is_declared_not_assumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = cast_manifest(Path(tmp), language="sa")
            plan = finite.load_plan(manifest, cap(manifest))
        self.assertEqual(plan["config"]["language"], "sa")
        self.assertEqual(finite.request_body(plan, plan["items"][1])["language_code"], "sa")
        self.assertNotIn("language_code", finite.request_body(plan, plan["items"][0]))

    def test_dry_run_names_models_priced_or_run_only_by_assumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = cast_manifest(Path(tmp))
            report = finite.execute(manifest, cap(manifest), False)
        # The approval binds characters; eleven_v4 publishes no credit rate.
        self.assertEqual(report["unpublishedPricingModels"], ["eleven_v4"])
        self.assertEqual(report["unexercisedModels"], ["eleven_v4"])
        teller = report["items"][1]
        self.assertEqual((teller["modelId"], teller["creditsPerCharacter"], teller["exercised"]),
                         ("eleven_v4", None, False))
        self.assertEqual(report["items"][0]["creditsPerCharacter"], 1)
        for model, entry in finite.MODELS.items():
            self.assertIn("creditsPerCharacter", entry, model)

    def test_version_one_dry_run_keeps_its_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = VersionOneUnchangedTest().manifest(Path(tmp))
            report = finite.execute(manifest, len("एक पंक्ति।"), False)
        self.assertNotIn("unpublishedPricingModels", report)
        self.assertEqual(set(report["items"][0]), {"id", "characters", "output"})

    def test_mocked_cast_batch_posts_once_per_item_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tone.mp3"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                            "sine=frequency=330:duration=1.0", "-c:a", "libmp3lame", str(source)],
                           check=True)
            audio = source.read_bytes()
            source.unlink()
            manifest = cast_manifest(root)
            plan = finite.load_plan(manifest, cap(manifest))
            approval = write_approval(root / "approval.json", episode="DEMO",
                                      lane=finite.APPROVAL_LANE, manifest=manifest,
                                      scope=finite.approval_scope(plan))
            calls = []

            def fake_post(url, body, key):
                calls.append((url, body))
                return audio, {"status": 200, "requestId": "offline", "contentType": "audio/mpeg"}

            state = manifest.with_suffix(".eleven-state.json")
            first = finite.execute(manifest, cap(manifest), True, state_path=state,
                                   post_fn=fake_post, api_key="offline", approval_admission=approval)
            self.assertEqual(first["networkCalls"], 3)
            self.assertEqual([url.split("/")[-1].split("?")[0] for url, _ in calls],
                             ["demoRootVoice", "demoTellerVoice", "demoChildVoice"])
            recorded = json.loads(state.read_text())
            self.assertEqual(recorded["configSha256"], plan["configSha256"])
            self.assertEqual(recorded["items"]["demo-s002"]["requestSha256"],
                             finite._canonical_sha(calls[1][1]))
            self.assertEqual(hashlib.sha256((root / "audio/demo-s003.mp3").read_bytes()).hexdigest(),
                             hashlib.sha256(audio).hexdigest())
            again = finite.execute(manifest, cap(manifest), True, state_path=state,
                                   post_fn=fake_post, api_key="offline", approval_admission=approval)
            self.assertEqual((again["status"], len(calls)), ("ALREADY_COMPLETED", 3))


class CastFreezeTest(unittest.TestCase):
    def test_a_cast_freezes_version_two_from_a_speaker_named_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            draft = project / "work/intake/narration-plan.json"
            draft.parent.mkdir(parents=True)
            write(draft, {"episode": "DEMO", "model": "eleven_multilingual_v2", "voice": None,
                          "items": [
                              {"id": "demo-s001-hi", "speaker": "demo-teller", "text": "पहली।",
                               "output": "audio/n/demo-s001-hi.mp3"},
                              {"id": "demo-s002-hi", "text": "दूसरी।",
                               "output": "audio/n/demo-s002-hi.mp3"}]})
            cast = {"demo-teller": {"voiceId": "demoTellerVoice", "modelId": "eleven_v3",
                                    "voiceSettings": SETTINGS}}
            manifest = project / "work/narration-v2/manifest.json"
            with self.assertRaisesRegex(finite.PlanError, "no root voice"):
                finite.freeze(draft, manifest, project, voice_settings=None, cast=cast)
            self.assertFalse(manifest.exists())
            result = finite.freeze(draft, manifest, project, voice_settings=SETTINGS,
                                   voice_id="demoRootVoice", cast=cast)
            doc = json.loads(manifest.read_text())
            self.assertEqual((doc["version"], doc["financialApproval"]), (2, False))
            self.assertEqual(doc["cast"], cast)
            self.assertEqual(doc["items"][0]["speaker"], "demo-teller")
            self.assertNotIn("speaker", doc["items"][1])
            self.assertEqual([row["speaker"] for row in result["items"]], ["demo-teller", None])

    def test_no_cast_still_freezes_version_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            draft = write(project / "plan.json", {"episode": "DEMO", "voice": "demoVoice",
                "items": [{"id": "a", "speaker": "ignored", "text": "पाठ।", "output": "a.mp3"}]})
            finite.freeze(draft, project / "work/m.json", project, voice_settings=SETTINGS)
            doc = json.loads((project / "work/m.json").read_text())
        self.assertEqual(doc["version"], 1)
        self.assertEqual(set(doc["items"][0]), {"id", "text", "output"})


class PreparedPlanTest(unittest.TestCase):
    def test_a_voice_over_row_names_its_speaker_in_the_narration_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            (project / "story").mkdir()
            (project / "art").mkdir()
            rows, records = [], []
            for key, voice in (("demo-s001", None), ("demo-s002", "demo-teller")):
                still = project / f"art/{key}.png"
                still.write_bytes(b"demo still bytes " + key.encode())
                sidecar = project / f"art/{key}.imagegen.json"
                write(sidecar, {"target_sha256": hashlib.sha256(still.read_bytes()).hexdigest()})
                row = {"still_id": key, "speaker": "", "narration_hi": "पाठ।",
                       "narration_en": "demo text", "beat": 1}
                if voice:
                    row["voice"] = voice
                rows.append(row)
                records.append({"still_id": key, "target_output_path": f"art/{key}.png",
                                "provenance_sidecar": f"art/{key}.imagegen.json"})
            write(project / "story/storyboard_plan.json", {"stills": rows})
            write(project / "story/generation_manifest.json", {"episode": "DEMO", "stills": records})
            script = Path(finite.__file__).parent / "edit/prepare_episode_production.py"
            subprocess.run([sys.executable, str(script), str(project), "--inventory-only",
                            "--voice-id", "demoVoice"], check=True, capture_output=True)
            plan = json.loads((project / "work/production-intake-v1/narration-plan.json").read_text())
        self.assertNotIn("speaker", plan["items"][0])
        self.assertEqual(plan["items"][1]["speaker"], "demo-teller")


if __name__ == "__main__":
    unittest.main()
