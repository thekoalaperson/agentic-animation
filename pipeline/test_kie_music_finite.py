import contextlib
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from pipeline import kie_music_finite as finite
from pipeline.test_paid_approval import write_approval


class FiniteKieMusicTest(unittest.TestCase):
    def fixture(self, root: Path, approved=False, cue_count=2, legacy=False, cues=None,
                extra=None):
        """A version 2 V6 manifest polled without a callback; legacy=True is a recorded v1."""
        if legacy:
            common = {
                "model": "V5_5", "customMode": True, "instrumental": True,
                "prompt": "", "callBackUrl": "https://localhost.invalid/suno-callback"
            }
            per_cue = {"negativeTags": "vocals, lyrics"}
        else:
            common = {"model": "V6", "customMode": True, "instrumental": True}
            per_cue = {}
        if cues is not None:
            common = {}
        else:
            cues = []
            for index in range(1, cue_count + 1):
                cues.append({
                    "id": f"m{index}-test",
                    "shots": [index, index],
                    "emotionalArc": "test arc",
                    "payload": {
                        **common, "title": f"Test {index}", "duration": 10, **per_cue,
                        "style": "A distinct instrumental test cue."
                    }
                })
        cue_count = len(cues)
        source = root / "creative-draft.json"
        source_cues = [
            {**cue, "payload": {key: value for key, value in cue["payload"].items()
                                if key not in common}}
            for cue in cues
        ]
        source.write_text(json.dumps({"commonPayload": common, "cues": source_cues}),
                          encoding="utf-8")
        manifest = root / "execution-manifest.json"
        doc = {
            "version": 1 if legacy else 2, "episode": "EPXX", "financialApproval": approved,
            "currentPriceVerified": True,
            "sourceDraft": "creative-draft.json",
            "sourceDraftSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "creditsPerRequest": 12, "maxRequests": cue_count,
            "maxCredits": cue_count * 12, "outputDirectory": "sources",
            "expectedCandidatesPerRequest": 2,
            "cues": copy.deepcopy(cues)
        }
        doc.update(extra or {})
        manifest.write_text(json.dumps(doc), encoding="utf-8")
        self.approval_path = self.approve(manifest) if approved else None
        return manifest

    @staticmethod
    def approve(manifest: Path) -> Path:
        doc = json.loads(manifest.read_text())
        plan = finite.load_plan(manifest, doc["maxRequests"], doc["maxCredits"])
        return write_approval(manifest.parent / "approval.json", episode=doc["episode"],
                              lane=finite.APPROVAL_LANE, manifest=manifest,
                              scope=finite.approval_scope(plan))

    @staticmethod
    def state_for(manifest: Path) -> Path:
        return manifest.with_suffix(".state.json")

    def test_authority_and_creative_bindings_are_checked_by_the_dry_run(self):
        for changed in ("GOAL.md", "timing.json"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = self.fixture(root)
                for name in ("GOAL.md", "timing.json"):
                    (root / name).write_text("demo\n", encoding="utf-8")
                digest = hashlib.sha256(b"demo\n").hexdigest()
                doc = json.loads(manifest.read_text())
                doc.update(authority="GOAL.md", authoritySha256=digest,
                           creativeBindings=[{"path": "timing.json", "sha256": digest}])
                manifest.write_text(json.dumps(doc), encoding="utf-8")
                self.assertEqual(finite.execute(manifest, 2, 24)["operation"], "DRY_RUN")
                (root / changed).write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(finite.PlanError, "changed after freezing|authority changed"):
                    finite.execute(manifest, 2, 24)

    def test_gpu_wrapper_imports_shared_receipt_helpers(self):
        wrapper = importlib.import_module("run_score_gpu")
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            wrapper._atomic_json(receipt, {"topUpAttempted": False})
            self.assertEqual(json.loads(receipt.read_text()), {"topUpAttempted": False})
            self.assertEqual(wrapper._sha_file(receipt),
                             hashlib.sha256(receipt.read_bytes()).hexdigest())

    def test_schema_covers_a_valid_manifest(self):
        shown = finite.schema()
        for legacy in (True, False):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                doc = json.loads(self.fixture(Path(tmp), legacy=legacy).read_text())
                self.assertLessEqual(set(doc), set(shown["requiredRootFields"])
                                     | set(shown["optionalRootFields"]))
                self.assertLessEqual(set(shown["requiredRootFields"]), set(doc) | {"episode"})
                self.assertEqual(set(doc["cues"][0]), set(shown["cueFields"]))
                payload = set(doc["cues"][0]["payload"])
                if legacy:
                    self.assertEqual(payload, set(shown["payloadFields"]))
                else:
                    route = shown["payloadRoutesV2"]["generate-v6"]
                    self.assertLessEqual(set(route["required"]), payload)
                    self.assertLessEqual(payload, set(route["required"]) | set(route["optional"]))

    def test_default_dry_run_is_zero_network_and_writes_no_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            with mock.patch.object(finite, "_create") as create, \
                 mock.patch.object(finite, "_record") as record, \
                 mock.patch.object(finite, "_download") as download:
                result = finite.execute(manifest, 2, 24)
            self.assertEqual(result["operation"], "DRY_RUN")
            self.assertEqual(result["networkCalls"], 0)
            create.assert_not_called()
            record.assert_not_called()
            download.assert_not_called()
            self.assertFalse(manifest.with_suffix(".state.json").exists())

    def test_both_request_and_credit_caps_are_hard(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(Path(tmp))
            with self.assertRaises(finite.PlanError):
                finite.execute(manifest, 1, 24)
            with self.assertRaises(finite.PlanError):
                finite.execute(manifest, 2, 23)
            with self.assertRaises(finite.PlanError):
                finite.execute(manifest, 3, 24)
            with self.assertRaises(finite.PlanError):
                finite.execute(manifest, 2, 25)

    def test_financial_gate_blocks_before_network(self):
        for boolean in (False, True):
            with self.subTest(financialApproval=boolean), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = self.fixture(root, approved=False)
                doc = json.loads(manifest.read_text())
                doc["financialApproval"] = boolean
                manifest.write_text(json.dumps(doc), encoding="utf-8")
                with mock.patch.object(finite, "_create") as create:
                    with self.assertRaisesRegex(finite.PlanError, "requires a current approval record"):
                        finite.execute(manifest, 2, 24, "submit", self.state_for(manifest),
                                       create_fn=create, api_key="offline-placeholder")
                create.assert_not_called()
                self.assertFalse(self.state_for(manifest).exists())
                # Recorded task IDs stay pollable without any new record.
                state = finite._state_for(finite.load_plan(manifest, 2, 24))
                for item in state["items"].values():
                    item.update(status="SUBMITTED", taskId="recorded-task")
                finite._atomic_json(self.state_for(manifest), state)
                polled = finite.execute(manifest, 2, 24, "poll",
                                        record_fn=lambda task, key: {"status": "PENDING"},
                                        api_key="offline-placeholder")
                self.assertEqual(polled["recordGets"], 2)

    @staticmethod
    def tone() -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            tone = Path(tmp) / "tone.mp3"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                            "sine=frequency=330:duration=6.0", "-c:a", "libmp3lame",
                            str(tone)], check=True)
            return tone.read_bytes()

    @staticmethod
    def record(task_id, key):
        return {"status": "SUCCESS", "response": {"sunoData": [
            {"audioUrl": f"https://invalid/{task_id}/take1", "id": "a"},
            {"audioUrl": f"https://invalid/{task_id}/take2", "id": "b"}]}}

    def test_mock_submit_posts_one_cue_first_then_the_rest_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root, approved=True, cue_count=3)
            calls = []

            def create(payload, key):
                calls.append((payload, key))
                return {"taskId": f"task-{len(calls)}"}

            state = self.state_for(manifest)

            def submit():
                return finite.execute(manifest, 3, 36, "submit", state, create_fn=create,
                                      api_key="offline-placeholder",
                                      approval_admission=self.approval_path)

            first = submit()
            self.assertEqual((first["generationPosts"], first["stillPlanned"]), (1, 2))
            # While the first cue is in flight, nothing more is posted.
            self.assertEqual(submit()["generationPosts"], 0)
            raw = self.tone()
            polled = finite.execute(manifest, 3, 36, "poll", state, record_fn=self.record,
                                    download_fn=lambda url: raw, api_key="offline-placeholder",
                                    approval_admission=self.approval_path)
            self.assertEqual((polled["recordGets"], polled["notYetPosted"]), (1, 2))
            rest = submit()
            self.assertEqual((rest["generationPosts"], rest["stillPlanned"]), (2, 0))
            self.assertEqual(submit()["generationPosts"], 0)
            self.assertEqual([payload["title"] for payload, _ in calls],
                             ["Test 1", "Test 2", "Test 3"])

    def test_gpu_wrapper_waits_for_the_first_cue_and_keeps_a_legacy_receipt(self):
        wrapper = importlib.import_module("run_score_gpu")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root, cue_count=2)
            (root / "GOAL.md").write_text("demo\n", encoding="utf-8")
            doc = json.loads(manifest.read_text())
            doc.update(authority="GOAL.md", authoritySha256=hashlib.sha256(b"demo\n").hexdigest())
            manifest.write_text(json.dumps(doc), encoding="utf-8")
            approval = self.approve(manifest)
            legacy = {"status": "COMPLETED", "walletAfter": 70.0}
            (root / "execution-receipt.json").write_text(json.dumps(legacy))
            raw, events, wallet = self.tone(), [], iter([100.0, 88.0, 76.0])

            def create(payload, key):
                events.append(("post", payload["title"]))
                return {"taskId": f"task-{len(events)}"}

            def real_execute(*args, **kwargs):
                events.append(("execute", args[3]))
                return finite.execute(*args, **kwargs, create_fn=create, record_fn=self.record,
                                      download_fn=lambda url: raw)

            def urlopen(request, timeout):
                events.append(("wallet",))
                return io.BytesIO(json.dumps({"code": 200, "data": next(wallet)}).encode())

            argv = ["run_score_gpu.py", str(manifest), "--max-requests", "2",
                    "--max-credits", "24", "--approval-admission", str(approval)]
            with mock.patch.object(wrapper, "platform") as host, \
                 mock.patch.object(wrapper, "subprocess"), \
                 mock.patch.object(wrapper, "execute", side_effect=real_execute), \
                 mock.patch.object(wrapper.urllib.request, "urlopen", side_effect=urlopen), \
                 mock.patch.object(wrapper.time, "sleep"), \
                 mock.patch.object(wrapper.sys, "argv", argv), \
                 mock.patch.object(wrapper.sys, "stdin", io.StringIO('{"kieApiKey": "offline"}')), \
                 contextlib.redirect_stdout(io.StringIO()):
                host.system.return_value, host.node.return_value = "Linux", "demo-host"
                wrapper.main()
            self.assertEqual(events, [
                ("wallet",), ("execute", "submit"), ("post", "Test 1"), ("execute", "poll"),
                ("wallet",), ("execute", "submit"), ("post", "Test 2"), ("execute", "poll"),
                ("wallet",)])
            index = json.loads((root / "execution-receipt.json").read_text())
            records = [attempt["record"] for attempt in index["attempts"]]
            self.assertEqual(records[0], {"legacyRecord": "execution-receipt.json", "record": legacy})
            self.assertEqual([r.get("status") for r in records[1:]], ["FIRST_CUE_LANDED", "COMPLETED"])
            self.assertEqual((records[1]["firstCues"], records[1]["observedCreditsPerLandedCue"],
                              records[1]["creditsPerRequest"]), (["m1-test"], 12.0, 12))
            self.assertEqual(index["status"], "COMPLETED")

    def test_mock_poll_preserves_every_returned_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root, approved=True, cue_count=1)
            state = self.state_for(manifest)
            finite.execute(manifest, 1, 12, "submit", state,
                           create_fn=lambda payload, key: {"taskId": "task-1"},
                           api_key="offline-placeholder", approval_admission=self.approval_path)
            tone = root / "tone.mp3"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                            "sine=frequency=330:duration=6.0", "-c:a", "libmp3lame",
                            str(tone)], check=True)
            raw = tone.read_bytes()
            tone.unlink()

            def record(task_id, key):
                return {"status": "SUCCESS", "response": {"sunoData": [
                    {"audioUrl": "https://invalid/take1", "id": "a"},
                    {"audioUrl": "https://invalid/take2", "id": "b"}
                ]}}

            polled = finite.execute(manifest, 1, 12, "poll", state,
                                    record_fn=record, download_fn=lambda url: raw,
                                    api_key="offline-placeholder", approval_admission=self.approval_path)
            self.assertEqual(polled["recordGets"], 1)
            self.assertEqual(polled["candidateDownloads"], 2)
            saved = json.loads(state.read_text())
            item = saved["items"]["m1-test"]
            self.assertEqual(item["status"], "DOWNLOADED")
            self.assertEqual(len(item["candidates"]), 2)
            for candidate in item["candidates"]:
                self.assertEqual(candidate["status"], "COMPLETE")
                self.assertEqual(Path(candidate["file"]).read_bytes(), raw)
                self.assertTrue(Path(candidate["sidecar"]).is_file())

    def test_price_verification_blocks_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root, approved=True, cue_count=1)
            doc = json.loads(manifest.read_text())
            doc["currentPriceVerified"] = False
            manifest.write_text(json.dumps(doc), encoding="utf-8")
            self.approval_path = self.approve(manifest)
            with mock.patch.object(finite, "_create") as create:
                with self.assertRaisesRegex(finite.PlanError, "currentPriceVerified is false"):
                    finite.execute(manifest, 1, 12, "submit", api_key="offline-placeholder", approval_admission=self.approval_path)
            create.assert_not_called()
            self.assertFalse(self.state_for(manifest).exists())

    def test_intermediate_success_remains_pending_and_recovers_old_misclassification(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(Path(tmp), approved=True, cue_count=1)
            state = self.state_for(manifest)
            finite.execute(manifest, 1, 12, "submit", state,
                           create_fn=lambda payload, key: {"taskId": "existing-task"},
                           api_key="offline-placeholder", approval_admission=self.approval_path)
            for provider_status in ("TEXT_SUCCESS", "FIRST_SUCCESS"):
                result = finite.execute(manifest, 1, 12, "poll", state,
                    record_fn=lambda task, key: {"status": provider_status},
                    download_fn=lambda url: self.fail("partial audio must not download"),
                    api_key="offline-placeholder", approval_admission=self.approval_path)
                self.assertEqual(result["candidateDownloads"], 0)
                saved = json.loads(state.read_text())
                self.assertEqual(saved["items"]["m1-test"]["status"], "PENDING")
            saved["items"]["m1-test"]["status"] = "GENERATION_FAILED"
            state.write_text(json.dumps(saved))
            finite.execute(manifest, 1, 12, "poll", state,
                           record_fn=lambda task, key: {"status": "FIRST_SUCCESS"},
                           api_key="offline-placeholder", approval_admission=self.approval_path)
            fixed = json.loads(state.read_text())["items"]["m1-test"]
            self.assertEqual(fixed["status"], "PENDING")
            self.assertEqual(fixed["taskId"], "existing-task")
            self.assertEqual(fixed["postAttempts"], 1)

    def test_stale_output_namespace_blocks_all_posts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root, approved=True, cue_count=2)
            stale = root / "sources/m2-test-take-1.mp3"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")
            calls = []
            with self.assertRaisesRegex(finite.PlanError, "output namespace is not empty"):
                finite.execute(
                    manifest, 2, 24, "submit",
                    create_fn=lambda payload, key: calls.append(payload) or {"taskId": "never"},
                    api_key="offline-placeholder", approval_admission=self.approval_path,
                )
            self.assertEqual(calls, [])

    def test_custom_state_and_override_hash_drift_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root, approved=True, cue_count=1)
            with self.assertRaisesRegex(finite.PlanError, "custom state paths are forbidden"):
                finite.execute(manifest, 1, 12, "submit", root / "other-state.json",
                               api_key="offline-placeholder", approval_admission=self.approval_path)

            override = root / "override.json"
            override.write_text("{}", encoding="utf-8")
            doc = json.loads(manifest.read_text())
            doc.update({
                "provenancePathBase": "manifest",
                "productionOverrides": "override.json",
                "productionOverridesSha256": hashlib.sha256(override.read_bytes()).hexdigest(),
            })
            manifest.write_text(json.dumps(doc), encoding="utf-8")
            finite.load_plan(manifest, 1, 12)
            override.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(finite.PlanError, "changed after"):
                finite.load_plan(manifest, 1, 12)


def cue(cue_id, index, **payload):
    base = {"model": "V6", "customMode": True, "instrumental": True,
            "title": f"Demo {index}", "duration": 10, "style": "demo-cue style"}
    return {"id": cue_id, "shots": [index, index], "emotionalArc": "demo arc",
            "payload": {**base, **payload}}


def extend(cue_id, index, parent, take=1, continue_at=4.0, **payload):
    base = {"model": "V6", "instrumental": True, "style": "demo-cue style"}
    return {"id": cue_id, "shots": [index, index], "emotionalArc": "demo arc",
            "payload": {**base, **payload},
            "derivedFrom": {"cueId": parent, "take": take, "operation": "extend",
                            "continueAt": continue_at}}


class ScoreFixtures(unittest.TestCase):
    """The shared fixtures, without re-running FiniteKieMusicTest's own tests."""
    fixture = FiniteKieMusicTest.fixture
    approve = staticmethod(FiniteKieMusicTest.approve)
    state_for = staticmethod(FiniteKieMusicTest.state_for)
    tone = staticmethod(FiniteKieMusicTest.tone)
    record = staticmethod(FiniteKieMusicTest.record)


class ScorePayloadVersionTwoTest(ScoreFixtures):
    def plan(self, cues, **extra):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(Path(tmp), cues=cues, extra=extra)
            doc = json.loads(manifest.read_text())
            return finite.load_plan(manifest, doc["maxRequests"], doc["maxCredits"])

    def test_recorded_version_one_manifest_keeps_its_rule_and_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(Path(tmp), legacy=True)
            plan = finite.load_plan(manifest, 2, 24)
            self.assertEqual(plan["expectedCredits"], 24)
            self.assertIs(finite.record_route(plan), finite._record)
            self.assertEqual(finite.approval_scope(plan),
                             {"requests": 2, "credits": 24, "sourceSeconds": 20})
            # What version 1 refused it still refuses, even where version 2 allows it.
            for change in ({"instrumental": False, "prompt": "demo words"},
                           {"negativeTags": ""}, {"model": "V6_MINI"}, {"variety": 2}):
                doc = json.loads(manifest.read_text())
                for frozen in doc["cues"]:
                    frozen["payload"].update(change)
                source = json.loads((Path(tmp) / "creative-draft.json").read_text())
                for draft in source["cues"]:
                    draft["payload"].update(change)
                (Path(tmp) / "creative-draft.json").write_text(json.dumps(source))
                doc["sourceDraftSha256"] = hashlib.sha256(
                    (Path(tmp) / "creative-draft.json").read_bytes()).hexdigest()
                broken = Path(tmp) / "broken.json"
                broken.write_text(json.dumps(doc))
                with self.subTest(change=change), self.assertRaises(finite.PlanError):
                    finite.load_plan(broken, 2, 24)
        posts = []
        with mock.patch.object(finite, "_request_json",
                               side_effect=lambda *a: posts.append(a) or {"taskId": "t"}):
            legacy = {"model": "V5_5", "customMode": True, "instrumental": True, "prompt": "",
                      "callBackUrl": "https://localhost.invalid/cb", "title": "T",
                      "duration": 10, "negativeTags": "x", "style": "s"}
            finite._create(legacy, "offline")
            finite._create({**legacy, "model": "V6"}, "offline")
        self.assertEqual(posts[0][1], finite.API_ROOT + "generate")
        self.assertEqual(posts[0][3], legacy)
        # A recorded V6 manifest posts the same body it always did.
        self.assertEqual(posts[1][3], {
            "model": "ai-music-api/generate", "callBackUrl": "https://localhost.invalid/cb",
            "input": {"model": "V6", "custom_mode": True, "instrumental": True, "prompt": "",
                      "title": "T", "duration": 10, "negative_tags": "x", "style": "s"}})

    def test_every_v6_family_model_polls_the_jobs_record(self):
        for model in ("V6", "V6_MINI", "V6_WILD"):
            with self.subTest(model=model):
                plan = self.plan([cue("demo-m1", 1, model=model)])
                self.assertIs(finite.record_route(plan), finite._record_v6)

    def test_callback_is_sent_only_when_frozen_and_knobs_map_to_provider_names(self):
        body = finite.provider_body({"model": "V6_WILD", "customMode": True, "instrumental": False,
                                     "prompt": "demo words", "title": "T", "style": "s",
                                     "styleWeight": 0.4, "weirdnessConstraint": 0.75,
                                     "variety": 3, "vocalGender": "f", "personaId": "demo-p"})
        self.assertNotIn("callBackUrl", body)
        self.assertEqual(body["model"], "ai-music-api/generate")
        self.assertEqual(body["input"], {
            "model": "V6_WILD", "custom_mode": True, "instrumental": False,
            "prompt": "demo words", "title": "T", "style": "s", "style_weight": 0.4,
            "weirdness_constraint": 0.75, "variety": 3, "vocal_gender": "f",
            "persona_id": "demo-p"})

    def test_optional_ranges_are_frozen_exactly_and_unknown_keys_refused(self):
        plan = self.plan([cue("demo-m1", 1, variety=4, styleWeight=0.35, weirdnessConstraint=0,
                              negativeTags="demo-tag", callBackUrl="https://localhost.invalid/cb",
                              personaModel="style_persona", personaId="demo-persona")])
        self.assertEqual(plan["cues"][0]["payload"]["variety"], 4)
        refused = [{"variety": 5}, {"variety": True}, {"styleWeight": 0.333},
                   {"weirdnessConstraint": 1.5}, {"stylWeight": 0.2}, {"negativeTags": ""},
                   {"vocalGender": "f"}, {"audioWeight": 0.5}, {"personaModel": "other"},
                   {"callBackUrl": "http://localhost.invalid/cb"}, {"customMode": False},
                   {"model": "V7"}, {"duration": 400}, {"title": "x" * 81}]
        for change in refused:
            with self.subTest(change=change), self.assertRaises(finite.PlanError):
                self.plan([cue("demo-m1", 1, **change)])

    def test_a_sung_cue_freezes_its_lyrics_and_an_instrumental_carries_none(self):
        plan = self.plan([cue("demo-m1", 1, instrumental=False, prompt="demo-line one\ndemo-line two",
                              vocalGender="m")])
        self.assertEqual(plan["cues"][0]["payload"]["prompt"], "demo-line one\ndemo-line two")
        self.plan([cue("demo-m1", 1, prompt="")])
        for change in ({"instrumental": False}, {"instrumental": False, "prompt": " "},
                       {"prompt": "demo words"}, {"instrumental": "no"}):
            with self.subTest(change=change), self.assertRaises(finite.PlanError):
                self.plan([cue("demo-m1", 1, **change)])

    def test_freeze_writes_version_two_and_prices_extends_on_their_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            authority = root / "GOAL.md"
            authority.write_text("demo\n", encoding="utf-8")
            draft = root / "draft.json"
            draft.write_text(json.dumps({"episode": "DEMO",
                                         "commonPayload": {"model": "V6", "instrumental": True},
                                         "cues": [cue("demo-m1", 1), extend("demo-m2", 2, "demo-m1")]}))
            kwargs = dict(credits_per_request=12, price_verified_at="2026-01-01T00:00:00Z",
                          price_evidence="offline fixture; not a provider quote",
                          authority_path=authority)
            target = root / "score/execution-manifest.json"
            with self.assertRaisesRegex(finite.PlanError, "credits-per-extend-request"):
                finite.freeze(draft, target, **kwargs)
            result = finite.freeze(draft, target, credits_per_extend_request=7, **kwargs)
            doc = json.loads(target.read_text())
            self.assertEqual((doc["version"], doc["maxCredits"], doc["creditsPerExtendRequest"]),
                             (2, 19, 7))
            self.assertEqual((result["requests"], result["expectedCredits"], result["sourceSeconds"]),
                             (2, 19, 10))


class DerivedCueTest(ScoreFixtures):
    def derived_fixture(self, root, continue_at=4.0, extra=None):
        cues = [cue("demo-m1", 1), cue("demo-m2", 2),
                extend("demo-m3", 3, "demo-m2", continue_at=continue_at)]
        manifest = self.fixture(root, cues=cues, extra={"creditsPerExtendRequest": 7,
                                                        "maxCredits": 31, **(extra or {})})
        self.approval_path = self.approve(manifest)
        return manifest

    def run_op(self, manifest, operation, **kwargs):
        return finite.execute(manifest, 3, 31, operation, api_key="offline-placeholder",
                              approval_admission=self.approval_path, **kwargs)

    def test_extend_contract_is_checked_when_the_plan_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = finite.load_plan(self.derived_fixture(Path(tmp)), 3, 31)
            self.assertEqual((plan["requestCount"], plan["expectedCredits"]), (3, 31))
            self.assertEqual(finite.approval_scope(plan)["sourceSeconds"], 20)
        refused = {
            "parent later": [cue("demo-m1", 1), extend("demo-m2", 2, "demo-m3"), cue("demo-m3", 3)],
            "other model": [cue("demo-m1", 1), extend("demo-m2", 2, "demo-m1", model="V6_WILD")],
            "duration": [cue("demo-m1", 1), extend("demo-m2", 2, "demo-m1", duration=20)],
            "customMode": [cue("demo-m1", 1), extend("demo-m2", 2, "demo-m1", customMode=True)],
            "take": [cue("demo-m1", 1), extend("demo-m2", 2, "demo-m1", take=3)],
            "continueAt": [cue("demo-m1", 1), extend("demo-m2", 2, "demo-m1", continue_at=0)],
            "instrumental by omission": [cue("demo-m1", 1), extend("demo-m2", 2, "demo-m1")],
        }
        del refused["instrumental by omission"][1]["payload"]["instrumental"]
        for label, cues in refused.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                manifest = self.fixture(Path(tmp), cues=cues,
                                        extra={"creditsPerExtendRequest": 7, "maxCredits": 99})
                with self.assertRaises(finite.PlanError):
                    finite.load_plan(manifest, len(cues), 99)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(Path(tmp), cues=[cue("demo-m1", 1), extend("demo-m2", 2, "demo-m1")],
                                    extra={"maxCredits": 99})
            with self.assertRaisesRegex(finite.PlanError, "creditsPerExtendRequest"):
                finite.load_plan(manifest, 2, 99)

    def test_a_waiting_cue_is_posted_once_after_its_parent_lands(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.derived_fixture(Path(tmp))
            posts, raw = [], self.tone()

            def create(payload, key):
                posts.append(payload)
                return {"taskId": f"task-{len(posts)}"}

            def submit():
                return self.run_op(manifest, "submit", create_fn=create)

            def poll():
                return self.run_op(manifest, "poll", record_fn=self.record,
                                   download_fn=lambda url: raw)

            self.assertEqual(submit()["generationPosts"], 1)        # the first cue alone
            poll()
            rest = submit()                                          # demo-m2 posts; demo-m3 waits
            self.assertEqual((rest["generationPosts"], rest["waitingParent"]), (1, ["demo-m3"]))
            for _ in range(2):                                       # parent still in flight
                self.assertEqual(submit()["generationPosts"], 0)
            self.assertEqual(poll()["notYetPosted"], 1)
            state = json.loads(self.state_for(manifest).read_text())
            self.assertEqual(state["items"]["demo-m3"]["status"], "WAITING_PARENT")
            self.assertEqual(submit()["generationPosts"], 1)        # the extend, once
            self.assertEqual(submit()["generationPosts"], 0)
            self.assertEqual([p.get("audioId") for p in posts], [None, None, "a"])
            self.assertEqual(posts[2]["continueAt"], 4.0)
            self.assertEqual(finite.provider_body(posts[2])["model"], "ai-music-api/extend")
            self.assertEqual(finite.provider_body(posts[2])["input"]["audio_id"], "a")
            poll()
            item = json.loads(self.state_for(manifest).read_text())["items"]["demo-m3"]
            self.assertEqual(item["status"], "DOWNLOADED")
            parent = json.loads(self.state_for(manifest).read_text())["items"]["demo-m2"]
            sidecar = json.loads(Path(item["candidates"][0]["sidecar"]).read_text())
            self.assertEqual(sidecar["derivedFrom"]["sidecarSha256"],
                             parent["candidates"][0]["sidecarSha256"])
            self.assertEqual((sidecar["derivedFrom"]["audioId"], sidecar["derivedFrom"]["taskId"]),
                             ("a", "task-2"))
            self.assertEqual(sidecar["requestSha256"], item["requestSha256"])

    def test_an_interrupted_extend_is_never_posted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.derived_fixture(Path(tmp))
            raw = self.tone()
            posts = []

            def create(payload, key):
                posts.append(payload)
                if payload.get("audioId"):
                    raise SystemExit("process killed mid-POST")
                return {"taskId": f"task-{len(posts)}"}

            for _ in range(2):
                self.run_op(manifest, "submit", create_fn=create)
                self.run_op(manifest, "poll", record_fn=self.record, download_fn=lambda url: raw)
            with self.assertRaises(SystemExit):
                self.run_op(manifest, "submit", create_fn=create)
            with self.assertRaisesRegex(finite.UncertainRequest, "will not be retried"):
                self.run_op(manifest, "submit", create_fn=create)
            self.assertEqual(sum(bool(p.get("audioId")) for p in posts), 1)

    def test_a_synthesized_parent_id_or_a_point_past_its_end_is_refused_before_posting(self):
        synthesized = [
            lambda task, key: {"status": "SUCCESS", "response": {"sunoData": [
                {"audioUrl": "https://invalid/1", "id": f"{task}-1", "idSynthesized": True},
                {"audioUrl": "https://invalid/2", "id": f"{task}-2", "idSynthesized": True}]}},
            lambda task, key: {"status": "SUCCESS", "response": {"sunoData": [
                {"audioUrl": "https://invalid/1", "id": f"{task}-1"},
                {"audioUrl": "https://invalid/2", "id": f"{task}-2"}]}},
        ]
        cases = [(record, 4.0, "synthesized") for record in synthesized]
        cases.append((self.record, 9.0, "outside the parent take"))
        for record, continue_at, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                manifest = self.derived_fixture(Path(tmp), continue_at=continue_at)
                raw, posts = self.tone(), []

                def create(payload, key):
                    posts.append(payload)
                    return {"taskId": f"task-{len(posts)}"}

                for _ in range(2):
                    self.run_op(manifest, "submit", create_fn=create)
                    self.run_op(manifest, "poll", record_fn=record, download_fn=lambda url: raw)
                with self.assertRaisesRegex(finite.PlanError, message):
                    self.run_op(manifest, "submit", create_fn=create)
                self.assertEqual(len(posts), 2)

    def test_gpu_wrapper_posts_waves_until_no_cue_waits(self):
        wrapper = importlib.import_module("run_score_gpu")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "GOAL.md").write_text("demo\n", encoding="utf-8")
            manifest = self.derived_fixture(root, extra={
                "authority": "GOAL.md", "authoritySha256": hashlib.sha256(b"demo\n").hexdigest()})
            raw, events, wallet = self.tone(), [], iter([100.0, 88.0, 76.0, 69.0])

            def create(payload, key):
                events.append(("post", payload["title"] if "title" in payload else payload["audioId"]))
                return {"taskId": f"task-{len(events)}"}

            def real_execute(*args, **kwargs):
                return finite.execute(*args, **kwargs, create_fn=create, record_fn=self.record,
                                      download_fn=lambda url: raw)

            def urlopen(request, timeout):
                events.append(("wallet",))
                return io.BytesIO(json.dumps({"code": 200, "data": next(wallet)}).encode())

            argv = ["run_score_gpu.py", str(manifest), "--max-requests", "3",
                    "--max-credits", "31", "--approval-admission", str(self.approval_path)]
            with mock.patch.object(wrapper, "platform") as host, \
                 mock.patch.object(wrapper, "subprocess"), \
                 mock.patch.object(wrapper, "execute", side_effect=real_execute), \
                 mock.patch.object(wrapper.urllib.request, "urlopen", side_effect=urlopen), \
                 mock.patch.object(wrapper.time, "sleep"), \
                 mock.patch.object(wrapper.sys, "argv", argv), \
                 mock.patch.object(wrapper.sys, "stdin", io.StringIO('{"kieApiKey": "offline"}')), \
                 contextlib.redirect_stdout(io.StringIO()):
                host.system.return_value, host.node.return_value = "Linux", "demo-host"
                wrapper.main()
            self.assertEqual(events, [("wallet",), ("post", "Demo 1"), ("wallet",), ("post", "Demo 2"),
                                      ("wallet",), ("post", "a"), ("wallet",)])
            index = json.loads((root / "execution-receipt.json").read_text())
            self.assertEqual(index["status"], "COMPLETED")

    def test_v6_resulturls_fallback_marks_its_ids_synthesized(self):
        raw = {"state": "success", "resultJson": json.dumps({"resultUrls": ["https://invalid/a"]})}
        with mock.patch.object(finite, "_request_json", return_value=raw):
            record = finite._record_v6("task-9", "offline")
        self.assertEqual(record["response"]["sunoData"],
                         [{"audioUrl": "https://invalid/a", "id": "task-9-1", "idSynthesized": True}])


if __name__ == "__main__":
    unittest.main()
