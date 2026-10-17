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
    def fixture(self, root: Path, approved=False, cue_count=2):
        common = {
            "model": "V5_5", "customMode": True, "instrumental": True,
            "prompt": "", "callBackUrl": "https://localhost.invalid/suno-callback"
        }
        cues = []
        for index in range(1, cue_count + 1):
            cues.append({
                "id": f"m{index}-test",
                "shots": [index, index],
                "emotionalArc": "test arc",
                "payload": {
                    **common, "title": f"Test {index}", "duration": 10,
                    "negativeTags": "vocals, lyrics", "style": "A distinct instrumental test cue."
                }
            })
        source = root / "creative-draft.json"
        source_cues = [
            {**cue, "payload": {key: value for key, value in cue["payload"].items()
                                if key not in common}}
            for cue in cues
        ]
        source.write_text(json.dumps({"commonPayload": common, "cues": source_cues}),
                          encoding="utf-8")
        manifest = root / "execution-manifest.json"
        manifest.write_text(json.dumps({
            "version": 1, "episode": "EPXX", "financialApproval": approved,
            "currentPriceVerified": True,
            "sourceDraft": "creative-draft.json",
            "sourceDraftSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "creditsPerRequest": 12, "maxRequests": cue_count,
            "maxCredits": cue_count * 12, "outputDirectory": "sources",
            "expectedCandidatesPerRequest": 2,
            "cues": copy.deepcopy(cues)
        }), encoding="utf-8")
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
        with tempfile.TemporaryDirectory() as tmp:
            doc = json.loads(self.fixture(Path(tmp)).read_text())
        shown = finite.schema()
        self.assertLessEqual(set(doc), set(shown["requiredRootFields"]) | set(shown["optionalRootFields"]))
        self.assertLessEqual(set(shown["requiredRootFields"]), set(doc) | {"episode"})
        self.assertEqual(set(doc["cues"][0]), set(shown["cueFields"]))
        self.assertEqual(set(doc["cues"][0]["payload"]), set(shown["payloadFields"]))

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


if __name__ == "__main__":
    unittest.main()
