import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.error

from pipeline import eleven_batch_finite as finite
from pipeline.paid_approval import validate_paid_approval
from pipeline.test_paid_approval import write_approval


class FiniteElevenBatchTest(unittest.TestCase):
    def manifest(self, root: Path, items=None, financial_approval=False) -> Path:
        path = root / "narration.json"
        final_items = items or [{
            "id": "part-01",
            "text": "यह केवल एक सुरक्षित परीक्षण वाक्य है।",
            "output": "audio/part-01.mp3"
        }]
        path.write_text(json.dumps({
            "version": 1,
            "episode": "DEMO",
            "financialApproval": financial_approval,
            "language": "hi",
            "voiceId": "testVoice_123",
            "modelId": "eleven_multilingual_v2",
            "outputFormat": "mp3_44100_192",
            "voiceSettings": {
                "stability": 0.6,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True,
                "speed": 0.9
            },
            "aggregateCharacters": sum(len(row["text"].strip()) for row in final_items),
            "items": final_items,
        }, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def approval(manifest: Path) -> Path:
        plan = finite.load_plan(manifest, json.loads(manifest.read_text())["aggregateCharacters"])
        return write_approval(manifest.parent / "approval.json", episode="DEMO",
                              lane=finite.APPROVAL_LANE, manifest=manifest,
                              scope=finite.approval_scope(plan))

    @staticmethod
    def state_for(manifest: Path) -> Path:
        return manifest.with_suffix(".eleven-state.json")

    @staticmethod
    def cap(manifest: Path) -> int:
        return json.loads(manifest.read_text())["aggregateCharacters"]

    def test_submit_is_refused_before_network_without_an_approval_record(self):
        for boolean in (False, True):
            with self.subTest(financialApproval=boolean), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = self.manifest(root, financial_approval=boolean)
                state_path = self.state_for(manifest)
                with mock.patch.object(finite, "_post_once") as post:
                    with self.assertRaisesRegex(finite.PlanError, "requires a current approval record"):
                        finite.execute(manifest, self.cap(manifest), submit=True,
                                       state_path=state_path,
                                       api_key="offline-placeholder")
                post.assert_not_called()
                self.assertFalse(state_path.exists())
                dry = finite.execute(manifest, self.cap(manifest), submit=False)
                self.assertEqual((dry["financialApproval"], dry["financialAuthorized"]),
                                 (boolean, False))

    def test_gpu_wrapper_imports_shared_receipt_helpers(self):
        wrapper = importlib.import_module("run_narration_gpu")
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            wrapper._atomic_json(receipt, {"noOverage": True})
            self.assertEqual(json.loads(receipt.read_text()), {"noOverage": True})
            self.assertEqual(wrapper._sha256_file(receipt),
                             hashlib.sha256(receipt.read_bytes()).hexdigest())

    def test_default_is_dry_run_with_no_network_or_state_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest(root)
            with mock.patch.object(finite, "_post_once") as post:
                result = finite.execute(manifest, self.cap(manifest), submit=False)
            self.assertEqual(result["mode"], "DRY_RUN")
            self.assertEqual(result["networkCalls"], 0)
            post.assert_not_called()
            self.assertFalse(manifest.with_suffix(".eleven-state.json").exists())

    def test_episode_relative_provenance_is_portable_and_contained(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Path(tmp) / 'episode'
            work = episode / 'work' / 'narration-v1'
            work.mkdir(parents=True)
            source = episode / 'story.json'
            source.write_text('{}')
            manifest = self.manifest(work)
            doc = json.loads(manifest.read_text())
            doc.update(episodeRoot='../..', provenancePathBase='episode',
                       sourcePlan='story.json', sourcePlanSha256=finite._sha256_file(source))
            manifest.write_text(json.dumps(doc, ensure_ascii=False))
            result = finite.load_plan(manifest, self.cap(manifest))
            self.assertEqual(result['sourceBindings'][0]['path'], str(source.resolve()))
            outside = Path(tmp) / 'outside.json'
            outside.write_text('{}')
            doc.update(sourcePlan='../outside.json', sourcePlanSha256=finite._sha256_file(outside))
            manifest.write_text(json.dumps(doc, ensure_ascii=False))
            with self.assertRaisesRegex(finite.PlanError, 'escapes'):
                finite.load_plan(manifest, self.cap(manifest))

    def test_cap_and_duplicate_outputs_are_refused_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest(root)
            with self.assertRaises(finite.PlanError):
                finite.execute(manifest, 3, submit=False)
            duplicate = self.manifest(root, [
                {"id": "a", "text": "हिंदी पाठ।", "output": "same.mp3"},
                {"id": "b", "text": "दूसरा पाठ।", "output": "same.mp3"}
            ])
            with self.assertRaises(finite.PlanError):
                finite.execute(duplicate, self.cap(duplicate), submit=False)
            case_variant = self.manifest(root, [
                {"id": "a", "text": "हिंदी पाठ।", "output": "audio/Part-01.mp3"},
                {"id": "b", "text": "दूसरा पाठ।", "output": "audio/part-01.mp3"}
            ])
            with self.assertRaisesRegex(finite.PlanError, "duplicate output path"):
                finite.execute(case_variant, self.cap(case_variant), submit=False)
            duplicate_id = self.manifest(root, [
                {"id": "same", "text": "हिंदी पाठ।", "output": "a.mp3"},
                {"id": "same", "text": "दूसरा पाठ।", "output": "b.mp3"}
            ])
            with self.assertRaises(finite.PlanError):
                finite.execute(duplicate_id, self.cap(duplicate_id), submit=False)

    def test_authored_mixed_script_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.manifest(Path(tmp), items=[
                {"id": "part-01", "text": 'उन्होंने कहा, "Hello, दुनिया!"', "output": "audio/part-01.mp3"}
            ])
            plan = finite.load_plan(manifest, self.cap(manifest))
            self.assertEqual(plan["items"][0]["text"], 'उन्होंने कहा, "Hello, दुनिया!"')

    def test_uncertain_state_is_never_reposted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest(root)
            plan = finite.load_plan(manifest, self.cap(manifest))
            approval = self.approval(manifest)
            evidence = validate_paid_approval(
                approval, episode="DEMO", lane=finite.APPROVAL_LANE, manifest_path=manifest,
                manifest_sha256=plan["manifestSha256"], finite_execution=finite.approval_scope(plan))
            plan.update(approvalAdmission=evidence)
            state_path = self.state_for(manifest)
            state = finite._initial_state(plan)
            state["items"]["part-01"]["status"] = "UNCERTAIN_NETWORK_ERROR"
            finite._atomic_json(state_path, state)
            with mock.patch.object(finite, "_post_once") as post:
                with self.assertRaises(finite.UncertainPost):
                    finite.execute(manifest, self.cap(manifest), submit=True,
                                   state_path=state_path, api_key="unused",
                                   approval_admission=approval)
            post.assert_not_called()

    def test_http_4xx_is_recorded_and_never_reposted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest(root)
            state_path = self.state_for(manifest)
            calls = []

            def reject(url, body, key):
                calls.append(url)
                raise urllib.error.HTTPError(
                    url, 400, "bad request", {}, io.BytesIO(b"invalid input"))

            approval = self.approval(manifest)
            with self.assertRaises(finite.PlanError):
                finite.execute(manifest, self.cap(manifest), submit=True,
                               state_path=state_path, post_fn=reject,
                               api_key="offline-placeholder", approval_admission=approval)
            self.assertEqual(len(calls), 1)
            state = json.loads(state_path.read_text())
            self.assertEqual(state["items"]["part-01"]["status"],
                             "HTTP_4XX_REFUSED")
            with self.assertRaises(finite.UncertainPost):
                finite.execute(manifest, self.cap(manifest), submit=True,
                               state_path=state_path, post_fn=reject,
                               api_key="offline-placeholder", approval_admission=approval)
            self.assertEqual(len(calls), 1)

    def test_mocked_submit_posts_once_validates_and_hash_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp3"
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                "sine=frequency=440:duration=1.5", "-c:a", "libmp3lame",
                str(source)
            ], check=True)
            audio = source.read_bytes()
            source.unlink()
            manifest = self.manifest(root)
            state_path = self.state_for(manifest)
            calls = []

            def fake_post(url, body, key):
                calls.append((url, body, key))
                return audio, {"status": 200, "requestId": "offline-test",
                               "contentType": "audio/mpeg"}

            approval = self.approval(manifest)
            first = finite.execute(manifest, self.cap(manifest), submit=True,
                                   state_path=state_path, post_fn=fake_post,
                                   api_key="offline-placeholder", approval_admission=approval)
            self.assertEqual(first["networkCalls"], 1)
            self.assertEqual(len(calls), 1)
            output = root / "audio/part-01.mp3"
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),
                             hashlib.sha256(audio).hexdigest())
            second = finite.execute(manifest, self.cap(manifest), submit=True,
                                    state_path=state_path, post_fn=fake_post,
                                    api_key="offline-placeholder", approval_admission=approval)
            self.assertEqual(second["status"], "ALREADY_COMPLETED")
            self.assertEqual(second["networkCalls"], 0)
            self.assertEqual(len(calls), 1)

    def test_declared_aggregate_requires_exact_cli_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest(root)
            doc = json.loads(manifest.read_text())
            total = sum(len(row["text"].strip()) for row in doc["items"])
            doc["aggregateCharacters"] = total
            manifest.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(finite.load_plan(manifest, total)["aggregateCharacters"], total)
            with self.assertRaisesRegex(finite.PlanError, "must equal frozen"):
                finite.load_plan(manifest, total + 1)

    def test_provenance_hash_and_episode_containment_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text("{}", encoding="utf-8")
            manifest = self.manifest(root)
            doc = json.loads(manifest.read_text())
            doc.update({
                "provenancePathBase": "manifest",
                "sourcePlan": "source.json",
                "sourcePlanSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "episodeRoot": ".",
            })
            manifest.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            finite.load_plan(manifest, self.cap(manifest))
            source.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(finite.PlanError, "changed after"):
                finite.load_plan(manifest, self.cap(manifest))

            source.write_text("{}", encoding="utf-8")
            doc["items"][0]["output"] = "../escape.mp3"
            manifest.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(finite.PlanError, "escapes episodeRoot"):
                finite.load_plan(manifest, self.cap(manifest))

    def test_custom_state_path_is_refused_before_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest(root)
            with self.assertRaisesRegex(finite.PlanError, "custom state paths are forbidden"):
                finite.execute(manifest, self.cap(manifest), submit=True, state_path=root / "alternate.json",
                               api_key="offline-placeholder")


if __name__ == "__main__":
    unittest.main()
