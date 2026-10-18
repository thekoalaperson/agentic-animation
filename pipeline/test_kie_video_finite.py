import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VIDEO = load(ROOT / "pipeline/kie_video.py", "finite_kie_video_test_target")
DIALOGUE = load(ROOT / "pipeline/kie_dialogue_finite.py", "finite_dialogue_test_target")


class KieVideoFiniteSafetyTest(unittest.TestCase):
    def test_public_media_download_never_sends_provider_authorization(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'public-result'
        with mock.patch.object(VIDEO.urllib.request, 'urlopen', return_value=response) as opened, \
             mock.patch.object(VIDEO, 'env_key') as credential:
            self.assertEqual(VIDEO.download_public_media('https://cdn.example/result.mp4'), b'public-result')
        credential.assert_not_called()
        request = opened.call_args.args[0]
        self.assertIsNone(request.get_header('Authorization'))
        self.assertEqual(request.full_url, 'https://cdn.example/result.mp4')

    def test_public_media_download_rejects_local_or_credential_urls(self):
        for url in ('file:///private/data', 'http://cdn.example/result.mp4', 'https://user:password@cdn.example/result.mp4'):
            with self.subTest(url=url), mock.patch.object(VIDEO.urllib.request, 'urlopen') as opened:
                with self.assertRaises(ValueError):
                    VIDEO.download_public_media(url)
                opened.assert_not_called()

    def fixture(self, root: Path):
        prompt = root / "prompt.txt"
        frame = root / "frame.png"
        stem = root / "motion" / "demo-s1001"
        prompt.write_text("Speak this exact Hindi sentence once: नमस्ते।", encoding="utf-8")
        frame.write_bytes(b"offline-source-frame")
        source_sha = hashlib.sha256(frame.read_bytes()).hexdigest()
        manifest_sha = "a" * 64
        execution_sha = "b" * 64
        argv = [
            "grok15", str(stem), str(prompt), str(frame),
            "--res", "720p", "--dur", "6", "--ar", "16:9",
            "--job-id", "demo-s1001",
            "--manifest-sha256", manifest_sha,
            "--source-sha256", source_sha,
            "--execution-request-sha256", execution_sha,
        ]
        return prompt, frame, stem, argv, source_sha, manifest_sha, execution_sha

    # Pinned from the hand-written parser this argparse front end replaced:
    # recorded pending records keep resuming only if this digest is unchanged.
    RECORDED_LOCAL_REQUEST_SHA256 = "f0fd525b786d01e6d12d027281d8f15ed14964a32b5441c5e1f554c482ce04b0"

    def test_print_request_is_offline_and_keeps_the_recorded_local_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, stem, argv, *_ = self.fixture(Path(tmp))
            printed = io.StringIO()
            with mock.patch.object(VIDEO, "upload_ref") as upload, \
                 mock.patch.object(VIDEO, "kie") as kie, \
                 mock.patch.object(VIDEO, "env_key") as credential, \
                 contextlib.redirect_stdout(printed):
                VIDEO.main(argv + ["--print-request"])
            upload.assert_not_called()
            kie.assert_not_called()
            credential.assert_not_called()
            self.assertFalse(stem.with_suffix(".pending.json").exists())
            shown = json.loads(printed.getvalue())
            self.assertEqual(shown["localRequestSha256"], self.RECORDED_LOCAL_REQUEST_SHA256)
            self.assertEqual(shown["endpoint"], "jobs/createTask")
            body = shown["body"]["input"]
            self.assertEqual(body["image_urls"], [VIDEO.PLACEHOLDER_URL + "frame.png"])
            self.assertEqual((body["duration"], body["resolution"], body["aspect_ratio"]),
                             (6, "720p", "16:9"))

    def test_resume_only_never_creates_a_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, stem, argv, *_ = self.fixture(Path(tmp))
            with mock.patch.object(VIDEO, "upload_ref") as upload, \
                 mock.patch.object(VIDEO, "kie") as kie:
                with self.assertRaisesRegex(SystemExit, "never creates"):
                    VIDEO.main(argv + ["--resume-only"])
            upload.assert_not_called()
            kie.assert_not_called()
            self.assertFalse(stem.with_suffix(".pending.json").exists())

    def test_uncertain_post_is_durable_and_never_auto_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, stem, argv, *_ = self.fixture(Path(tmp))
            creates = []

            def kie(route, payload=None):
                if route == "chat/credit":
                    return 1000
                if route == "jobs/createTask":
                    creates.append(payload)
                    raise RuntimeError("connection lost after request write")
                self.fail(f"unexpected route: {route}")

            with mock.patch.object(VIDEO, "upload_ref", return_value="https://offline/frame.png"), \
                 mock.patch.object(VIDEO, "kie", side_effect=kie):
                with self.assertRaises(RuntimeError):
                    VIDEO.main(argv)

            pending = stem.with_suffix(".pending.json")
            record = json.loads(pending.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "POSTING_UNCERTAIN")
            self.assertIsNone(record["taskId"])
            self.assertEqual(len(creates), 1)

            with mock.patch.object(VIDEO, "upload_ref") as upload, \
                 mock.patch.object(VIDEO, "kie", side_effect=kie):
                with self.assertRaisesRegex(SystemExit, "manual reconciliation"):
                    VIDEO.main(argv)
            upload.assert_not_called()
            self.assertEqual(len(creates), 1)

    def test_direct_upload_is_logged_beside_the_output_without_configuration(self):
        common = sys.modules[VIDEO.upload_ref.__module__]
        with tempfile.TemporaryDirectory() as tmp:
            _, frame, stem, argv, source_sha, *_ = self.fixture(Path(tmp))

            def kie(route, payload=None):
                if route == "chat/credit":
                    return 1000
                raise RuntimeError("offline: stop before any generation request")

            reply = {"data": {"downloadUrl": "https://files.invalid/frame.png"}}
            with mock.patch.dict(VIDEO.os.environ), \
                 mock.patch.object(common, "http_json", return_value=reply), \
                 mock.patch.object(common, "env_key", return_value="offline"), \
                 mock.patch.object(VIDEO, "kie", side_effect=kie):
                VIDEO.os.environ.pop("AGENTIC_UPLOAD_LOG", None)
                with self.assertRaises(RuntimeError):
                    VIDEO.main(argv)
            lines = (stem.parent / "upload-log.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["sha256"] for line in lines], [source_sha])

    def test_incomplete_landed_pair_refuses_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, stem, argv, *_ = self.fixture(Path(tmp))
            stem.parent.mkdir(parents=True)
            stem.with_suffix(".mp4").write_bytes(b"stale")
            with mock.patch.object(VIDEO, "upload_ref") as upload, \
                 mock.patch.object(VIDEO, "kie") as kie:
                with self.assertRaisesRegex(SystemExit, "incomplete landed pair"):
                    VIDEO.main(argv)
            upload.assert_not_called()
            kie.assert_not_called()

    def test_recorded_pending_task_is_polled_without_a_second_generation_post(self):
        for extra, receipt in (([], 27), (["--resume-only"], 27), ([], None),
                               (["--resume-only"], None)):
            with self.subTest(extra=extra, receipt=receipt):
                self.recover_recorded_pending_task(extra, receipt)

    def recover_recorded_pending_task(self, extra, receipt):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt, frame, stem, argv, source_sha, manifest_sha, execution_sha = self.fixture(root)
            model = "grok-imagine-video-1-5-preview"
            prompt_text = prompt.read_text(encoding="utf-8").strip()
            prompt_sha = hashlib.sha256(prompt.read_bytes()).hexdigest()
            local = VIDEO.local_request_contract(
                model=model, prompt_sha256=prompt_sha, source_sha256=source_sha,
                manifest_sha256=manifest_sha,
                execution_request_sha256=execution_sha, job_id="demo-s1001",
                res="720p", dur="6", ar="16:9", frame=str(frame), last=None,
                refs=[], videos=[], ref_names=[], reference_only=False,
                generate_audio=True, multi_clip=False)
            local_sha = VIDEO.canonical_sha256(local)
            provider_input = {
                "prompt": prompt_text, "image_urls": ["https://offline/frame.png"],
                "resolution": "720p", "aspect_ratio": "16:9", "duration": 6,
            }
            request = {
                "version": 1, "model": model, "input": provider_input,
                "sourceSha256": source_sha, "promptSha256": prompt_sha,
                "promptTextSha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                "executionManifestSha256": manifest_sha,
                "executionRequestSha256": execution_sha,
                "localRequestSha256": local_sha,
            }
            fingerprint = VIDEO.canonical_sha256(request)
            pending = stem.with_suffix(".pending.json")
            VIDEO.atomic_json(pending, {
                "provenanceVersion": 2, "status": "PENDING",
                "taskId": "task-recovered-1", "jobId": "demo-s1001",
                "model": model, "request": request,
                "requestFingerprint": fingerprint, "requestSha256": fingerprint,
                "localRequest": local, "localRequestSha256": local_sha,
                "executionManifestSha256": manifest_sha,
                "executionRequestSha256": execution_sha,
                "sourceSha256": source_sha, "promptSha256": prompt_sha,
                "walletBefore": 1000,
            }, replace=False)
            calls = []

            def kie(route, payload=None):
                calls.append(route)
                if route == "chat/credit":
                    return 980
                if route.startswith("jobs/recordInfo"):
                    record = {
                        "state": "success",
                        "resultJson": json.dumps({"resultUrls": ["https://offline/result.mp4"]}),
                    }
                    if receipt is not None:
                        record["creditsConsumed"] = receipt
                    return record
                if route == "jobs/createTask":
                    self.fail("recovery attempted a second generation POST")
                self.fail(f"unexpected route: {route}")

            with mock.patch.object(VIDEO, "upload_ref", return_value="https://offline/frame.png") as upload, \
                 mock.patch.object(VIDEO, "kie", side_effect=kie), \
                 mock.patch.object(VIDEO, "env_key", return_value="offline"), \
                 mock.patch.object(VIDEO, "download_public_media", return_value=b"recovered-media"), \
                 mock.patch.object(VIDEO.time, "sleep"):
                VIDEO.main(argv + extra)
            self.assertFalse(any(route == "jobs/createTask" for route in calls))
            self.assertTrue(any(route.startswith("jobs/recordInfo") for route in calls))
            self.assertFalse(pending.exists())
            if extra:
                upload.assert_not_called()
                self.assertNotIn("chat/credit", calls)
            meta = json.loads(stem.with_suffix(".vid.json").read_text(encoding="utf-8"))
            if receipt is None and extra:
                # Resume-only reads no wallet: without a receipt the charge is unknown.
                self.assertEqual((meta["credits"], meta["creditsSource"]), (None, None))
            elif receipt is None:
                # Wallet deltas can overlap concurrent jobs; they are labelled.
                self.assertEqual((meta["credits"], meta["creditsSource"]), (20.0, "wallet-delta"))
            else:
                self.assertEqual((meta["credits"], meta["creditsSource"]), (27, "receipt"))

    def test_success_writes_bound_immutable_sidecar_and_validated_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt, _, stem, argv, source_sha, manifest_sha, execution_sha = self.fixture(root)
            media_path = root / "provider.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=24:d=6",
                "-f", "lavfi", "-i", "sine=frequency=330:duration=6",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(media_path),
            ], check=True)
            raw = media_path.read_bytes()
            creates = []

            def kie(route, payload=None):
                if route == "chat/credit":
                    return 1000
                if route == "jobs/createTask":
                    creates.append(payload)
                    return {"taskId": "task-offline-1"}
                if route.startswith("jobs/recordInfo"):
                    return {
                        "state": "success", "creditsConsumed": 27,
                        "resultJson": json.dumps({"resultUrls": ["https://offline/result.mp4"]}),
                    }
                self.fail(f"unexpected route: {route}")

            with mock.patch.object(VIDEO, "upload_ref", return_value="https://offline/frame.png"), \
                 mock.patch.object(VIDEO, "kie", side_effect=kie), \
                 mock.patch.object(VIDEO, "env_key", return_value="offline"), \
                 mock.patch.object(VIDEO, "download_public_media", return_value=raw), \
                 mock.patch.object(VIDEO, "usd_per_credit", return_value=0.01), \
                 mock.patch.object(VIDEO.time, "sleep"):
                VIDEO.main(argv)

            video = stem.with_suffix(".mp4")
            sidecar = stem.with_suffix(".vid.json")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            output_sha = hashlib.sha256(raw).hexdigest()
            expected_fields = {
                "executionManifestSha256": manifest_sha,
                "executionRequestSha256": execution_sha,
                "sourceSha256": source_sha,
                "promptSha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
                "outputSha256": output_sha,
                "sha256": output_sha,
                "taskId": "task-offline-1",
                "jobId": "demo-s1001",
                "model": "grok-imagine-video-1-5-preview",
                "durationSeconds": 6,
                "resolution": "720p",
                "aspectRatio": "16:9",
                "audio": True,
            }
            for key, value in expected_fields.items():
                self.assertEqual(meta[key], value, key)
            self.assertEqual(meta["requestFingerprint"], meta["requestSha256"])
            self.assertEqual(VIDEO.canonical_sha256(meta["request"]), meta["requestFingerprint"])
            self.assertEqual((meta["creditsSource"], meta["cost_usd"]), ("receipt", 0.27))
            self.assertFalse(stem.with_suffix(".pending.json").exists())

            plan = {"manifestSha256": manifest_sha, "model": "grok-imagine-video-1-5-preview",
                    "resolution": "720p", "aspectRatio": "16:9"}
            row = {
                "id": "demo-s1001", "sourceSha256": source_sha,
                "promptSha256": expected_fields["promptSha256"],
                "durationSeconds": 6, "promptPath": prompt, "outputPath": stem,
                "executionRequestSha256": execution_sha, "sourcePath": root / "frame.png",
                "referencePaths": [], "referenceSha256s": [],
            }
            verified = DIALOGUE.validate_completed(plan, row)
            self.assertEqual(verified["sha256"], output_sha)

            with mock.patch.object(VIDEO, "upload_ref") as upload, \
                 mock.patch.object(VIDEO, "kie") as kie:
                VIDEO.main(argv)
            upload.assert_not_called()
            kie.assert_not_called()
            self.assertEqual(len(creates), 1)

    def test_references_land_as_version_three_and_bind_their_hashes(self):
        from dialogue_provenance import validate_dialogue_request
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt, frame, stem, argv, source_sha, manifest_sha, execution_sha = self.fixture(root)
            refs = []
            for name in ("demo-ref-a.png", "demo-ref-b.png"):
                (root / name).write_bytes(name.encode())
                refs.append(root / name)
            argv = argv + [argument for ref in refs for argument in ("--ref", str(ref))]
            media_path = root / "provider.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=24:d=6",
                "-f", "lavfi", "-i", "sine=frequency=330:duration=6",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(media_path),
            ], check=True)
            creates = []

            def kie(route, payload=None):
                if route == "chat/credit":
                    return 1000
                if route == "jobs/createTask":
                    creates.append(payload)
                    return {"taskId": "task-offline-refs"}
                return {"state": "success", "creditsConsumed": 30,
                        "resultJson": json.dumps({"resultUrls": ["https://offline/result.mp4"]})}

            with mock.patch.object(VIDEO, "upload_ref", side_effect=lambda path, log: f"https://offline/{Path(path).name}"), \
                 mock.patch.object(VIDEO, "kie", side_effect=kie), \
                 mock.patch.object(VIDEO, "download_public_media", return_value=media_path.read_bytes()), \
                 mock.patch.object(VIDEO, "usd_per_credit", return_value=None), \
                 mock.patch.object(VIDEO.time, "sleep"):
                VIDEO.main(argv)
            [create] = creates
            self.assertEqual(create["input"]["image_urls"], [
                "https://offline/frame.png", "https://offline/demo-ref-a.png",
                "https://offline/demo-ref-b.png"])
            meta = json.loads(stem.with_suffix(".vid.json").read_text(encoding="utf-8"))
            ref_hashes = [hashlib.sha256(ref.read_bytes()).hexdigest() for ref in refs]
            self.assertEqual(meta["provenanceVersion"], 3)
            self.assertEqual(meta["localRequest"]["inputHashes"]["references"], ref_hashes)

            plan = {"manifestSha256": manifest_sha, "model": "grok-imagine-video-1-5-preview",
                    "resolution": "720p", "aspectRatio": "16:9"}
            row = {"id": "demo-s1001", "sourceSha256": source_sha,
                   "promptSha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
                   "durationSeconds": 6, "promptPath": prompt, "outputPath": stem,
                   "executionRequestSha256": execution_sha, "sourcePath": frame,
                   "referencePaths": refs, "referenceSha256s": ref_hashes}
            self.assertEqual(DIALOGUE.validate_completed(plan, row)["taskId"], "task-offline-refs")
            expected = {**{key: meta[key] for key in (
                "jobId", "model", "sourceSha256", "promptSha256", "executionManifestSha256",
                "executionRequestSha256", "resolution", "aspectRatio", "durationSeconds", "audio")},
                "referenceSha256s": ref_hashes}
            validate_dialogue_request(meta, expected, prompt.read_text(encoding="utf-8"))
            for wrong in (ref_hashes[::-1], []):
                with self.subTest(references=wrong):
                    with self.assertRaises(DIALOGUE.PlanError):
                        DIALOGUE.validate_completed(plan, {**row, "referenceSha256s": wrong})
                    with self.assertRaisesRegex(ValueError, "frozen still and references"):
                        validate_dialogue_request(meta, {**expected, "referenceSha256s": wrong},
                                                  prompt.read_text(encoding="utf-8"))
            # The same request resumes without a second POST.
            with mock.patch.object(VIDEO, "upload_ref") as upload, mock.patch.object(VIDEO, "kie") as call:
                VIDEO.main(argv)
            upload.assert_not_called()
            call.assert_not_called()

    def test_first_frame_difference_is_measured_against_the_source_still(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, black, red = root / "clip.mp4", root / "black.png", root / "red.png"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=1280x720:r=24:d=1", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", str(video)], check=True)
            for path, colour in ((black, "black"), (red, "red")):
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                                f"color=c={colour}:s=1280x720", "-frames:v", "1", str(path)],
                               check=True)
            same = DIALOGUE.first_frame_difference(video, black)
            other = DIALOGUE.first_frame_difference(video, red)
            self.assertLess(same["meanAbsoluteDifference"], 2)
            self.assertGreater(other["meanAbsoluteDifference"], 50)
            self.assertEqual(DIALOGUE.first_frame_difference(video, root / "missing.png")["method"],
                             "unmeasured")


if __name__ == "__main__":
    unittest.main()
