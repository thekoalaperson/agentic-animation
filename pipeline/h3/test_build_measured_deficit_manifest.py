import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("build_measured_deficit_manifest.py")
SPEC = importlib.util.spec_from_file_location("measured_deficit", MODULE_PATH)
assert SPEC and SPEC.loader
measured_deficit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = measured_deficit
SPEC.loader.exec_module(measured_deficit)
CLIENT_PATH = Path(__file__).with_name("box") / "finite_client.py"
CLIENT_SPEC = importlib.util.spec_from_file_location("finite_h3_client", CLIENT_PATH)
assert CLIENT_SPEC and CLIENT_SPEC.loader
finite_client = importlib.util.module_from_spec(CLIENT_SPEC)
sys.modules[CLIENT_SPEC.name] = finite_client
CLIENT_SPEC.loader.exec_module(finite_client)


class MeasuredDeficitManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.ep = self.repo / "private-project"
        self.work = self.ep / "work"
        self.assembly = self.work / "assembly-v1"
        self.continuations = self.work / "h3-continuations-v1"
        self.assembly.mkdir(parents=True)
        self.continuations.mkdir(parents=True)
        self._make_fixture()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def media(self, relative, payload):
        path = self.ep / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def _make_fixture(self):
        storyboard = self.ep / "story/storyboard_plan.json"
        narration_manifest = self.work / "narration-hi-v1/manifest.json"
        dialogue_manifest = self.work / "grok-dialogue-v1/execution-manifest.json"
        production_overrides = self.ep / "story/production-overrides-v1.json"
        narration_state = self.work / "narration-hi-v1/manifest.eleven-state.json"
        for path, value in (
            (storyboard, {"episode": "DEMO", "scenes": [1, 2, 3]}),
            (narration_manifest, {"items": [{"id": "demo-s001-hi"}, {"id": "demo-s002-hi"}]}),
            (dialogue_manifest, {"jobs": [{"id": "demo-s003"}]}),
            (production_overrides, {"overrides": []}),
            (narration_state, {"items": {}}),
        ):
            self.write_json(path, value)

        self.h3a = self.media("work/media/s001.mp4", b"silent-h3-a")
        self.h3b = self.media("work/media/s002.mp4", b"silent-h3-b")
        self.narr1 = self.media("audio/narration-v1/s001.mp3", b"narration-one")
        self.narr2 = self.media("audio/narration-v1/s002.mp3", b"narration-two")
        self.dialogue = self.media("video/dialogue-v1/s003.mp4", b"grok-dialogue")
        self.sidecar = self.ep / "video/dialogue-v1/s003.source.json"
        self.review = self.ep / "work/review/admitted/dialogue.json"
        self.write_json(self.sidecar, {"source": "current"})
        self.write_json(self.review, {"decision": "approved"})

        ledger_path = self.work / "h3-review-ledger-v1.json"
        ledger = {
            "version": 1,
            "episode": "DEMO",
            "entries": {
                "demo-s001": {
                    "pictureRoute": "h3",
                    "segments": [{
                        "source": "work/media/s001.mp4", "sourceSha256": self.sha(self.h3a),
                        "in": 0, "out": 2, "speed": 1, "role": "walk-a", "verdict": "accepted",
                    }],
                },
                "demo-s002": {
                    "pictureRoute": "h3",
                    "segments": [{
                        "source": "work/media/s002.mp4", "sourceSha256": self.sha(self.h3b),
                        "in": 0, "out": 2, "speed": 1, "role": "walk-b", "verdict": "accepted",
                    }],
                },
                "demo-s003": {"pictureRoute": "grok-dialogue", "segments": []},
            },
        }
        self.write_json(ledger_path, ledger)
        self.timeline_inputs = self.assembly / "timeline-inputs.json"
        self.write_json(self.timeline_inputs, {"h3ReviewLedger": "work/h3-review-ledger-v1.json"})

        documents = {}
        for label, path in (
            ("storyboard", storyboard),
            ("narrationManifest", narration_manifest),
            ("dialogueManifest", dialogue_manifest),
            ("h3ReviewLedger", ledger_path),
            ("productionOverrides", production_overrides),
            ("narrationState", narration_state),
        ):
            documents[label] = {"path": str(path), "sha256": self.sha(path)}

        def row(path, **fields):
            return {"sha256": self.sha(path), **fields}

        media = {
            str(self.h3a): row(
                self.h3a, duration=2, videoDuration=2, audioDuration=None,
                videoFrames=48, videoFps=24, videoStreams=1, audioStreams=0,
            ),
            str(self.h3b): row(
                self.h3b, duration=2, videoDuration=2, audioDuration=None,
                videoFrames=48, videoFps=24, videoStreams=1, audioStreams=0,
            ),
            str(self.narr1): row(
                self.narr1, duration=3, videoDuration=None, audioDuration=3,
                videoFrames=None, videoFps=None, videoStreams=0, audioStreams=1,
            ),
            str(self.narr2): row(
                self.narr2, duration=2, videoDuration=None, audioDuration=2,
                videoFrames=None, videoFps=None, videoStreams=0, audioStreams=1,
            ),
            str(self.dialogue): row(
                self.dialogue, duration=2, videoDuration=2, audioDuration=2,
                videoFrames=48, videoFps=24, videoStreams=1, audioStreams=1,
            ),
        }
        admission = {
            "sidecar": "video/dialogue-v1/s003.source.json",
            "sidecarSha256": self.sha(self.sidecar),
            "reviewExport": "work/review/admitted/dialogue.json",
            "reviewExportSha256": self.sha(self.review),
            "reviewFingerprint": "a" * 64,
        }
        events = [
            {
                "id": "demo-s001", "kind": "narration", "audio": "audio/narration-v1/s001.mp3",
                "audioSha256": self.sha(self.narr1), "start": 0, "duration": 3, "end": 3,
            },
            {
                "id": "demo-s002", "kind": "narration", "audio": "audio/narration-v1/s002.mp3",
                "audioSha256": self.sha(self.narr2), "start": 3, "duration": 2, "end": 5,
            },
            {
                "id": "demo-s003", "kind": "dialogue", "audio": "video/dialogue-v1/s003.mp4",
                "picture": "video/dialogue-v1/s003.mp4", "audioSha256": self.sha(self.dialogue),
                "start": 5, "duration": 2, "end": 7, "admission": admission,
            },
        ]
        self.clock = self.assembly / "measured-clock.json"
        clock = {
            "schemaVersion": 1,
            "kind": measured_deficit.MEASURED_CLOCK_KIND,
            "episode": "DEMO", "language": "hi", "deliveryFps": 24,
            "clockFingerprint": "b" * 64, "runtimeSeconds": 7, "eventCount": 3,
            "events": events,
            "inputEvidence": {
                "config": {"path": str(self.timeline_inputs), "sha256": self.sha(self.timeline_inputs)},
                "documents": documents,
                "media": media,
                "events": events,
            },
            "renderable": False,
        }
        self.write_json(self.clock, clock)

        candidates = [
            {
                "sceneId": "demo-s001", "path": "work/media/s001.mp4",
                "sourceSha256": self.sha(self.h3a), "sourceStart": 0, "sourceDuration": 2,
                "sourceOut": 2, "duration": 2, "frames": 48, "speed": 1,
                "role": "walk-a", "status": "approved", "verdict": "accepted",
            },
            {
                "sceneId": "demo-s002", "path": "work/media/s002.mp4",
                "sourceSha256": self.sha(self.h3b), "sourceStart": 0, "sourceDuration": 2,
                "sourceOut": 2, "duration": 2, "frames": 48, "speed": 1,
                "role": "walk-b", "status": "approved", "verdict": "accepted",
            },
        ]
        self.readiness = self.assembly / "timeline-readiness.json"
        readiness = {
            "schemaVersion": 1, "episode": "DEMO", "ready": False, "dryRun": False,
            "storyboardScenes": 3, "narrationScenes": 2, "dialogueScenes": 1,
            "measuredNarration": 2, "measuredDialogue": 1, "approvedH3Scenes": 2,
            "missing": [{
                "type": "h3-coverage-deficit", "block": "b001", "sceneId": "demo-s001",
                "frames": 24, "seconds": 1,
            }],
            "errors": [], "warnings": [], "documents": documents, "media": media,
            "measuredClock": {
                "path": str(self.clock), "sha256": self.sha(self.clock),
                "clockFingerprint": "b" * 64, "runtimeSeconds": 7, "renderable": False,
            },
            "pictureBlocks": [{
                "id": "b001", "kind": "narration-fluid-picture", "start": 0,
                "duration": 5, "events": ["demo-s001", "demo-s002"], "candidates": candidates,
                "availableSeconds": 4, "selectedSeconds": 4, "surplusSeconds": 0,
                "deficitSeconds": 1, "status": "blocked-coverage-deficit",
            }],
        }
        self.write_json(self.readiness, readiness)

        actions = {
            "demo-s001": {"id": "demo-s001", "action": "Motion: the lantern keeps swinging as the curtain sways."},
            "demo-s002": {"id": "demo-s002", "action": "Motion: the windmill continues turning above the field."},
        }
        self.actions = self.work / "action-lines.json"
        self.write_json(self.actions, actions)
        self.config = self.continuations / "config.json"
        self.write_json(
            self.config,
            {
                "schemaVersion": 1, "episode": "DEMO",
                "timelineInputs": "work/assembly-v1/timeline-inputs.json",
                "timelineReadiness": "work/assembly-v1/timeline-readiness.json",
                "measuredClock": "work/assembly-v1/measured-clock.json",
                "actionLines": "work/action-lines.json",
                "outputDirectory": "work/h3-continuations-v1/generated",
                "batchSize": 32, "seedBase": 437000, "deadlineAt": 4102444800,
                "qualifiedRequest": {
                    "seconds": 7, "expectedFrames": 175, "shortEdge": 768,
                    "aspect": "auto", "steps": 30,
                },
            },
        )

    @staticmethod
    def fake_extract(source, frame_index, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"PNG" + source.name.encode() + str(frame_index).encode())

    def test_builds_hash_bound_finite_manifest_from_exact_deficit(self):
        result = measured_deficit.build(
            self.ep, self.config, wave=1, extractor=self.fake_extract
        )
        self.assertTrue(result["readyForH100"])
        self.assertEqual(result["exactDeficitFrames"], 24)
        self.assertEqual(result["jobs"], 1)
        self.assertEqual(result["batches"], 1)
        manifest_path = self.ep / result["manifests"][0]["manifest"]
        manifest = json.loads(manifest_path.read_text())
        finite_client.validate_manifest(manifest_path, maximum=1)
        self.assertEqual(manifest["route"], "owned-h100-h3")
        self.assertIs(manifest["generationApiAllowed"], False)
        self.assertIs(manifest["productionAudioAllowed"], False)
        self.assertIs(manifest["implicitRetimeAllowed"], False)
        self.assertIs(manifest["freezeFrameAllowed"], False)
        self.assertIs(manifest["duplicateFootageAllowed"], False)
        self.assertEqual(len(manifest["items"]), 1)
        item = manifest["items"][0]
        self.assertEqual(item["exactMeasuredDeficitFrames"], 24)
        self.assertEqual(item["plannedCoverageFramesThisWave"], 24)
        self.assertEqual(item["residualFramesAfterThisWave"], 0)
        self.assertEqual(item["expectedFrames"], 175)
        self.assertEqual(item["usableExpectedFramesAfterAnchorDrop"], 174)
        self.assertEqual(item["anchorSourceFrame"], 47)
        self.assertIs(item["dropConditioningFrameAtAcceptedIngest"], True)
        self.assertEqual(item["coverageMode"], "exact-last-frame-continuation")
        self.assertEqual(item["editorialSpeed"], 1.0)
        self.assertIs(item["productionAudioAllowed"], False)
        self.assertTrue(item["prompt"].strip())
        staged = manifest_path.parent / "stills" / item["still"]
        self.assertEqual(self.sha(staged), item["sourceSha256"])
        handoff = json.loads(Path(result["lifecycleHandoff"]).read_text())
        self.assertEqual(handoff["executionState"], "not-started-by-builder")
        self.assertIn("Do not use raw H3 audio.", handoff["prohibitions"])

    def test_refuses_partial_or_dry_run_timeline(self):
        readiness = json.loads(self.readiness.read_text())
        readiness["dryRun"] = True
        self.write_json(self.readiness, readiness)
        with self.assertRaisesRegex(measured_deficit.DeficitPlanBlocked, "dry-run"):
            measured_deficit.build(self.ep, self.config, extractor=self.fake_extract)

    def test_refuses_changed_measured_media_hash(self):
        self.h3a.write_bytes(b"changed-after-measurement")
        with self.assertRaisesRegex(measured_deficit.DeficitPlanBlocked, "Measured media changed"):
            measured_deficit.build(self.ep, self.config, extractor=self.fake_extract)

    def test_refuses_generation_while_an_accepted_window_is_unused(self):
        readiness = json.loads(self.readiness.read_text())
        first = readiness["pictureBlocks"][0]["candidates"][0]
        first.update({"sourceDuration": 40 / 24, "sourceOut": 40 / 24, "duration": 40 / 24, "frames": 40})
        readiness["pictureBlocks"][0].update(
            {"selectedSeconds": 88 / 24, "deficitSeconds": 32 / 24}
        )
        readiness["missing"][0].update({"frames": 32, "seconds": 32 / 24})
        self.write_json(self.readiness, readiness)
        with self.assertRaisesRegex(measured_deficit.DeficitPlanBlocked, "accepted unused"):
            measured_deficit.build(self.ep, self.config, extractor=self.fake_extract)

    def test_refuses_h3_source_with_audio(self):
        readiness = json.loads(self.readiness.read_text())
        readiness["media"][str(self.h3a)]["audioStreams"] = 1
        clock = json.loads(self.clock.read_text())
        clock["inputEvidence"]["media"] = readiness["media"]
        self.write_json(self.clock, clock)
        readiness["measuredClock"]["sha256"] = self.sha(self.clock)
        self.write_json(self.readiness, readiness)
        with self.assertRaisesRegex(measured_deficit.DeficitPlanBlocked, "contains audio"):
            measured_deficit.build(self.ep, self.config, extractor=self.fake_extract)

    def add_picture_event(self, **fields):
        clock = json.loads(self.clock.read_text())
        event = {"id": "demo-s004", "kind": "picture", "speaker": None, "audio": None,
                 "picture": None, "start": 7, "duration": 1, "end": 8, **fields}
        clock["events"].append(event)
        clock["inputEvidence"]["events"] = clock["events"]
        clock.update(eventCount=4, runtimeSeconds=8)
        self.write_json(self.clock, clock)
        readiness = json.loads(self.readiness.read_text())
        readiness.update(storyboardScenes=4, pictureScenes=1)
        readiness["measuredClock"]["sha256"] = self.sha(self.clock)
        self.write_json(self.readiness, readiness)
        return readiness

    def test_wordless_picture_event_needs_no_voice_binding(self):
        self.add_picture_event()
        result = measured_deficit.build(self.ep, self.config, wave=1, extractor=self.fake_extract)
        self.assertTrue(result["readyForH100"])
        self.assertEqual(result["exactDeficitFrames"], 24)

    def test_picture_event_counts_and_silence_are_checked(self):
        readiness = self.add_picture_event(audio="audio/narration-v1/s001.mp3")
        with self.assertRaisesRegex(measured_deficit.DeficitPlanBlocked, "carries voice"):
            measured_deficit.build(self.ep, self.config, extractor=self.fake_extract)
        self._make_fixture()
        readiness = self.add_picture_event()
        readiness.update(pictureScenes=0, storyboardScenes=3)
        self.write_json(self.readiness, readiness)
        with self.assertRaisesRegex(measured_deficit.DeficitPlanBlocked, "complete storyboard"):
            measured_deficit.build(self.ep, self.config, extractor=self.fake_extract)


if __name__ == "__main__":
    unittest.main()
