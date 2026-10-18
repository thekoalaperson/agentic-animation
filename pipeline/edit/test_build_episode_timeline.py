from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


timeline = load_module("build_episode_timeline", "build_episode_timeline.py")
from test_dialogue_provenance import bind_fixture
assembly = load_module("assemble_longform_for_timeline_test", "assemble_longform.py")


class TimelineBuilderTest(unittest.TestCase):
    def test_keyed_frozen_production_overrides(self):
        row = {"narration_hi": "नई पंक्ति", "narration_en": "New line."}
        self.assertEqual(timeline._production_overrides({"items": {"s001": row}}), {"s001": row})
        for invalid in ({"items": {}, "overrides": []},
                        {"items": {"s001": {"id": "s002"}}},
                        {"overrides": [{"id": "s001"}, {"id": "s001"}]}):
            with self.assertRaises(ValueError):
                timeline._production_overrides(invalid)

    def test_hindi_dialogue_attribution_not_spoken(self):
        self.assertEqual(timeline._spoken_dialogue('माली ने कहा, “यह मेरा उत्तर है।”'), 'यह मेरा उत्तर है।')
        self.assertEqual(timeline._spoken_dialogue('उसने “धर्म” पूछा।'), 'उसने “धर्म” पूछा।')

    def test_spoken_only_captions_preserve_the_source_default(self):
        hi='माली ने कहा, “यह मेरा उत्तर है।”'
        en='GARDENER: “This is my answer.”'
        self.assertEqual(timeline._dialogue_subtitle_text(hi,en,'source'),dict(hi=hi,en=en))
        self.assertEqual(timeline._dialogue_subtitle_text(hi,en,'spoken-only'),
                         dict(hi='यह मेरा उत्तर है।',en='This is my answer.'))
        with self.assertRaises(ValueError):timeline._dialogue_subtitle_text(hi,en,'rewrite')

    def test_worker_state_relocation_requires_exact_relative_path(self):
        ep = self.ep
        source = ep / 'audio/narration-v1/s001.mp3'
        original = '/opt/old-production/film/' + ep.name
        recorded = original + '/audio/narration-v1/s001.mp3'
        self.assertFalse(timeline._state_path_matches(recorded, source, ep))
        self.assertTrue(timeline._state_path_matches(recorded, source, ep, original))
        self.assertFalse(timeline._state_path_matches(recorded, ep / 'work/s001.mp3', ep, original))
        self.assertFalse(timeline._state_path_matches(recorded, source, ep, '/opt'))
        self.assertFalse(timeline._state_path_matches(original + '/../s001.mp3', source, ep, original))

    def test_cover_art_is_not_a_second_motion_stream(self):
        moving = {'codec_type':'video','avg_frame_rate':'24/1','nb_frames':'241','duration':'10.041667','disposition':{'attached_pic':0}}
        cover = {'codec_type':'video','disposition':{'attached_pic':1}}
        sound = {'codec_type':'audio','duration':'10.0'}
        payload = {'streams':[moving,sound,cover],'format':{'duration':'10.041667'}}
        with mock.patch.object(timeline.subprocess,'check_output',return_value=json.dumps(payload)):
            info = timeline.probe_media(Path('native-grok.mp4'))
        self.assertEqual((info.video_streams,info.audio_streams,info.video_frames),(1,1,241))
        payload['streams'].append(dict(moving))
        with mock.patch.object(timeline.subprocess,'check_output',return_value=json.dumps(payload)):
            self.assertEqual(timeline.probe_media(Path('ambiguous.mp4')).video_streams,2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.ep = self.repo / "private-project"
        for directory in (
            self.ep / "story",
            self.ep / "audio/narration-v1",
            self.ep / "work/narration-hi-v1",
            self.ep / "work/grok-dialogue-v1",
            self.ep / "work/grok-dialogue-v1/prompts",
            self.ep / "work/grok",
            self.ep / "work/review",
            self.ep / "work/h3",
            self.ep / "work/assembly-v1",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.rows = [
            self.row("demo-s001", 1),
            self.row("demo-s002", 1),
            self.row("demo-s003", 1, speaker="Character One"),
            self.row("demo-s004", 2),
        ]
        self.write_json(
            self.ep / "story/storyboard_plan.json",
            {"episode": "DEMO", "stills": self.rows},
        )
        self.write_json(
            self.ep / "work/narration-hi-v1/manifest.json",
            {
                "items": [
                    {
                        "id": f"{scene}-hi",
                        "shotId": scene,
                        "output": f"../../audio/narration-v1/{scene}-hi.mp3",
                    }
                    for scene in ("demo-s001", "demo-s002", "demo-s004")
                ]
            },
        )
        self.dialogue_source = self.ep / "work/grok-dialogue-v1/source.png"
        self.dialogue_prompt = self.ep / "work/grok-dialogue-v1/prompts/demo-s003.txt"
        self.dialogue_source.write_bytes(b"source-still")
        self.dialogue_prompt.write_text("offline prompt", encoding="utf-8")
        self.dialogue_manifest = self.ep / "work/grok-dialogue-v1/execution-manifest.json"
        self.write_json(self.dialogue_manifest, {
            "model": "grok-imagine-video-1-5-preview",
            "resolution": "720p", "aspectRatio": "16:9",
            "jobs": [{
                "id": "demo-s003", "speaker": "Character One", "hindi": "मैं बगीचे में हूँ।",
                "durationSeconds": 6,
                "source": str(self.dialogue_source.relative_to(self.ep)),
                "sourceSha256": self.sha(self.dialogue_source),
                "prompt": str(self.dialogue_prompt.relative_to(self.ep)),
                "promptSha256": self.sha(self.dialogue_prompt),
                "outputStem": "work/grok/demo-s003",
            }],
        })
        ledger_entries = {}
        for scene in ("demo-s001", "demo-s002", "demo-s004"):
            ledger_entries[scene] = {
                "pictureRoute": "h3",
                "segments": [
                    {
                        "source": f"work/h3/{scene}.mp4",
                        "in": 1 if scene == "demo-s004" else 0,
                        "out": 10 if scene == "demo-s001" else 3,
                        "role": "accepted-base",
                        "verdict": "approved",
                        **({"speed": 3.5} if scene == "demo-s001" else {}),
                    }
                ],
            }
        ledger_entries["demo-s003"] = {
            "pictureRoute": "grok-dialogue",
            "segments": [],
        }
        self.write_json(
            self.ep / "work/h3-review-ledger-v1.json", {"entries": ledger_entries}
        )
        self.config = self.ep / "work/assembly-v1/timeline-inputs.json"
        self.write_json(
            self.config,
            {
                "version": 1,
                "storyboard": "story/storyboard_plan.json",
                "narrationManifest": "work/narration-hi-v1/manifest.json",
                "dialogueManifest": "work/grok-dialogue-v1/execution-manifest.json",
                "h3ReviewLedger": "work/h3-review-ledger-v1.json",
                "outputDirectory": "work/assembly-v1",
                "narrationBreathing": {
                    "sameBeatSeconds": 0.40,
                    "beatEndSeconds": 0.50,
                    "beforeDialogueSeconds": 0.35,
                    "minimumSeconds": 0.35,
                    "maximumSeconds": 0.55,
                },
                "dialogueHandles": {"preSeconds": 0.70, "postSeconds": 0.90},
                "dialogueOutputs": {
                    "demo-s003": {
                        "path": "work/grok/demo-s003.mp4",
                        "verdict": "approved",
                        "speechWindow": {"start": 1.0, "end": 3.5},
                    }
                },
            },
        )
        for scene in ("demo-s001", "demo-s002", "demo-s004"):
            (self.ep / f"work/h3/{scene}.mp4").write_bytes(scene.encode())
            (self.ep / f"audio/narration-v1/{scene}-hi.mp3").write_bytes(scene.encode())
        (self.ep / "work/grok/demo-s003.mp4").write_bytes(b"dialogue")
        self.refresh_provenance()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def row(scene: str, beat: int, speaker: str | None = None):
        return {
            "still_id": scene,
            "beat": {"number": beat, "title": f"Beat {beat}"},
            "speaker": speaker,
            "narration_hi": f"हिन्दी {scene}",
            "narration_en": f"English {scene}",
        }

    @staticmethod
    def write_json(path: Path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def canonical_sha(value) -> str:
        return hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    def refresh_provenance(self):
        narration_manifest = self.ep / "work/narration-hi-v1/manifest.json"
        narration = json.loads(narration_manifest.read_text())
        narration_state = {
            "manifestSha256": self.sha(narration_manifest),
            "items": {},
        }
        for item in narration["items"]:
            output = (narration_manifest.parent / item["output"]).resolve()
            digest = self.sha(output)
            narration_state["items"][item["id"]] = {
                "status": "COMPLETED", "output": str(output),
                "sha256": digest, "outputSha256": digest,
            }
        self.write_json(narration_manifest.with_suffix(".eleven-state.json"), narration_state)

        config = json.loads(self.config.read_text())
        canonical_dialogue_manifest = self.ep / config["dialogueManifest"]
        dialogue_manifest = self.ep / config["dialogueOutputs"]["demo-s003"].get(
            "executionManifest", config["dialogueManifest"]
        )
        dialogue = json.loads(dialogue_manifest.read_text())
        job = dialogue["jobs"][0]
        output = self.ep / config["dialogueOutputs"]["demo-s003"]["path"]
        output_sha = self.sha(output)
        request = {
            "model": dialogue["model"],
            "input": {"duration": job["durationSeconds"], "resolution": dialogue["resolution"]},
        }
        execution_request = {
            "version": 1, "jobId": job["id"],
            "executionManifestSha256": self.sha(dialogue_manifest),
            "model": dialogue["model"], "sourceSha256": job["sourceSha256"],
            "promptSha256": job["promptSha256"],
            "durationSeconds": job["durationSeconds"],
            "resolution": dialogue["resolution"], "aspectRatio": dialogue["aspectRatio"],
            "audio": True, "outputStem": job["outputStem"],
        }
        request_sha = self.canonical_sha(request)
        sidecar = output.with_suffix(".vid.json")
        self.write_json(sidecar, bind_fixture({
            "jobId": job["id"],
            "executionManifestSha256": self.sha(dialogue_manifest),
            "executionRequestSha256": self.canonical_sha(execution_request),
            "sourceSha256": job["sourceSha256"],
            "promptSha256": job["promptSha256"],
            "outputSha256": output_sha, "sha256": output_sha,
            "model": dialogue["model"], "durationSeconds": job["durationSeconds"],
            "resolution": dialogue["resolution"], "aspectRatio": dialogue["aspectRatio"],
            "audio": True, "taskId": "offline-task",
            "request": request, "requestSha256": request_sha,
            "requestFingerprint": request_sha,
        }, (self.ep / job['prompt']).read_text()))
        bindings = [{
            "path": str(canonical_dialogue_manifest.relative_to(self.ep)),
            "sha256": self.sha(canonical_dialogue_manifest),
        }]
        if dialogue_manifest != canonical_dialogue_manifest:
            bindings.append({"path": str(dialogue_manifest.relative_to(self.ep)),
                             "sha256": self.sha(dialogue_manifest)})
            config["dialogueOutputs"][job["id"]]["executionManifestSha256"] = self.sha(dialogue_manifest)
        if config.get("productionOverrides"):
            override = self.ep / config["productionOverrides"]
            bindings.append({
                "path": str(override.relative_to(self.ep)), "sha256": self.sha(override)
            })
        fingerprint = self.canonical_sha({"bindings": bindings, "media": output_sha})
        review = self.ep / "work/review/dialogue-review.json"
        self.write_json(review, {
            "schemaVersion": 2, "reviewType": "dialogue",
            "reviewFingerprint": fingerprint, "sourceBindings": bindings,
            "reviews": [{
                "itemId": job["id"], "decision": "approved", "reviewedAt": "offline",
                "source": {
                    "path": str(output.relative_to(self.ep)), "sha256": output_sha,
                    "sidecarPath": str(sidecar.relative_to(self.ep)),
                    "sidecarSha256": self.sha(sidecar),
                },
            }],
        })
        config["dialogueOutputs"][job["id"]].update({
            "sha256": output_sha,
            "sidecar": str(sidecar.relative_to(self.ep)),
            "sidecarSha256": self.sha(sidecar),
            "reviewExport": str(review.relative_to(self.ep)),
            "reviewExportSha256": self.sha(review),
            "reviewFingerprint": fingerprint,
        })
        self.write_json(self.config, config)

    def fake_probe(self, path: Path):
        name = path.name
        if name.endswith("-hi.mp3"):
            duration = {
                "demo-s001-hi.mp3": 2.0,
                "demo-s002-hi.mp3": 1.0,
                "demo-s004-hi.mp3": 1.2,
            }[name]
            return timeline.MediaProbe(duration, None, duration, None, None, 0, 1)
        if name.startswith("m1-") and name.endswith(".mp3"):
            return timeline.MediaProbe(10.0, None, 10.0, None, None, 0, 1)
        if path.parent.name == "h3":
            if name == "demo-s001.mp4":
                return timeline.MediaProbe(10.0, 10.0, None, 240, 24.0, 1, 0)
            return timeline.MediaProbe(3.0, 3.0, None, 72, 24.0, 1, 0)
        if name == "demo-s003.mp4":
            return timeline.MediaProbe(5.0, 5.0, 5.0, 120, 24.0, 1, 1)
        raise AssertionError(path)

    def _add_continuation_fixture(self):
        ledger_path=self.ep/'work/h3-review-ledger-v1.json'
        ledger=json.loads(ledger_path.read_text())
        parent=ledger['entries']['demo-s001']['segments'][0]
        parent.update(out=4.0,speed=1.0)
        parent_path=self.ep/parent['source']
        continuation=self.ep/'work/h3/continued.mp4'
        continuation.write_bytes(b'continued-silent-picture')
        link=dict(source=parent['source'],sourceSha256=self.sha(parent_path),endFrameExclusive=96)
        segment=dict(source=str(continuation.relative_to(self.ep)),sourceSha256=self.sha(continuation),
            **{'in':1/24,'out':3.0},role='true-continuation',verdict='accepted',continuationOf=link)
        ledger['entries']['demo-s001']['segments'].append(segment)
        self.write_json(ledger_path,ledger)
        return ledger_path,ledger,segment

    def _reuse_dialogue_in_another_scene(self):
        self.rows[2].update(still_id="demo-s005", narration_hi="मैं बगीचे में हूँ।")
        self.write_json(self.ep / "story/storyboard_plan.json", {"episode": "DEMO", "stills": self.rows})
        path = self.ep / "work/h3-review-ledger-v1.json"
        ledger = json.loads(path.read_text())
        ledger["entries"]["demo-s005"] = ledger["entries"].pop("demo-s003")
        self.write_json(path, ledger)
        config = json.loads(self.config.read_text())
        choice = config["dialogueOutputs"].pop("demo-s003")
        choice.update(sourceSceneId="demo-s003", reuseReason="Same event quoted in the opening; complete native performance.")
        config["dialogueOutputs"]["demo-s005"] = choice
        self.write_json(self.config, config)

    def test_reused_dialogue_keeps_original_job_provenance(self):
        self._reuse_dialogue_in_another_scene()
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"])
        event = next(e for e in plan["events"] if e["id"] == "demo-s005")
        self.assertEqual(event["sourceSceneId"], "demo-s003")
        self.assertEqual(event["audio"], "work/grok/demo-s003.mp4")
        self.assertEqual(event["picture"], event["audio"])
        self.assertEqual(event["subtitleTextHi"], "मैं बगीचे में हूँ।")

    def test_reused_dialogue_cannot_change_words_or_speaker(self):
        self._reuse_dialogue_in_another_scene()
        for field, value in [("narration_hi", "यह दूसरी बात है।"), ("speaker", "Another person")]:
            with self.subTest(field=field):
                changed = [dict(r) for r in self.rows]
                changed[2][field] = value
                self.write_json(self.ep / "story/storyboard_plan.json", {"episode": "DEMO", "stills": changed})
                with self.assertRaises(timeline.TimelineIncomplete):
                    timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
                ready = json.loads((self.ep / "work/assembly-v1/timeline-readiness.json").read_text())
                self.assertTrue(any(e["type"] == "dialogue-reuse-contract" for e in ready["errors"]))

    def test_authored_dialogue_crop_preserves_audio_and_clock(self):
        from dataclasses import replace
        _, before = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        config = json.loads(self.config.read_text())
        config["dialogueOutputs"]["demo-s003"].update(
            pictureCrop={"x": 440, "y": 0, "width": 824, "height": 464},
            pictureCropReason="Keep the complete speaking performance; exclude an unrelated listener gesture.")
        self.write_json(self.config, config)
        def probe(path):
            return replace(self.fake_probe(path), video_width=1264, video_height=720)
        _, after = timeline.build_timeline(self.ep, self.config, probe=probe)
        a = next(e for e in before["events"] if e["kind"] == "dialogue")
        b = next(e for e in after["events"] if e["kind"] == "dialogue")
        for field in ("audio", "audioSha256", "sourceStart", "audioSourceDuration", "start", "end"):
            self.assertEqual(a[field], b[field])
        candidate = next(b for b in after["pictureBlocks"] if b["kind"] == "dialogue-fixed-sync")["candidates"][0]
        self.assertEqual(candidate["crop"], config["dialogueOutputs"]["demo-s003"]["pictureCrop"])
        config["dialogueOutputs"]["demo-s003"]["pictureCrop"]["x"] = 1200
        self.write_json(self.config, config)
        with self.assertRaises(timeline.TimelineIncomplete):
            timeline.build_timeline(self.ep, self.config, probe=probe)

    def test_timed_dialogue_crop_changes_only_tail_picture(self):
        from dataclasses import replace
        _, before = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        config = json.loads(self.config.read_text())
        crop = {"x": 440, "y": 0, "width": 824, "height": 464}
        config["dialogueOutputs"]["demo-s003"].update(
            pictureCrop=crop, pictureCropFrom=1.5,
            pictureCropReason="Preserve the establishing view and punch in only for the late listener gesture.")
        self.write_json(self.config, config)
        def probe(path):
            return replace(self.fake_probe(path), video_width=1264, video_height=720)
        _, after = timeline.build_timeline(self.ep, self.config, probe=probe)
        a = next(e for e in before["events"] if e["kind"] == "dialogue")
        b = next(e for e in after["events"] if e["kind"] == "dialogue")
        for field in ("audio", "audioSha256", "sourceStart", "audioSourceDuration", "start", "end"):
            self.assertEqual(a[field], b[field])
        blocks = [b for b in after["pictureBlocks"] if b["kind"] == "dialogue-fixed-sync"]
        self.assertEqual(len(blocks), 2)
        first, tail = [block["candidates"][0] for block in blocks]
        self.assertNotIn("crop", first)
        self.assertEqual(first["frames"], 36 - round(b["sourceStart"] * 24))
        self.assertEqual(tail["crop"], crop)
        self.assertEqual(tail["sourceStart"], 1.5)
        self.assertEqual(first["frames"] + tail["frames"], round(b["duration"] * 24))
        self.assertAlmostEqual(blocks[0]["start"] + blocks[0]["duration"], blocks[1]["start"], places=5)
        for invalid in (True, -1, 0, 1.51, 100, float("nan")):
            with self.subTest(cropFrom=invalid):
                config["dialogueOutputs"]["demo-s003"]["pictureCropFrom"] = invalid
                self.write_json(self.config, config)
                with self.assertRaises(timeline.TimelineIncomplete):
                    timeline.build_timeline(self.ep, self.config, probe=probe)

    def test_empty_picture_has_measured_clock_but_cannot_render(self):
        path = self.ep / "work/h3-review-ledger-v1.json"
        ledger = json.loads(path.read_text())
        ledger["entries"]["demo-s001"]["segments"] = []
        self.write_json(path, ledger)
        config = json.loads(self.config.read_text())
        config["allowCausalBorrowing"] = False
        self.write_json(self.config, config)
        with self.assertRaises(timeline.TimelineIncomplete):
            timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        ready = json.loads((self.ep / "work/assembly-v1/timeline-readiness.json").read_text())
        self.assertEqual(ready["measuredNarration"], 3)
        self.assertTrue(any(x["sceneId"] == "demo-s001" for x in ready["exactCoverageDeficits"]))
        self.assertFalse((self.ep / "work/assembly-v1/edit-plan.json").exists())
        self.assertFalse(ready["measuredClock"]["renderable"])

    def test_continuation_keeps_full_parent_and_short_continuous_tail(self):
        self._add_continuation_fixture()
        def probe(path):
            if path.name=='demo-s001-hi.mp3':
                return timeline.MediaProbe(4,None,4,None,None,0,1)
            return self.fake_probe(path)
        ready,plan=timeline.build_timeline(self.ep,self.config,probe=probe)
        self.assertTrue(ready['ready'])
        selected=plan['pictureBlocks'][0]['candidates'][:2]
        self.assertEqual([c['frames'] for c in selected],[96,10])
        self.assertEqual(selected[0]['sourceOut'],4.0)
        self.assertEqual(selected[1]['continuationOf']['endFrameExclusive'],96)
        self.assertAlmostEqual(selected[1]['sourceStart'],1/24,places=6)

    def test_true_continuation_can_cross_adjacent_narration_events(self):
        path, ledger, child = self._add_continuation_fixture()
        ledger['entries']['demo-s001']['segments'].pop()
        ledger['entries']['demo-s002']['segments'] = [child]
        self.write_json(path, ledger)
        def probe(path):
            if path.name == 'demo-s001-hi.mp3':
                return timeline.MediaProbe(3.6,None,3.6,None,None,0,1)
            return self.fake_probe(path)
        ready,plan=timeline.build_timeline(self.ep,self.config,probe=probe)
        self.assertTrue(ready['ready'])
        selected=plan['pictureBlocks'][0]['candidates'][:2]
        self.assertEqual(selected[0]['frames'],96)
        self.assertEqual(selected[1]['continuationOf']['endFrameExclusive'],96)
        self.assertEqual(selected[1]['sceneId'],'demo-s002')
        # A dialogue between the two sources is not a continuous predecessor.
        ledger['entries']['demo-s004']['segments']=[child]
        self.write_json(path,ledger)
        ready,_=timeline.build_timeline(self.ep,self.config,dry_run=True,probe=probe)
        self.assertTrue(any(e['type']=='h3-continuation-contract' for e in ready['errors']))

    def test_subtitle_manuscript_changes_only_subtitle_text_not_media_clock(self):
        from subtitle_manuscript import text_digest
        _,before=timeline.build_timeline(self.ep,self.config,probe=self.fake_probe)
        source=self.ep/'story/storyboard_plan.json'
        manuscript=self.ep/'work/review/subtitle-manuscript.json'
        items=[]
        for row in self.rows:
            hi,en=row['narration_hi'],row['narration_en']
            items.append({'id':row['still_id'],'sourceTextSha256':text_digest(hi,en),
                          'hi':hi,'en':en})
        items[0].update(en='Revised English.',editReason='Subtitle-only editorial test.',
                       cuePairs=[{'hi':items[0]['hi'],'en':'Revised English.'}],cueWeights=[1])
        self.write_json(manuscript,{'schemaVersion':1,'kind':'subtitle-text-manuscript',
            'sourceBindings':[{'path':'story/storyboard_plan.json','sha256':self.sha(source)}],
            'items':items})
        config=json.loads(self.config.read_text())
        config.update(subtitleManuscript=str(manuscript.relative_to(self.ep)),
                      subtitleManuscriptSha256=self.sha(manuscript))
        self.write_json(self.config,config)
        ready,after=timeline.build_timeline(self.ep,self.config,probe=self.fake_probe)
        self.assertTrue(ready['ready'])
        self.assertEqual(after['events'][0]['subtitleTextEn'],'Revised English.')
        fields=('id','text','audio','audioSha256','start','duration','end','audioSourceStart','audioSourceDuration')
        self.assertEqual([[e.get(k) for k in fields] for e in before['events']],
                         [[e.get(k) for k in fields] for e in after['events']])
        timing=timeline._subtitle_doc('en',after['events'],after['timelineFingerprint'])
        self.assertEqual(timing['shots'][0]['cueParts'],['Revised English.'])
        self.assertNotEqual(before['timelineFingerprint'],after['timelineFingerprint'])
        config['subtitleManuscriptSha256']='0'*64
        self.write_json(self.config,config)
        with self.assertRaises(timeline.TimelineIncomplete):
            timeline.build_timeline(self.ep,self.config,probe=self.fake_probe)

    def test_native_dialogue_endpoint_can_condition_the_following_h3(self):
        path=self.ep/'work/h3-review-ledger-v1.json'
        ledger=json.loads(path.read_text())
        child=ledger['entries']['demo-s004']['segments'][0]
        parent=self.ep/'work/grok/demo-s003.mp4'
        child.update({'in':1/24,'out':3.0,'sourceSha256':self.sha(self.ep/'work/h3/demo-s004.mp4'),
                      'continuationOf':dict(source=str(parent.relative_to(self.ep)),
                          sourceSha256=self.sha(parent),endFrameExclusive=120,parentKind='dialogue')})
        self.write_json(path,ledger)
        config=json.loads(self.config.read_text())
        config['dialogueOutputs']['demo-s003'].pop('speechWindow')
        self.write_json(self.config,config)
        ready,plan=timeline.build_timeline(self.ep,self.config,probe=self.fake_probe)
        self.assertTrue(ready['ready'])
        following=plan['pictureBlocks'][-1]['candidates'][0]
        self.assertEqual(following['continuationOf']['endFrameExclusive'],120)
        self.assertEqual(following['continuationOf']['parentKind'],'dialogue')
        config['dialogueOutputs']['demo-s003']['speechWindow']={'start':1.,'end':3.5}
        self.write_json(self.config,config)
        with self.assertRaises(timeline.TimelineIncomplete):
            timeline.build_timeline(self.ep,self.config,probe=self.fake_probe)
        ready=json.loads((self.ep/'work/assembly-v1/timeline-readiness.json').read_text())
        self.assertTrue(any(e['type']=='h3-continuation-contract' for e in ready['errors']))

    def test_episode_requiring_score_cannot_admit_an_empty_score(self):
        config=json.loads(self.config.read_text())
        config['requireApprovedScore']=True
        config['scorePlacements']=[]
        self.write_json(self.config,config)
        with self.assertRaises(timeline.TimelineIncomplete):
            timeline.build_timeline(self.ep,self.config,probe=self.fake_probe)
        ready=json.loads((self.ep/'work/assembly-v1/timeline-readiness.json').read_text())
        self.assertTrue(any(r['type']=='score-admission' for r in ready['missing']))

    def test_pending_score_measurement_never_creates_renderable_plan(self):
        config=json.loads(self.config.read_text())
        config.update(requireApprovedScore=True, scorePlacements=[])
        self.write_json(self.config,config)
        ready,plan=timeline.build_timeline(self.ep,self.config,probe=self.fake_probe,
                                          measure_pending_score=True)
        self.assertFalse(ready['ready'])
        self.assertIsNone(plan)
        self.assertTrue(any(r['type']=='score-admission' for r in ready['missing']))
        self.assertFalse((self.ep/'work/assembly-v1/edit-plan.json').exists())
        clock=json.loads((self.ep/'work/assembly-v1/measured-clock.json').read_text())
        self.assertFalse(clock['renderable'])
        self.assertEqual(len(clock['events']),4)
        self.assertTrue(json.loads(self.config.read_text())['requireApprovedScore'])

    def test_pending_score_measurement_does_not_waive_dialogue_errors(self):
        config=json.loads(self.config.read_text())
        config.update(requireApprovedScore=True, scorePlacements=[])
        config['dialogueOutputs']={}
        self.write_json(self.config,config)
        with self.assertRaises(timeline.TimelineIncomplete):
            timeline.build_timeline(self.ep,self.config,probe=self.fake_probe,
                                    measure_pending_score=True)

    def test_invalid_continuation_binding_or_retime_is_refused(self):
        path,ledger,segment=self._add_continuation_fixture()
        import copy
        variants=[]
        for field,value in [('source','work/h3/demo-s002.mp4'),('sourceSha256','0'*64),
                            ('endFrameExclusive',95),('endFrameExclusive',True)]:
            changed=copy.deepcopy(ledger)
            changed['entries']['demo-s001']['segments'][1]['continuationOf'][field]=value
            variants.append(changed)
        for field,value in [('in',0.0),('speed',0.9)]:
            changed=copy.deepcopy(ledger)
            changed['entries']['demo-s001']['segments'][1][field]=value
            variants.append(changed)
        changed=copy.deepcopy(ledger)
        changed['entries']['demo-s001']['segments'][0]['speed']=0.9
        variants.append(changed)
        changed=copy.deepcopy(ledger)
        changed['entries']['demo-s001']['segments'].pop(0)
        variants.append(changed)
        for changed in variants:
            with self.subTest(segments=changed['entries']['demo-s001']['segments']):
                self.write_json(path,changed)
                with self.assertRaises(timeline.TimelineIncomplete):
                    timeline.build_timeline(self.ep,self.config,probe=self.fake_probe)
                ready=json.loads((self.ep/'work/assembly-v1/timeline-readiness.json').read_text())
                self.assertTrue(any(e['type']=='h3-continuation-contract' for e in ready['errors']))

    def test_renderer_preserves_and_rechecks_continuation_endpoint(self):
        self._add_continuation_fixture()
        def probe(path):
            if path.name=='demo-s001-hi.mp3':
                return timeline.MediaProbe(4,None,4,None,None,0,1)
            return self.fake_probe(path)
        _,plan=timeline.build_timeline(self.ep,self.config,probe=probe)
        with mock.patch.object(assembly,'probe_duration',return_value=10.0):
            segments=assembly.build_picture_segments(self.ep,plan)
            self.assertEqual(segments[1]['continuationOf']['endFrameExclusive'],96)
            runs=assembly.coalesce_render_segments(segments)
            self.assertEqual(runs[1]['continuationOf']['endFrameExclusive'],96)
            plan['pictureBlocks'][0]['candidates'][0]['sourceDuration']=3.9
            with self.assertRaisesRegex(RuntimeError,'Continuation parent'):
                assembly.build_picture_segments(self.ep,plan)

    def test_shared_opening_chain_retains_both_complete_narration_events(self):
        path,ledger,_=self._add_continuation_fixture()
        ledger['entries']['demo-s002'].update(pictureRoute='omit',segments=[])
        ledger['pictureOwnershipContracts']=[dict(ownerSceneId='demo-s001',
            coveredSceneIds=['demo-s001','demo-s002'])]
        self.write_json(path,ledger)
        def probe(path):
            if path.name=='demo-s001-hi.mp3':
                return timeline.MediaProbe(4,None,4,None,None,0,1)
            return self.fake_probe(path)
        ready,plan=timeline.build_timeline(self.ep,self.config,probe=probe)
        self.assertTrue(ready['ready'])
        self.assertEqual([e['id'] for e in plan['events'][:2]],['demo-s001','demo-s002'])
        self.assertEqual([e['audioSourceDuration'] for e in plan['events'][:2]],[4,1])
        pictures=plan['pictureBlocks'][0]['candidates']
        self.assertEqual([p['sceneId'] for p in pictures],['demo-s001','demo-s001'])
        self.assertEqual(pictures[0]['frames'],96)
        self.assertEqual(sum(p['frames'] for p in pictures),sum(round(e['duration']*24) for e in plan['events'][:2]))
        subtitles=json.loads((self.ep/'work/assembly-v1/subtitle-hi-timing.json').read_text())
        self.assertEqual(len(subtitles['shots']),4)
        self.assertEqual(plan['pictureOwnershipContracts'],ledger['pictureOwnershipContracts'])
        with mock.patch.object(assembly,'probe_duration',return_value=10.0):
            segments=assembly.build_picture_segments(self.ep,plan)
        self.assertEqual(segments[0]['sceneId'],'demo-s001')
        for mutation in ('shorter','longer','early','wrong-owner','noncontiguous'):
            with self.subTest(mutation=mutation):
                changed=json.loads(json.dumps(segments))
                changed_plan=json.loads(json.dumps(plan))
                if mutation=='shorter': changed[1]['frames']-=1
                if mutation=='longer': changed[1]['frames']+=1
                if mutation=='early': changed[0]['timelineStart']-=1/24
                if mutation=='wrong-owner': changed[1]['sceneId']='demo-s002'
                if mutation=='noncontiguous':
                    changed_plan['pictureOwnershipContracts'][0]['coveredSceneIds']=['demo-s001','demo-s004']
                with self.assertRaisesRegex(RuntimeError,'Picture ownership'):
                    assembly.validate_picture_ownership(changed_plan,changed)

    def test_microcut_rebalance_does_not_shorten_a_linked_parent(self):
        candidates=[dict(availableFrames=100,role='base'),
                    dict(availableFrames=35,role='continuation',continuationOf={'source':'base'}),
                    dict(availableFrames=100,role='different-angle')]
        warnings=[]
        rows=timeline._allocate_windows(candidates,140,31,set(),warnings,'test')
        self.assertEqual([r['useFrames'] for r in rows],[100,35,5])
        self.assertEqual(len(warnings),1)

    def test_unlinked_microcut_rebalance_still_works(self):
        rows=timeline._allocate_windows([dict(availableFrames=100,role='base'),
            dict(availableFrames=100,role='independent')],110,31,set(),[],'test')
        self.assertEqual([r['useFrames'] for r in rows],[79,31])

    def test_measured_timeline_breaths_handles_and_bilingual_inputs(self):
        ready, plan = timeline.build_timeline(
            self.ep, self.config, probe=self.fake_probe
        )
        self.assertTrue(ready["ready"])
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan["events"]), 4)
        self.assertEqual(len(plan["pictureBlocks"]), 3)

        first, second, dialogue, last = plan["events"]
        self.assertAlmostEqual(first["duration"], 58 / 24, places=6)
        self.assertAlmostEqual(first["narrationBreath"]["actualSeconds"], 10 / 24, places=6)
        self.assertEqual(second["narrationBreath"]["reason"], "dialogue-entry")
        self.assertGreaterEqual(second["narrationBreath"]["actualSeconds"], 0.35)
        self.assertAlmostEqual(dialogue["sourceStart"], 7 / 24, places=6)
        self.assertAlmostEqual(dialogue["duration"], 99 / 24, places=6)
        self.assertAlmostEqual(dialogue["subtitleAudioAt"], 17 / 24, places=6)
        self.assertAlmostEqual(dialogue["subtitleAudioDuration"], 2.5, places=6)
        self.assertGreater(dialogue["dialogueHandles"]["preSeconds"], 0.69)
        self.assertGreater(dialogue["dialogueHandles"]["postSeconds"], 0.89)
        self.assertEqual(last["narrationBreath"]["reason"], "beat-boundary")
        first_picture = plan["pictureBlocks"][0]["candidates"][0]
        self.assertEqual(first_picture["speed"], 3.5)
        self.assertEqual(first_picture["frames"], 58)
        self.assertAlmostEqual(first_picture["sourceDuration"], 58 * 3.5 / 24, places=6)
        last_picture = plan["pictureBlocks"][-1]["candidates"][0]
        self.assertEqual(last_picture["sceneId"], "demo-s004")
        self.assertEqual(last_picture["sourceStart"], 1.0)

        hi = json.loads((self.ep / "work/assembly-v1/subtitle-hi-timing.json").read_text())
        en = json.loads((self.ep / "work/assembly-v1/subtitle-en-timing.json").read_text())
        self.assertEqual(len(hi["shots"]), 4)
        self.assertEqual(hi["duration"], en["duration"])
        self.assertNotEqual(hi["shots"][0]["text"], en["shots"][0]["text"])
        self.assertEqual(hi["timelineFingerprint"], plan["timelineFingerprint"])

        # A repeated run over identical bytes and decisions is byte-stable.
        original = (self.ep / "work/assembly-v1/edit-plan.json").read_bytes()
        ready_again, plan_again = timeline.build_timeline(
            self.ep, self.config, probe=self.fake_probe
        )
        self.assertTrue(ready_again["ready"])
        self.assertEqual(plan["timelineFingerprint"], plan_again["timelineFingerprint"])
        self.assertEqual(original, (self.ep / "work/assembly-v1/edit-plan.json").read_bytes())

    def test_dry_run_reports_unresolved_dialogue_without_inventing_plan(self):
        config = json.loads(self.config.read_text())
        config["dialogueOutputs"] = {}
        self.write_json(self.config, config)
        ready, plan = timeline.build_timeline(
            self.ep, self.config, dry_run=True, probe=self.fake_probe
        )
        self.assertFalse(ready["ready"])
        self.assertIsNone(plan)
        self.assertTrue(
            any(row["type"] == "approved-dialogue-output" for row in ready["missing"])
        )
        self.assertFalse((self.ep / "work/assembly-v1/edit-plan.json").exists())

    def test_authored_motion_breath_preserves_voice_and_requires_real_coverage(self):
        config = json.loads(self.config.read_text())
        config['narrationBreathOverrides'] = {'demo-s001': {
            'seconds': 1.25, 'reason': 'Visible final footstep before the next sentence.'}}
        self.write_json(self.config, config)
        ledger_path = self.ep / 'work/h3-review-ledger-v1.json'
        ledger = json.loads(ledger_path.read_text())
        ledger['entries']['demo-s001']['segments'][0]['speed'] = 1
        self.write_json(ledger_path, ledger)
        _, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        event = plan['events'][0]
        self.assertEqual(event['audioSourceDuration'], 2.0)
        self.assertEqual(event['duration'], 3.25)
        self.assertEqual(event['subtitleAudioDuration'], 2.0)
        self.assertEqual(event['narrationBreath']['actualSeconds'], 1.25)
        self.assertIn('Visible final footstep', event['narrationBreath']['reason'])
        config['narrationBreathOverrides']['demo-s001']['seconds'] = 100
        self.write_json(self.config, config)
        with self.assertRaises(timeline.TimelineIncomplete):
            timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)

    def test_authored_breath_rejects_unknown_dialogue_and_unreasoned_values(self):
        for scene, seconds, reason in [('demo-s003', 2, 'dialogue cannot be padded'),
                ('missing', 2, 'unknown'), ('demo-s001', float('nan'), 'invalid'),
                ('demo-s001', True, 'invalid'), ('demo-s001', -1, 'invalid'),
                ('demo-s001', 2, '   ')]:
            with self.subTest(scene=scene, seconds=seconds):
                config = json.loads(self.config.read_text())
                config['narrationBreathOverrides'] = {scene: {'seconds': seconds, 'reason': reason}}
                self.write_json(self.config, config)
                with self.assertRaises(timeline.TimelineIncomplete):
                    timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)

    def install_correction(self):
        correction_dir = self.ep / "work/grok-dialogue-corrections-v1"
        correction_dir.mkdir()
        correction_path = correction_dir / "execution-manifest.json"
        correction = json.loads(self.dialogue_manifest.read_text())
        job = correction["jobs"][0]
        prompt = correction_dir / "prompt.txt"
        prompt.write_text("Corrected action; same Hindi words.", encoding="utf-8")
        job.update(prompt=str(prompt.relative_to(self.ep)), promptSha256=self.sha(prompt),
                   outputStem=str((correction_dir / "demo-s003").relative_to(self.ep)))
        self.write_json(correction_path, correction)
        output = correction_dir / "demo-s003.mp4"
        output.write_bytes(b"corrected-dialogue")
        config = json.loads(self.config.read_text())
        config["dialogueOutputs"]["demo-s003"].update(
            path=str(output.relative_to(self.ep)),
            executionManifest=str(correction_path.relative_to(self.ep)),
        )
        self.write_json(self.config, config)
        self.refresh_provenance()
        return correction_path

    def test_correction_keeps_own_manifest_and_source_bound_approval(self):
        correction_path = self.install_correction()
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        self.assertIn("grok-dialogue-corrections-v1", plan["events"][2]["picture"])
        self.assertIn(self.sha(correction_path), json.dumps(plan["events"][2]))
        self.assertEqual((self.ep / "work/grok/demo-s003.mp4").read_bytes(), b"dialogue")

    def test_correction_requires_its_manifest_in_review_bindings(self):
        correction_path = self.install_correction()
        review_path = self.ep / "work/review/dialogue-review.json"
        review = json.loads(review_path.read_text())
        review["sourceBindings"] = [row for row in review["sourceBindings"]
                                    if row["sha256"] != self.sha(correction_path)]
        self.write_json(review_path, review)
        config = json.loads(self.config.read_text())
        config["dialogueOutputs"]["demo-s003"]["reviewExportSha256"] = self.sha(review_path)
        self.write_json(self.config, config)
        ready, plan = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertFalse(ready["ready"])
        self.assertIsNone(plan)
        self.assertIn("binding", json.dumps(ready).lower())

    def test_correction_manifest_tamper_is_rejected(self):
        correction_path = self.install_correction()
        correction_path.write_text(correction_path.read_text() + "\n")
        ready, plan = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertFalse(ready["ready"])
        self.assertIsNone(plan)
        self.assertIn("correction execution manifest changed", json.dumps(ready))

    def test_correction_does_not_inherit_original_delivery_window(self):
        self.install_correction()
        original = json.loads(self.dialogue_manifest.read_text())
        original['jobs'][0]['speechWindow'] = dict(start=0.5, end=1.0)
        self.write_json(self.dialogue_manifest, original)
        config = json.loads(self.config.read_text())
        config['dialogueOutputs']['demo-s003'].pop('speechWindow')
        self.write_json(self.config, config)
        self.refresh_provenance()
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready['ready'], ready)
        self.assertEqual(plan['events'][2]['duration'], 5.0)

    def test_correction_cannot_change_spoken_dialogue_even_with_fresh_hashes(self):
        correction_path = self.install_correction()
        correction = json.loads(correction_path.read_text())
        correction["jobs"][0]["hindi"] = "यह अलग बात है।"
        self.write_json(correction_path, correction)
        self.refresh_provenance()
        ready, plan = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertFalse(ready["ready"])
        self.assertIsNone(plan)
        self.assertIn("correction changes current spoken dialogue", json.dumps(ready))

    def test_picture_deficit_keeps_hash_bound_measured_speech_clock(self):
        ledger_path = self.ep / "work/h3-review-ledger-v1.json"
        ledger = json.loads(ledger_path.read_text())
        # Make both opening narration sources too short even after neighbor
        # borrowing. All speech remains valid and therefore measurable.
        ledger["entries"]["demo-s001"]["segments"][0].update(
            {"in": 0.0, "out": 0.5, "speed": 1.0}
        )
        ledger["entries"]["demo-s002"]["segments"][0].update(
            {"in": 0.0, "out": 0.5}
        )
        self.write_json(ledger_path, ledger)
        with self.assertRaises(timeline.TimelineIncomplete):
            timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        clock_path = self.ep / "work/assembly-v1/measured-clock.json"
        self.assertTrue(clock_path.is_file())
        clock = json.loads(clock_path.read_text())
        self.assertFalse(clock["renderable"])
        self.assertEqual(clock["kind"], "measured-speech-clock-not-renderable-edit-plan")
        self.assertEqual(len(clock["events"]), 4)
        self.assertEqual(clock["runtimeSeconds"], clock["events"][-1]["end"])
        readiness = json.loads(
            (self.ep / "work/assembly-v1/timeline-readiness.json").read_text()
        )
        self.assertEqual(readiness["measuredClock"]["sha256"], self.sha(clock_path))
        self.assertTrue(any(row["type"] == "h3-coverage-deficit"
                            for row in readiness["missing"]))
        self.assertEqual(readiness["pictureBlocks"][0]["status"], "blocked-coverage-deficit")
        self.assertEqual(
            readiness["exactCoverageDeficits"],
            [row for row in readiness["missing"] if row["type"] == "h3-coverage-deficit"],
        )
        self.assertFalse((self.ep / "work/assembly-v1/edit-plan.json").exists())

    def test_production_override_owns_submitted_hindi_and_english_translation(self):
        manifest_path = self.ep / "work/narration-hi-v1/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["items"][0]["text"] = "बदली हुई हिन्दी पंक्ति"
        self.write_json(manifest_path, manifest)
        override_path = self.ep / "story/production-overrides-v1.json"
        self.write_json(
            override_path,
            {
                "overrides": [
                    {
                        "id": "demo-s001",
                        "narration_hi": "बदली हुई हिन्दी पंक्ति",
                        "narration_en": "Authored replacement translation.",
                    }
                ]
            },
        )
        config = json.loads(self.config.read_text())
        config["productionOverrides"] = "story/production-overrides-v1.json"
        self.write_json(self.config, config)
        self.refresh_provenance()
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"])
        self.assertEqual(plan["events"][0]["text"], "बदली हुई हिन्दी पंक्ति")
        self.assertEqual(
            plan["events"][0]["subtitleTextEn"], "Authored replacement translation."
        )

    def test_execution_manifest_and_dialogue_override_own_spoken_and_subtitle_text(self):
        execution_path = self.ep / "work/grok-dialogue-v1/execution-manifest.json"
        execution = json.loads(execution_path.read_text())
        execution["jobs"][0]["hindi"] = "सीमा का सम्मान करो।"
        self.write_json(execution_path, execution)
        override_path = self.ep / "story/production-overrides-v1.json"
        self.write_json(
            override_path,
            {
                "overrides": [
                    {
                        "id": "demo-s003",
                        "narration_hi": "माली: “सीमा का सम्मान करो।”",
                        "narration_en": "Character One: “Respect the boundary.”",
                    }
                ]
            },
        )
        config = json.loads(self.config.read_text())
        config["dialogueManifest"] = "work/grok-dialogue-v1/execution-manifest.json"
        config["productionOverrides"] = "story/production-overrides-v1.json"
        self.write_json(self.config, config)
        self.refresh_provenance()

        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"])
        dialogue = plan["events"][2]
        self.assertEqual(dialogue["text"], "सीमा का सम्मान करो।")
        self.assertEqual(dialogue["subtitleTextHi"], "माली: “सीमा का सम्मान करो।”")
        self.assertEqual(dialogue["subtitleTextEn"], "Character One: “Respect the boundary.”")

    def test_dialogue_override_drift_is_rejected_before_timeline_write(self):
        execution_path = self.ep / "work/grok-dialogue-v1/execution-manifest.json"
        execution = json.loads(execution_path.read_text())
        execution["jobs"][0]["hindi"] = "पुरानी पंक्ति।"
        self.write_json(execution_path, execution)
        override_path = self.ep / "story/production-overrides-v1.json"
        self.write_json(
            override_path,
            {"overrides": [{"id": "demo-s003", "narration_hi": "माली: “नई पंक्ति।”", "narration_en": "New line."}]},
        )
        config = json.loads(self.config.read_text())
        config["dialogueManifest"] = "work/grok-dialogue-v1/execution-manifest.json"
        config["productionOverrides"] = "story/production-overrides-v1.json"
        self.write_json(self.config, config)
        self.refresh_provenance()

        ready, plan = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertFalse(ready["ready"])
        self.assertIsNone(plan)
        self.assertTrue(any(
            row.get("type") == "production-override-drift" and row.get("field") == "dialogue_hi"
            for row in ready["errors"]
        ))

    def test_renderer_honors_bounded_source_start_for_dialogue_dict(self):
        source = self.ep / "work/grok/demo-s003.mp4"
        plan = {
            "pictureBlocks": [
                {
                    "id": "b001",
                    "kind": "dialogue-fixed-sync",
                    "start": 0.0,
                    "duration": 2.0,
                    "candidates": [
                        {
                            "path": str(source), "sourceStart": 1.25, "duration": 2.0,
                            "sourceSha256": "a" * 64,
                        }
                    ],
                }
            ]
        }
        with mock.patch.object(assembly, "probe_duration", return_value=5.0), mock.patch.object(
            assembly, "sha", return_value="a" * 64
        ):
            segments = assembly.build_picture_segments(self.ep, plan)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["sourceStart"], 1.25)
        self.assertEqual(segments[0]["frames"], 48)

    def test_unsafe_explicit_h3_speed_is_refused(self):
        ledger_path = self.ep / "work/h3-review-ledger-v1.json"
        ledger = json.loads(ledger_path.read_text())
        ledger["entries"]["demo-s001"]["segments"][0]["speed"] = 4.1
        self.write_json(ledger_path, ledger)
        ready, plan = timeline.build_timeline(
            self.ep, self.config, dry_run=True, probe=self.fake_probe
        )
        self.assertFalse(ready["ready"])
        self.assertIsNone(plan)
        self.assertTrue(any(row["type"] == "h3-speed-contract" for row in ready["errors"]))

    def test_stale_narration_or_review_evidence_is_rejected(self):
        state_path = (self.ep / "work/narration-hi-v1/manifest.json").with_suffix(
            ".eleven-state.json"
        )
        state = json.loads(state_path.read_text())
        state["items"]["demo-s001-hi"]["sha256"] = "0" * 64
        self.write_json(state_path, state)
        ready, plan = timeline.build_timeline(
            self.ep, self.config, dry_run=True, probe=self.fake_probe
        )
        self.assertFalse(ready["ready"])
        self.assertIsNone(plan)
        self.assertTrue(any(row["type"] == "narration-admission" for row in ready["errors"]))

        self.refresh_provenance()
        review = self.ep / "work/review/dialogue-review.json"
        review.write_text(review.read_text() + " ", encoding="utf-8")
        ready, plan = timeline.build_timeline(
            self.ep, self.config, dry_run=True, probe=self.fake_probe
        )
        self.assertFalse(ready["ready"])
        self.assertIsNone(plan)
        self.assertTrue(any(row["type"] == "dialogue-admission" for row in ready["errors"]))

    def test_score_requires_finite_state_sidecar_hash_and_review(self):
        score_root = self.ep / "work/music-score-v1"
        sources = score_root / "sources"
        sources.mkdir(parents=True)
        draft = score_root / "creative-draft.json"
        self.write_json(draft, {"fixture": True})
        payload = {"duration": 10, "style": "offline"}
        manifest = score_root / "execution-manifest.json"
        self.write_json(manifest, {
            "sourceDraftSha256": self.sha(draft),
            "cues": [{"id": "m1-test", "payload": payload}],
        })
        track = sources / "m1-test-take-1.mp3"
        track.write_bytes(b"score-audio")
        sidecar = sources / "m1-test-take-1.source.json"
        track_sha = self.sha(track)
        self.write_json(sidecar, {
            "cueId": "m1-test", "candidateIndex": 1,
            "executionManifestSha256": self.sha(manifest),
            "sourceDraftSha256": self.sha(draft),
            "requestSha256": self.canonical_sha(payload), "sha256": track_sha,
        })
        sidecar_sha = self.sha(sidecar)
        state = manifest.with_suffix(".state.json")
        self.write_json(state, {
            "executionManifestSha256": self.sha(manifest),
            "items": {"m1-test": {
                "status": "DOWNLOADED",
                "candidates": [{
                    "index": 1, "status": "COMPLETE", "file": str(track),
                    "sha256": track_sha, "sidecar": str(sidecar),
                    "sidecarSha256": sidecar_sha,
                }],
            }},
        })
        bindings = [
            {"path": str(path.relative_to(self.ep)), "sha256": self.sha(path)}
            for path in (
                manifest,
                self.ep / "work/narration-hi-v1/manifest.json",
                self.dialogue_manifest,
            )
        ]
        fingerprint = self.canonical_sha({"score": track_sha, "bindings": bindings})
        review = self.ep / "work/review/score-review.json"
        self.write_json(review, {
            "schemaVersion": 2, "reviewType": "score",
            "reviewFingerprint": fingerprint, "sourceBindings": bindings,
            "reviews": [{
                "itemId": "m1-test", "decision": "approved",
                "source": {
                    "path": str(track.relative_to(self.ep)), "sha256": track_sha,
                    "sidecarPath": str(sidecar.relative_to(self.ep)),
                    "sidecarSha256": sidecar_sha,
                },
            }],
        })
        config = json.loads(self.config.read_text())
        config["scoreManifest"] = str(manifest.relative_to(self.ep))
        config["scorePlacements"] = [{
            "movement": "m1-test", "take": 1,
            "track": str(track.relative_to(self.ep)), "trackSha256": track_sha,
            "sidecar": str(sidecar.relative_to(self.ep)), "sidecarSha256": sidecar_sha,
            "reviewExport": str(review.relative_to(self.ep)),
            "reviewExportSha256": self.sha(review), "reviewFingerprint": fingerprint,
            "timelineStart": 0.0, "timelineDuration": 10.0,
        }]
        self.write_json(self.config, config)
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"])
        self.assertEqual(plan["scorePlacements"][0]["trackSha256"], track_sha)

        track.write_bytes(b"swapped-score")
        ready, plan = timeline.build_timeline(
            self.ep, self.config, dry_run=True, probe=self.fake_probe
        )
        self.assertFalse(ready["ready"])
        self.assertIsNone(plan)
        self.assertTrue(any(row["type"] == "score-admission" for row in ready["errors"]))

    def test_renderer_emits_explicit_setpts_speed_and_exact_frame_count(self):
        source = self.ep / "work/h3/demo-s001.mp4"
        item = {
            "id": "p001",
            "source": str(source),
            "sourceSha256": "abc",
            "sourceStart": 0.5,
            "sourceDuration": 7.0,
            "speed": 3.5,
            "frames": 48,
        }
        cache = self.ep / "work/render-test"
        cache.mkdir()
        commands = []

        def fake_run(command, **_kwargs):
            commands.append([str(value) for value in command])
            Path(command[-1]).write_bytes(b"rendered")
            return mock.Mock()

        with mock.patch.object(assembly, "run", side_effect=fake_run):
            output = assembly.render_segment(item, cache, 1280, 720)
        self.assertTrue(output.is_file())
        vf = commands[0][commands[0].index("-vf") + 1]
        self.assertIn("trim=duration=7.000000000", vf)
        self.assertIn("setpts=(PTS-STARTPTS)/3.500000000", vf)
        self.assertEqual(commands[0][commands[0].index("-frames:v") + 1], "48")

    def test_picture_renderer_coalesces_exact_adjacent_source_windows(self):
        shared = {
            "source": "/episode/work/head-flight.mp4",
            "sourceSha256": "a" * 64,
            "speed": 1.0,
        }
        segments = [
            {
                **shared,
                "id": "p112",
                "block": "b112",
                "role": "separation-to-head-flight",
                "sourceStart": 0.0,
                "sourceDuration": 5.0,
                "timelineStart": 10.0,
                "duration": 5.0,
                "frames": 120,
            },
            {
                **shared,
                "id": "p113",
                "block": "b113",
                "role": "head-flight-continuation",
                "sourceStart": 5.0,
                "sourceDuration": 1.75,
                "timelineStart": 15.0,
                "duration": 1.75,
                "frames": 42,
            },
        ]
        runs = assembly.coalesce_render_segments(segments)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["memberSegmentIds"], ["p112", "p113"])
        self.assertEqual(
            runs[0]["memberRoles"],
            ["separation-to-head-flight", "head-flight-continuation"],
        )
        self.assertEqual(runs[0]["frames"], 162)
        self.assertAlmostEqual(runs[0]["sourceDuration"], 6.75)

        broken = [dict(row) for row in segments]
        broken[1]["sourceStart"] = 5.04
        self.assertEqual(len(assembly.coalesce_render_segments(broken)), 2)

    def test_renderer_rechecks_dialogue_review_hash_before_decode(self):
        source = self.ep / "work/grok/demo-s003.mp4"
        sidecar = source.with_suffix(".vid.json")
        review = self.ep / "work/review/dialogue-review.json"
        plan = {
            "runtimeSeconds": 5.0,
            "events": [{
                "id": "demo-s003", "kind": "dialogue",
                "audio": str(source.relative_to(self.ep)),
                "audioSha256": self.sha(source), "start": 0.0, "duration": 5.0,
                "admission": {
                    "sidecar": str(sidecar.relative_to(self.ep)),
                    "sidecarSha256": self.sha(sidecar),
                    "reviewExport": str(review.relative_to(self.ep)),
                    "reviewExportSha256": "0" * 64,
                },
            }],
        }
        with mock.patch.object(assembly, "audio") as decode:
            with self.assertRaisesRegex(RuntimeError, "reviewExport hash changed"):
                assembly.build_voice(self.ep, self.ep / "work/assembly-v1", plan)
        decode.assert_not_called()

    def test_renderer_rechecks_score_hash_before_decode(self):
        track = self.ep / "work/score.mp3"
        sidecar = self.ep / "work/score.source.json"
        review = self.ep / "work/review/score-review.json"
        execution = self.ep / "work/score-execution.json"
        for path in (track, sidecar, review, execution):
            path.write_bytes(path.name.encode())
        plan = {
            "runtimeSeconds": 10.0,
            "scorePlacements": [{
                "movement": "m1", "take": 1,
                "track": str(track.relative_to(self.ep)), "trackSha256": "0" * 64,
                "sidecar": str(sidecar.relative_to(self.ep)),
                "sidecarSha256": self.sha(sidecar),
                "reviewExport": str(review.relative_to(self.ep)),
                "reviewExportSha256": self.sha(review),
                "executionManifest": str(execution.relative_to(self.ep)),
                "executionManifestSha256": self.sha(execution),
                "timelineStart": 0.0, "timelineDuration": 10.0,
            }],
        }
        with mock.patch.object(assembly, "audio") as decode:
            with self.assertRaisesRegex(RuntimeError, "track hash changed"):
                assembly.build_music(
                    self.ep, self.ep / "work/assembly-v1", plan, {"shots": []}
                )
        decode.assert_not_called()

    # Wordless picture, voice heard over picture, and the documents they need.

    def ledger(self):
        path = self.ep / "work/h3-review-ledger-v1.json"
        return path, json.loads(path.read_text())

    def write_rows(self, rows):
        self.write_json(self.ep / "story/storyboard_plan.json", {"episode": "DEMO", "stills": rows})

    def make_wordless(self, scene="demo-s002", **fields):
        rows = [dict(r) for r in self.rows]
        for row in rows:
            if row["still_id"] == scene:
                row.update(delivery="wordless", **fields)
                row.pop("narration_hi")
                row.pop("narration_en")
        self.write_rows(rows)
        path, ledger = self.ledger()
        for segment in ledger["entries"][scene]["segments"]:
            segment["sourceSha256"] = self.sha(self.ep / segment["source"])
        self.write_json(path, ledger)
        return rows

    def readiness(self):
        return json.loads((self.ep / "work/assembly-v1/timeline-readiness.json").read_text())

    def test_films_without_new_markers_gain_no_new_fields(self):
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertNotIn("pictureScenes", ready)
        audit = json.loads((self.ep / "work/assembly-v1/timeline-audit.json").read_text())
        self.assertNotIn("pictureCount", audit)
        narration = {"id", "shot", "kind", "speaker", "audio", "picture", "audioSha256", "start",
                     "duration", "end", "audioSourceStart", "audioSourceDuration", "subtitleAudioAt",
                     "subtitleAudioDuration", "narrationBreath", "text", "subtitleTextHi", "subtitleTextEn"}
        for event in plan["events"]:
            if event["kind"] == "narration":
                self.assertEqual(set(event), narration)
            else:
                self.assertEqual(event["kind"], "dialogue")
                self.assertNotIn("voice", event)
                self.assertNotIn("durationRule", event)
        for block in plan["pictureBlocks"]:
            for candidate in block["candidates"]:
                self.assertNotIn("origin", candidate)
        self.assertEqual(list(ready["documents"])[:4],
                         ["storyboard", "narrationManifest", "dialogueManifest", "h3ReviewLedger"])

    def test_wordless_row_is_clocked_by_its_accepted_take(self):
        self.make_wordless()
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        self.assertEqual((ready["narrationScenes"], ready["dialogueScenes"], ready["pictureScenes"]), (2, 1, 1))
        first, picture, dialogue, last = plan["events"]
        self.assertEqual(picture["kind"], "picture")
        self.assertIsNone(picture["audio"])
        self.assertEqual(picture["duration"], 3.0)
        self.assertEqual(picture["durationRule"]["rule"], "take")
        self.assertNotIn("narrationBreath", picture)
        self.assertNotIn("subtitleTextHi", picture)
        self.assertEqual(first["narrationBreath"]["reason"], "intra-beat")
        # Picture joins the neighbouring narration run as one fluid block.
        self.assertEqual(plan["pictureBlocks"][0]["events"], ["demo-s001", "demo-s002"])
        self.assertEqual([c["frames"] for c in plan["pictureBlocks"][0]["candidates"]], [58, 72])
        for language in ("hi", "en"):
            timing = json.loads((self.ep / f"work/assembly-v1/subtitle-{language}-timing.json").read_text())
            self.assertEqual([s["id"] for s in timing["shots"]], ["demo-s001", "demo-s003", "demo-s004"])
            self.assertEqual(timing["duration"], plan["runtimeSeconds"])
        audit = json.loads((self.ep / "work/assembly-v1/timeline-audit.json").read_text())
        self.assertEqual((audit["narrationCount"], audit["pictureCount"]), (2, 1))

    def test_wordless_picture_segment_must_declare_its_bytes(self):
        self.make_wordless()
        path, ledger = self.ledger()
        ledger["entries"]["demo-s002"]["segments"][0].pop("sourceSha256")
        self.write_json(path, ledger)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertTrue(any(e["type"] == "picture-segment-hash" for e in ready["errors"]))
        ledger["entries"]["demo-s002"]["segments"][0]["verdict"] = "rejected"
        self.write_json(path, ledger)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertTrue(any(m["type"] == "picture-take" for m in ready["missing"]))

    def test_authored_hold_trims_the_take_and_borrows_from_narration(self):
        self.make_wordless(holdSeconds=1.5)
        _, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        picture = plan["events"][1]
        self.assertEqual((picture["duration"], picture["durationRule"]["rule"]), (1.5, "authored"))
        self.assertEqual([c["frames"] for c in plan["pictureBlocks"][0]["candidates"]], [58, 36])
        # A hold longer than the take borrows from the narration neighbour.
        self.make_wordless(holdSeconds=3.25)
        _, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        audit = json.loads((self.ep / "work/assembly-v1/timeline-audit.json").read_text())
        self.assertEqual([(b["donorScene"], b["receiverScene"], b["frames"]) for b in audit["coverageBorrowing"]],
                         [("demo-s001", "demo-s002", 6)])
        self.assertEqual(sum(c["frames"] for c in plan["pictureBlocks"][0]["candidates"]), 58 + 78)
        # A hold no accepted picture can fill is a coverage deficit, as for speech.
        self.make_wordless(holdSeconds=10)
        with self.assertRaises(timeline.TimelineIncomplete):
            timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(any(m["type"] == "h3-coverage-deficit" and m["sceneId"] == "demo-s002"
                            for m in self.readiness()["missing"]))
        self.assertTrue((self.ep / "work/assembly-v1/measured-clock.json").is_file())

    def test_narration_borrows_from_a_held_picture_neighbour(self):
        self.make_wordless(holdFrames=24)
        path, ledger = self.ledger()
        ledger["entries"]["demo-s001"]["segments"][0].update(out=1.0, speed=1.0)
        self.write_json(path, ledger)
        _, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        audit = json.loads((self.ep / "work/assembly-v1/timeline-audit.json").read_text())
        self.assertEqual([(b["donorScene"], b["receiverScene"], b["frames"]) for b in audit["coverageBorrowing"]],
                         [("demo-s002", "demo-s001", 34)])
        self.assertEqual([(c["sceneId"], c["frames"]) for c in plan["pictureBlocks"][0]["candidates"]],
                         [("demo-s001", 24), ("demo-s002", 58)])

    def test_wordless_row_after_dialogue_forms_its_own_block(self):
        self.make_wordless("demo-s004")
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        self.assertEqual([b["kind"] for b in plan["pictureBlocks"]],
                         ["narration-fluid-picture", "dialogue-fixed-sync", "narration-fluid-picture"])
        self.assertEqual(plan["pictureBlocks"][-1]["events"], ["demo-s004"])
        self.assertEqual(plan["events"][-1]["duration"], 2.0)
        self.assertEqual(plan["events"][1]["narrationBreath"]["reason"], "dialogue-entry")

    def test_neighbour_may_own_a_held_picture_row(self):
        self.make_wordless(holdSeconds=1.0)
        path, ledger = self.ledger()
        ledger["entries"]["demo-s002"].update(pictureRoute="omit", segments=[])
        ledger["entries"]["demo-s001"]["segments"][0].update(speed=1.0)
        ledger["pictureOwnershipContracts"] = [dict(ownerSceneId="demo-s001",
                                                    coveredSceneIds=["demo-s001", "demo-s002"])]
        self.write_json(path, ledger)
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        with mock.patch.object(assembly, "probe_duration", return_value=10.0):
            segments = assembly.build_picture_segments(self.ep, plan)
        self.assertEqual([(s["sceneId"], s["frames"]) for s in segments[:1]], [("demo-s001", 58 + 24)])
        # Without a hold, a picture row whose neighbour owns it has no clock.
        self.make_wordless()
        path, ledger = self.ledger()
        ledger["entries"]["demo-s002"].update(pictureRoute="omit", segments=[])
        self.write_json(path, ledger)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertTrue(any(e["type"] == "picture-route" for e in ready["errors"]))

    def test_picture_route_name_is_an_alias_of_h3(self):
        path, ledger = self.ledger()
        ledger["entries"]["demo-s002"]["pictureRoute"] = "picture"
        self.write_json(path, ledger)
        _, renamed = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        ledger["entries"]["demo-s002"]["pictureRoute"] = "h3"
        self.write_json(path, ledger)
        _, original = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertEqual(renamed["pictureBlocks"], original["pictureBlocks"])

    def test_voice_over_is_narration_that_names_its_voice(self):
        rows = [dict(r) for r in self.rows]
        rows[1].update(delivery="unsynced", voice="Character Two")
        self.write_rows(rows)
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        event = plan["events"][1]
        self.assertEqual((event["kind"], event["speaker"], event["voice"]), ("narration", None, "Character Two"))
        self.assertIn("narrationBreath", event)
        timing = json.loads((self.ep / "work/assembly-v1/subtitle-en-timing.json").read_text())
        self.assertEqual(timing["shots"][1]["speaker"], "Character Two")
        self.assertFalse(timing["shots"][1]["dialogue"])
        # A storyboard speaker always means a lip-sync take; voice-over never hides behind it.
        rows[2].update(delivery="unsynced")
        self.write_rows(rows)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertTrue(any(e["type"] == "storyboard-delivery" and e["sceneId"] == "demo-s003"
                            and "clear the storyboard row's speaker" in e["error"]
                            for e in ready["errors"]), ready["errors"])

    def test_production_override_may_mark_delivery_but_not_contradict_it(self):
        override_path = self.ep / "work/script-v1/overrides.json"
        override_path.parent.mkdir(parents=True)
        self.write_json(override_path, {"items": {"demo-s002": {
            "speaker": None, "narration_hi": "", "narration_en": "", "delivery": "wordless", "reason": "picture"}}})
        config = json.loads(self.config.read_text())
        config["productionOverrides"] = str(override_path.relative_to(self.ep))
        self.write_json(self.config, config)
        self.refresh_provenance()
        path, ledger = self.ledger()
        ledger["entries"]["demo-s002"]["segments"][0]["sourceSha256"] = self.sha(self.ep / "work/h3/demo-s002.mp4")
        self.write_json(path, ledger)
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        self.assertEqual(plan["events"][1]["kind"], "picture")
        rows = [dict(r) for r in self.rows]
        rows[1]["delivery"] = "unsynced"
        self.write_rows(rows)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertTrue(any(e["type"] == "storyboard-delivery" for e in ready["errors"]))

    def test_invalid_delivery_and_duration_rules_name_the_row(self):
        for scene, fields in [
                ("demo-s002", dict(delivery="wordless", holdSeconds=0.01)),
                ("demo-s002", dict(delivery="wordless", durationRule="take", holdSeconds=1)),
                ("demo-s002", dict(delivery="wordless", holdSeconds=1, holdFrames=24)),
                ("demo-s002", dict(delivery="wordless", holdFrames=True)),
                ("demo-s002", dict(delivery="wordless", durationRule="speech")),
                ("demo-s002", dict(delivery="wordless", voice="Character Two")),
                ("demo-s002", dict(delivery="synced")),
                ("demo-s002", dict(delivery="whispered")),
                ("demo-s001", dict(holdSeconds=1.0)),
                ("demo-s003", dict(delivery="wordless"))]:
            with self.subTest(scene=scene, fields=fields):
                rows = [dict(r) for r in self.rows]
                next(r for r in rows if r["still_id"] == scene).update(fields)
                self.write_rows(rows)
                ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
                self.assertTrue(any(e["type"] == "storyboard-delivery" and e["sceneId"] == scene
                                    for e in ready["errors"]), ready["errors"])

    def test_breath_override_cannot_pad_a_wordless_row(self):
        self.make_wordless()
        config = json.loads(self.config.read_text())
        config["narrationBreathOverrides"] = {"demo-s002": {"seconds": 1.0, "reason": "hold"}}
        self.write_json(self.config, config)
        with self.assertRaisesRegex(timeline.TimelineIncomplete, "no narration scene"):
            timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)

    def test_lane_documents_are_required_only_when_rows_need_them(self):
        self.write_rows([self.rows[2]])
        self.refresh_provenance()
        audit_path = self.ep / "work/assembly-v1/timeline-audit.json"
        # Explicit keys keep both documents required, as every build did before.
        named, _ = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        named_audit = json.loads(audit_path.read_text())
        config = json.loads(self.config.read_text())
        config.pop("narrationManifest")
        config.pop("h3ReviewLedger")
        self.write_json(self.config, config)
        # An existing all-dialogue project relying on the default paths, with
        # the files present, still binds them: only the edited config differs.
        defaulted, _ = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        defaulted_audit = json.loads(audit_path.read_text())
        self.assertEqual(list(defaulted["documents"]),
                         ["storyboard", "narrationManifest", "dialogueManifest", "h3ReviewLedger",
                          "narrationState"])
        self.assertEqual(defaulted["documents"], named["documents"])
        evidence = named_audit["inputEvidence"]
        self.assertNotEqual(evidence["config"], defaulted_audit["inputEvidence"]["config"])
        evidence["config"] = defaulted_audit["inputEvidence"]["config"]
        self.assertEqual(evidence, defaulted_audit["inputEvidence"])
        self.assertEqual(timeline._canonical_sha(evidence), defaulted["timelineFingerprint"])
        for name in ("manifest.json", "manifest.eleven-state.json"):
            (self.ep / "work/narration-hi-v1" / name).unlink()
        (self.ep / "work/h3-review-ledger-v1.json").unlink()
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        self.assertEqual(set(ready["documents"]), {"storyboard", "dialogueManifest"})
        self.assertEqual([e["kind"] for e in plan["events"]], ["dialogue"])
        # Wordless picture needs the ledger but not the narration lane.
        self.tearDown()
        self.setUp()
        self.make_wordless()
        rows = json.loads((self.ep / "story/storyboard_plan.json").read_text())["stills"]
        self.write_rows(rows[1:3])
        config = json.loads(self.config.read_text())
        config.pop("narrationManifest")
        self.write_json(self.config, config)
        self.refresh_provenance()
        (self.ep / "work/narration-hi-v1/manifest.json").unlink()
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        self.assertNotIn("narrationManifest", ready["documents"])
        self.assertEqual([e["kind"] for e in plan["events"]], ["picture", "dialogue"])
        # A named narration state with no manifest to bind is a named error.
        config = json.loads(self.config.read_text())
        config["narrationState"] = "work/narration-hi-v1/manifest.eleven-state.json"
        self.write_json(self.config, config)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertTrue(any(e["type"] == "narration-provenance" for e in ready["errors"]), ready)
        config.pop("narrationState")
        self.write_json(self.config, config)
        # A document the config names explicitly stays required.
        config = json.loads(self.config.read_text())
        config["h3ReviewLedger"] = "work/missing-ledger.json"
        self.write_json(self.config, config)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertTrue(any(m.get("purpose") == "h3ReviewLedger" for m in ready["missing"]))

    def test_string_beats_are_compared_by_identity(self):
        rows = [dict(r) for r in self.rows]
        for row, beat in zip(rows, ("opening", "opening", "opening", "ending")):
            row["beat"] = beat
        self.write_rows(rows)
        _, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        reasons = [e["narrationBreath"]["reason"] for e in plan["events"] if e["kind"] == "narration"]
        self.assertEqual(reasons, ["intra-beat", "dialogue-entry", "beat-boundary"])
        self.assertEqual(timeline._beat_identity({"beat": {"number": 3}}), 3)
        self.assertIsNone(timeline._beat_identity({}))

    def test_source_unit_citations_change_no_event(self):
        _, plain = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        rows = [dict(r, source_units=[f"s001.e00{i}"]) for i, r in enumerate(self.rows, 1)]
        self.write_rows(rows)
        ready, cited = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        self.assertEqual(cited["events"], plain["events"])

    def test_passage_links_two_halves_of_one_speech(self):
        rows = [dict(r) for r in self.rows]
        rows[1]["passage"] = "demo-s001"
        self.write_rows(rows)
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        first, second = plan["events"][:2]
        self.assertNotIn("passage", first)
        self.assertEqual(second["passage"], "demo-s001")
        # Each half keeps its own words and its own measured clip.
        self.assertEqual((first["subtitleTextHi"], second["subtitleTextHi"]), ("हिन्दी demo-s001", "हिन्दी demo-s002"))
        self.assertNotEqual(first["audio"], second["audio"])
        for language in ("hi", "en"):
            timing = json.loads((self.ep / f"work/assembly-v1/subtitle-{language}-timing.json").read_text())
            self.assertEqual([s.get("passage") for s in timing["shots"]], [None, "demo-s001", None, None])
        audit = json.loads((self.ep / "work/assembly-v1/timeline-audit.json").read_text())
        # The audit counts utterances, so the two narration halves count once.
        self.assertEqual((audit["eventCount"], audit["narrationCount"], audit["dialogueCount"]), (4, 2, 1))

    def test_passage_must_continue_the_row_before_it_in_the_same_voice(self):
        cases = [
            (3, "demo-s001", "directly before"),
            (2, "demo-s002", "spoken the same way"),
            (0, "demo-s000", "directly before"),
        ]
        for index, link, message in cases:
            with self.subTest(index=index):
                rows = [dict(r) for r in self.rows]
                rows[index]["passage"] = link
                self.write_rows(rows)
                ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
                self.assertTrue(any(e["type"] == "storyboard-passage" and e["sceneId"] == rows[index]["still_id"]
                                    and message in e["error"] for e in ready["errors"]), ready["errors"])
        rows = [dict(r) for r in self.rows]
        rows[1].update(passage="demo-s001", delivery="unsynced", voice="Character Two")
        self.write_rows(rows)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertTrue(any(e["type"] == "storyboard-passage" and "same speaker" in e["error"]
                            for e in ready["errors"]), ready["errors"])

    def test_passage_halves_of_dialogue_share_a_speaker(self):
        rows = [{"still_id": "a", "speaker": "One"}, {"still_id": "b", "speaker": "One", "passage": "a"},
                {"still_id": "c", "speaker": "Two", "passage": "b"}]
        kinds = {"a": "dialogue", "b": "dialogue", "c": "dialogue"}
        errors = []
        links = timeline._passages(rows, {"b": {"passage": "a"}}, kinds, {}, errors)
        self.assertEqual(links, {"b": "a"})
        self.assertEqual([(e["sceneId"], "same speaker" in e["error"]) for e in errors], [("c", True)])
        errors = []
        timeline._passages(rows[:2], {"b": {"passage": "x"}}, kinds, {}, errors)
        self.assertIn("differs from production override", errors[0]["error"])

    def write_origin(self, name, record):
        path = self.ep / "work/h3" / name
        self.write_json(path, record)
        return {"record": str(path.relative_to(self.ep)), "recordSha256": self.sha(path)}

    def test_version_two_ledger_binds_every_segment_to_its_origin(self):
        path, ledger = self.ledger()
        ledger["schemaVersion"] = 2
        self.write_json(path, ledger)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertEqual(sum(e["type"] == "picture-segment-hash" for e in ready["errors"]), 3)
        sha = {scene: self.sha(self.ep / f"work/h3/{scene}.mp4") for scene in ("demo-s001", "demo-s002", "demo-s004")}
        origins = {
            "demo-s001": dict(kind="h3-ingest", **self.write_origin(
                "ingest-receipt.json", {"items": [{"id": "demo-s001", "silentSha256": sha["demo-s001"]}]})),
            "demo-s002": dict(kind="derivative", **self.write_origin(
                "demo-s002.derive.json", {"sourceSha256": "b" * 64, "outputSha256": sha["demo-s002"],
                                          "parameters": {"audio": "removed", "fps": 24}})),
            "demo-s004": dict(kind="kie-video", **self.write_origin(
                "demo-s004.vid.json", {"sha256": sha["demo-s004"], "outputSha256": sha["demo-s004"],
                                       "taskId": "offline-task"})),
        }
        for scene, origin in origins.items():
            segment = ledger["entries"][scene]["segments"][0]
            segment.update(sourceSha256=sha[scene], origin=origin)
        self.write_json(path, ledger)
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready)
        kinds = [c["origin"]["kind"] for b in plan["pictureBlocks"] if b["kind"] != "dialogue-fixed-sync"
                 for c in b["candidates"]]
        self.assertEqual(kinds, ["h3-ingest", "derivative", "kie-video"])
        supplied = self.write_origin("demo-s004.supplied.json", {"fileSha256": sha["demo-s004"]})
        ledger["entries"]["demo-s004"]["segments"][0]["origin"] = dict(kind="supplied", **supplied)
        self.write_json(path, ledger)
        self.assertTrue(timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)[0]["ready"])
        import copy
        bad = []
        changed = copy.deepcopy(ledger)
        changed["entries"]["demo-s002"]["segments"][0]["origin"]["recordSha256"] = "0" * 64
        bad.append(changed)
        changed = copy.deepcopy(ledger)
        changed["entries"]["demo-s002"]["segments"][0]["origin"].update(self.write_origin(
            "empty.derive.json", {"sourceSha256": "b" * 64, "outputSha256": sha["demo-s002"], "parameters": {}}))
        bad.append(changed)
        changed = copy.deepcopy(ledger)
        changed["entries"]["demo-s001"]["segments"][0]["origin"]["kind"] = "guess"
        bad.append(changed)
        changed = copy.deepcopy(ledger)
        changed["entries"]["demo-s001"]["segments"][0]["origin"].update(self.write_origin(
            "flat-receipt.json", {"sha256": sha["demo-s001"]}))
        bad.append(changed)
        changed = copy.deepcopy(ledger)
        changed["entries"]["demo-s004"]["segments"][0]["origin"].update(self.write_origin(
            "other.supplied.json", {"fileSha256": sha["demo-s001"]}))
        bad.append(changed)
        for changed in bad:
            with self.subTest(origin=changed["entries"]):
                self.write_json(path, changed)
                ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
                self.assertTrue(any(e["type"] == "picture-origin" for e in ready["errors"]), ready["errors"])

    def test_assembler_keeps_picture_events_silent_and_listed(self):
        self.make_wordless()
        _, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        import numpy as np
        work = self.ep / "work/assembly-v1"
        with mock.patch.object(assembly, "audio", side_effect=lambda *_: np.full((480, 2), 0.1, np.float32)), \
                mock.patch.object(assembly, "write_audio"):
            assembly.build_voice(self.ep, work, plan)
        levels = json.loads((work / "voice-levels.json").read_text())
        self.assertEqual([row["event"] for row in levels], [e["id"] for e in plan["events"]])
        self.assertEqual(levels[1], {"event": "demo-s002", "kind": "picture", "voice": None})
        plan["events"] = [dict(plan["events"][1], audio=plan["events"][0]["audio"])]
        with mock.patch.object(assembly, "audio") as decode, mock.patch.object(assembly, "write_audio"):
            with self.assertRaisesRegex(RuntimeError, "must not carry voice"):
                assembly.build_voice(self.ep, work, plan)
        decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
