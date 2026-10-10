#!/usr/bin/env python3
"""Tests for the work-only, final-master-bound bilingual SRT exporter."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "export_bilingual_subtitles_for_test", HERE / "export_bilingual_subtitles.py"
)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BilingualSubtitleExportTests(unittest.TestCase):
    expected_scenes = 144

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.episode = Path(self.temporary.name) / "demo-fixture"
        self.assembly = self.episode / "work/assembly-v1"
        self.plan_path = self.assembly / "edit-plan.json"
        self.hi_path = self.assembly / "subtitle-hi-timing.json"
        self.en_path = self.assembly / "subtitle-en-timing.json"
        self.master = self.episode / "work/final-master-v1/final-720p.mp4"
        self.clock_path = self.episode / "work/final-master-v1/clock-audit.json"
        self.sync_path = self.episode / "work/final-master-v1/voice-sync-audit.json"
        self.output = self.episode / "work/subtitles-v1"
        self.master.parent.mkdir(parents=True, exist_ok=True)
        self.master.write_bytes(b"synthetic-final-master-fixture")
        write_json(
            self.episode / "episode.json",
            {"slug": "demo_episode"},
        )
        self._write_valid_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_valid_fixture(self) -> None:
        fingerprint = "a" * 64
        events = []
        hi_shots = []
        en_shots = []
        for index in range(self.expected_scenes):
            scene_id = f"demo-s{index + 1:03d}"
            dialogue = index % 12 == 0
            event = {
                "id": scene_id,
                "kind": "dialogue" if dialogue else "narration",
                "start": float(index),
                "duration": 1.0,
                "end": float(index + 1),
                "subtitleAudioAt": 0.1,
                "subtitleAudioDuration": 0.75,
                "subtitleTextHi": f"यह दृश्य संख्या {index + 1} है।",
                "subtitleTextEn": f"This is scene number {index + 1}.",
            }
            events.append(event)
            common = {
                "id": scene_id,
                "start": float(index),
                "dur": 1.0,
                "audioAt": 0.1,
                "audioDuration": 0.75,
                "dialogue": dialogue,
                "speaker": "CHARACTER ONE" if dialogue else None,
                "sourceLanguage": "hi",
            }
            hi_shots.append({**common, "text": event["subtitleTextHi"]})
            en_shots.append({**common, "text": event["subtitleTextEn"]})
        plan = {
            "schemaVersion": 2,
            "episode": "DEMO",
            "language": "hi",
            "timelineFingerprint": fingerprint,
            "runtimeSeconds": float(self.expected_scenes),
            "deliveryFps": 24,
            "events": events,
            "pictureBlocks": [],
            "scorePlacements": [],
        }
        write_json(self.plan_path, plan)
        for language, path, shots in (
            ("hi", self.hi_path, hi_shots),
            ("en", self.en_path, en_shots),
        ):
            write_json(
                path,
                {
                    "schemaVersion": 1,
                    "kind": "hindi-master-subtitle-timing-input",
                    "language": language,
                    "masterLanguage": "hi",
                    "duration": float(self.expected_scenes),
                    "timelineFingerprint": fingerprint,
                    "shots": shots,
                },
            )
        master_sha = exporter.sha256_file(self.master)
        write_json(
            self.clock_path,
            {
                "master": str(self.master.resolve()),
                "sha256": master_sha,
                "frames": (self.expected_scenes + 6) * 24,
                "fps": 24,
                "durationSeconds": float(self.expected_scenes + 6),
                "decodeTimestampsStrictlyIncreasing": True,
                "decode": "passed",
                "passed": True,
            },
        )
        write_json(
            self.sync_path,
            {
                "master": str(self.master.resolve()),
                "masterSha256": master_sha,
                "planSha256": exporter.sha256_file(self.plan_path),
                "events": self.expected_scenes,
                "timingPassed": True,
                "voicePresencePassed": True,
                "passed": True,
                "results": [
                    {"event": event["id"], "pass": True} for event in events
                ],
            },
        )

    def export(self, **overrides):
        arguments = {
            "master": self.master,
            "clock_audit_path": self.clock_path,
            "voice_sync_audit_path": self.sync_path,
            "output_dir": self.output,
            "expected_scenes": self.expected_scenes,
        }
        arguments.update(overrides)
        return exporter.export_subtitles(self.episode, **arguments)

    def test_exports_144_scene_bilingual_srt_with_final_hash_lineage(self) -> None:
        audit = self.export()
        hi = self.output / "demo_episode-hi.srt"
        en = self.output / "demo_episode-en.srt"
        self.assertEqual(audit["sceneCount"], 144)
        self.assertEqual(audit["cueCount"], 144)
        self.assertEqual(audit["timelineFingerprint"], "a" * 64)
        self.assertEqual(audit["finalMaster"]["sha256"], exporter.sha256_file(self.master))
        self.assertEqual(audit["finalMaster"]["durationSeconds"], 150.0)
        self.assertEqual(audit["finalMaster"]["tailAfterTimelineSeconds"], 6.0)
        self.assertTrue(hi.is_file())
        self.assertTrue(en.is_file())
        self.assertIn("यह दृश्य संख्या 1 है।", hi.read_text(encoding="utf-8"))
        self.assertNotRegex(en.read_text(encoding="utf-8"), exporter.DEVANAGARI_RE)
        hi_timings = [line for line in hi.read_text().splitlines() if " --> " in line]
        en_timings = [line for line in en.read_text().splitlines() if " --> " in line]
        self.assertEqual(hi_timings, en_timings)
        self.assertFalse(audit["deliveryApproval"])
        self.assertFalse(audit["publicationAuthorization"])

    def test_refuses_deliverables_or_non_work_output(self) -> None:
        for target in (
            self.episode / "deliverables/subtitles",
            self.episode.parent / "outside",
        ):
            with self.subTest(target=target):
                with self.assertRaisesRegex(exporter.SubtitleExportError, "deliverables|beneath episode work"):
                    self.export(output_dir=target)
                self.assertFalse(target.exists())

    def test_refuses_missing_or_overlapping_scenes(self) -> None:
        hi = read_json(self.hi_path)
        hi["shots"].pop()
        write_json(self.hi_path, hi)
        with self.assertRaisesRegex(exporter.SubtitleExportError, "cover all 144 scenes"):
            self.export()
        self._write_valid_fixture()
        plan = read_json(self.plan_path)
        plan["events"][1]["start"] = 0.5
        plan["events"][1]["end"] = 1.5
        write_json(self.plan_path, plan)
        sync = read_json(self.sync_path)
        sync["planSha256"] = exporter.sha256_file(self.plan_path)
        write_json(self.sync_path, sync)
        with self.assertRaisesRegex(exporter.SubtitleExportError, "Overlapping edit-plan scenes"):
            self.export()

    def test_refuses_english_devanagari_leakage(self) -> None:
        en = read_json(self.en_path)
        en["shots"][20]["text"] = "This names दीपक without an authored translation."
        write_json(self.en_path, en)
        plan = read_json(self.plan_path)
        plan["events"][20]["subtitleTextEn"] = en["shots"][20]["text"]
        write_json(self.plan_path, plan)
        sync = read_json(self.sync_path)
        sync["planSha256"] = exporter.sha256_file(self.plan_path)
        write_json(self.sync_path, sync)
        with self.assertRaisesRegex(exporter.SubtitleExportError, "contains Devanagari"):
            self.export()

    def test_refuses_stale_timing_fingerprint(self) -> None:
        en = read_json(self.en_path)
        en["timelineFingerprint"] = "b" * 64
        write_json(self.en_path, en)
        with self.assertRaisesRegex(exporter.SubtitleExportError, "Stale en timing"):
            self.export()

    def test_refuses_master_hash_or_duration_drift(self) -> None:
        clock = read_json(self.clock_path)
        clock["sha256"] = "0" * 64
        write_json(self.clock_path, clock)
        with self.assertRaisesRegex(exporter.SubtitleExportError, "hash differs"):
            self.export()
        self._write_valid_fixture()
        clock = read_json(self.clock_path)
        clock["frames"] = (self.expected_scenes - 1) * 24
        clock["durationSeconds"] = float(self.expected_scenes - 1)
        write_json(self.clock_path, clock)
        with self.assertRaisesRegex(exporter.SubtitleExportError, "ends before"):
            self.export()

    def test_requires_passed_voice_sync_bound_to_exact_plan(self) -> None:
        sync = read_json(self.sync_path)
        sync["planSha256"] = "f" * 64
        write_json(self.sync_path, sync)
        with self.assertRaisesRegex(exporter.SubtitleExportError, "not bound to the final edit plan"):
            self.export()
        self._write_valid_fixture()
        sync = read_json(self.sync_path)
        sync["results"][50]["pass"] = False
        write_json(self.sync_path, sync)
        with self.assertRaisesRegex(exporter.SubtitleExportError, "missing/pending"):
            self.export()

    def test_different_text_lengths_still_share_one_cue_grid(self) -> None:
        hi = read_json(self.hi_path)
        en = read_json(self.en_path)
        plan = read_json(self.plan_path)
        long_hi = (
            "यह एक विस्तृत दृश्य है जिसमें नदी का किनारा, शाम की हल्की हवा "
            "और पुराने पुल पर रुकी साइकिलें धीरे धीरे रोशन होती जाती हैं।"
        )
        long_en = "The river bank glows while a farmer walks home."
        hi["shots"][0]["text"] = long_hi
        en["shots"][0]["text"] = long_en
        plan["events"][0]["subtitleTextHi"] = long_hi
        plan["events"][0]["subtitleTextEn"] = long_en
        write_json(self.hi_path, hi)
        write_json(self.en_path, en)
        write_json(self.plan_path, plan)
        sync = read_json(self.sync_path)
        sync["planSha256"] = exporter.sha256_file(self.plan_path)
        write_json(self.sync_path, sync)
        audit = self.export()
        self.assertGreater(audit["cueCount"], 144)
        hi_srt = (self.output / "demo_episode-hi.srt").read_text()
        en_srt = (self.output / "demo_episode-en.srt").read_text()
        self.assertEqual(
            [line for line in hi_srt.splitlines() if " --> " in line],
            [line for line in en_srt.splitlines() if " --> " in line],
        )

    def test_refuses_cue_that_collapses_at_srt_millisecond_precision(self) -> None:
        with self.assertRaisesRegex(exporter.SubtitleExportError, "millisecond rounding"):
            exporter.validate_cues(
                [exporter.Cue(0.0, 0.0004, "Too brief.", "demo-s001")],
                "en",
            )

    def test_authored_pairs_preserve_words_and_weighted_shared_clock(self) -> None:
        common={'id':'demo-s122','start':5,'dur':9,'audioAt':0,'audioDuration':9,
                'cueWeights':[1,2]}
        hi={**common,'text':'पर हल्की बारिश उस सूखे खेत को हरा कर सकती है।',
            'cueParts':['पर हल्की बारिश','उस सूखे खेत को हरा कर सकती है।']}
        en={**common,'text':'But gentle rain can turn that field green.',
            'cueParts':['But gentle rain','can turn that field green.']}
        hc,ec,coverage=exporter.build_synchronized_cues([hi],[en])
        self.assertEqual([(c.start,c.end) for c in hc],[(5,8),(8,14)])
        self.assertEqual([(c.start,c.end) for c in hc],[(c.start,c.end) for c in ec])
        self.assertEqual(ec[0].text,'But gentle rain')
        self.assertEqual(coverage[0]['partitionPolicy'],'authored-paired-clauses')
        en['cueParts'][0]='But rain'
        with self.assertRaisesRegex(exporter.SubtitleExportError,'changed subtitle words'):
            exporter.build_synchronized_cues([hi],[en])

    def test_timing_document_cannot_inject_unbound_clause_plan(self) -> None:
        doc=read_json(self.hi_path)
        doc['shots'][0].update(cueParts=[doc['shots'][0]['text']],cueWeights=[1])
        write_json(self.hi_path,doc)
        with self.assertRaisesRegex(exporter.SubtitleExportError,'Unbound authored subtitle'):
            self.export()


if __name__ == "__main__":
    unittest.main()
