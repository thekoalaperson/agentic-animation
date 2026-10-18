#!/usr/bin/env python3
"""Tests for the hash-bound Stable-Audio long-form mixer contract."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "absolute_sfx_longform_mixer", HERE / "mix_absolute_sfx_longform.py"
)
assert SPEC and SPEC.loader
mixer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mixer
SPEC.loader.exec_module(mixer)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class MixerFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.episode = Path(self.temporary.name) / "demo-film"
        self.work = self.episode / "work"
        self.assembly = self.work / "assembly-v1"
        self.assembly.mkdir(parents=True)
        self.runtime = 6.0
        self.timeline = "b" * 64
        self.picture = self.assembly / "picture-720p.mp4"
        self.voice = self.assembly / "voice-hi-48k24.wav"
        self.music = self.assembly / "music-wave-48k24.wav"
        self.sfx_stem = self.work / "sfx-resolve-v1/sfx-48kf32.wav"
        for path in (self.picture, self.voice, self.music, self.sfx_stem):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode("utf-8"))

        events = []
        for number in range(1, 4):
            start = (number - 1) * 2
            events.append(
                {
                    "id": f"scene-{number}",
                    "shot": number * 10,
                    "kind": "dialogue" if number == 2 else "narration",
                    "start": start,
                    "end": number * 2,
                    "duration": 2,
                }
            )
        placements = []
        for number in range(1, 3):
            movement = f"m{number}"
            base = self.work / "score" / movement
            paths = {
                "track": base.with_suffix(".mp3"),
                "sidecar": base.with_suffix(".source.json"),
                "executionManifest": self.work / "score" / "execution.json",
            }
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(path.name.encode("utf-8"))
            start = (number - 1) * self.runtime / 2
            duration = self.runtime / 2
            row = {
                "movement": movement,
                "take": number,
                "timelineStart": start,
                "timelineDuration": duration,
            }
            for key, path in paths.items():
                row[key] = str(path.relative_to(self.episode))
                row[f"{key}Sha256"] = mixer.sha256_file(path)
            placements.append(row)
        self.plan = {
            "schemaVersion": 2,
            "episode": "demo-film",
            "language": "hi",
            "timelineFingerprint": self.timeline,
            "runtimeSeconds": self.runtime,
            "events": events,
            "scorePlacements": placements,
        }
        self.plan_path = self.assembly / "edit-plan.json"
        write_json(self.plan_path, self.plan)
        segments = [
            {
                "id": f"p{number:03d}",
                "timelineStart": (number - 1) * 2,
                "duration": 2,
                "frames": 48,
                "role": "picture",
            }
            for number in range(1, 4)
        ]
        self.segments_path = self.assembly / "picture-segments.json"
        write_json(self.segments_path, segments)
        runs = [
            {
                **row,
                "memberSegmentIds": [row["id"]],
                "memberRoles": [row["role"]],
            }
            for row in segments
        ]
        write_json(self.assembly / "picture-render-runs.json", runs)
        write_json(
            self.assembly / "voice-levels.json",
            [{"event": row["id"]} for row in events],
        )
        write_json(
            self.assembly / "music-conform.json",
            [
                {
                    "movement": row["movement"],
                    "take": row["take"],
                    "source": str((self.episode / row["track"]).resolve()),
                    "timelineDuration": row["timelineDuration"],
                }
                for row in placements
            ],
        )
        write_json(
            self.assembly / "music-wave-audit.json",
            {"eventTargets": [{"shot": row["shot"]} for row in events]},
        )
        self.wave_path = self.work / "dramatic-wave-v1/dramatic-wave.json"
        write_json(
            self.wave_path,
            {
                "clockSha256": mixer.sha256_file(self.plan_path),
                "runtimeSeconds": self.runtime,
                "shots": [{"shot": row["shot"], "id": row["id"]} for row in events],
            },
        )
        self.sfx_contract = {
            "schemaVersion": 2,
            "kind": "stable-audio-absolute-sfx-final-mix-input",
            "episode": "demo-film",
            "language": "hi",
            "timelineFingerprint": self.timeline,
            "runtimeSeconds": self.runtime,
            "inputEvidence": {
                "editPlan": {
                    "path": str(self.plan_path.relative_to(self.episode)),
                    "sha256": mixer.sha256_file(self.plan_path),
                },
                "pictureSegments": {
                    "path": str(self.segments_path.relative_to(self.episode)),
                    "sha256": mixer.sha256_file(self.segments_path),
                },
                "dramaticWave": {"path": str(self.wave_path.relative_to(self.episode)),
                                 "sha256": mixer.sha256_file(self.wave_path)},
                **{"reviewed" + key.title(): {"path": str(path.relative_to(self.episode)),
                                               "sha256": mixer.sha256_file(path)}
                   for key, path in (("picture", self.picture), ("voice", self.voice), ("music", self.music))},
            },
            "sfxStem": {
                "path": str(self.sfx_stem.relative_to(self.episode)),
                "sha256": mixer.sha256_file(self.sfx_stem),
            },
            "scoreNotchAutomation": [
                {
                    "cueId": "cue-water",
                    "startSeconds": 2.0,
                    "endSeconds": 2.7,
                    "scoreGainDb": -8.0,
                    "attackSeconds": 0.08,
                    "releaseSeconds": 0.22,
                },
                {
                    "cueId": "cue-stone",
                    "startSeconds": 2.2,
                    "endSeconds": 2.9,
                    "scoreGainDb": -5.0,
                    "attackSeconds": 0.08,
                    "releaseSeconds": 0.22,
                },
            ],
        }
        self.sfx_contract_path = self.work / "sfx-resolve-v1/final-mix-input.json"
        write_json(self.sfx_contract_path, self.sfx_contract)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def probe(self, path: Path) -> dict:
        if path.resolve() == self.picture.resolve():
            return {
                "durationSeconds": self.runtime,
                "audio": [],
                "video": [
                    {
                        "codec": "h264",
                        "width": 1280,
                        "height": 720,
                        "frameRate": "24/1",
                        "frames": 144,
                        "durationSeconds": self.runtime,
                    }
                ],
            }
        codec = (
            "pcm_f32le"
            if path.resolve() == self.sfx_stem.resolve()
            else "pcm_s24le"
        )
        return {
            "durationSeconds": self.runtime,
            "video": [],
            "audio": [
                {
                    "codec": codec,
                    "sampleRate": 48_000,
                    "channels": 2,
                    "durationSeconds": self.runtime,
                }
            ],
        }

    @staticmethod
    def accept_sfx(_episode: Path, loaded: dict) -> dict:
        return loaded

    def build(self) -> dict:
        return mixer.build_mix_contract(
            self.episode,
            sfx_input_path=self.sfx_contract_path,
            edit_plan_path=self.plan_path,
            picture_path=self.picture,
            voice_path=self.voice,
            music_path=self.music,
            probe=self.probe,
            sfx_revalidator=self.accept_sfx,
        )

    def test_builds_hash_bound_non_delivery_contract(self) -> None:
        contract = self.build()
        self.assertEqual(
            contract["kind"], "stable-audio-absolute-longform-mix-contract"
        )
        self.assertEqual(contract["expectedVideoFrames"], 144)
        self.assertEqual(contract["expectedAudioFrames"], 288_000)
        self.assertFalse(contract["safety"]["usesShortest"])
        self.assertFalse(contract["safety"]["creativeApprovalGrantedByThisContract"])
        self.assertIsNone(contract['contextReview'])
        self.assertEqual(
            contract["inputEvidence"]["picture"]["sha256"],
            mixer.sha256_file(self.picture),
        )
        expression = contract["scoreNotchEnvelope"]["ffmpegVolumeExpression"]
        self.assertIn("min(", expression)
        self.assertIn("pow(10", expression)

    def test_score_and_wave_drift_fail_closed(self) -> None:
        first = self.plan["scorePlacements"][0]
        (self.episode / first["track"]).write_bytes(b"changed")
        with self.assertRaisesRegex(mixer.MixError, "track changed"):
            self.build()

        (self.episode / first["track"]).write_bytes(Path(first["track"]).name.encode())
        wave = json.loads(self.wave_path.read_text())
        wave["clockSha256"] = "c" * 64
        write_json(self.wave_path, wave)
        with self.assertRaisesRegex(mixer.MixError, "dramaticWave"):
            self.build()

    def rewrite_plan(self, **changes) -> None:
        plan = {**self.plan, **changes}
        write_json(self.plan_path, plan)
        write_json(self.assembly / "music-conform.json", [
            {"movement": row["movement"], "silence": True, "timelineDuration": row["timelineDuration"]}
            if row.get("silence") else
            {"movement": row["movement"], "take": row["take"],
             "source": str((self.episode / row["track"]).resolve()),
             "timelineDuration": row["timelineDuration"]}
            for row in plan["scorePlacements"]])
        wave = json.loads(self.wave_path.read_text())
        wave["clockSha256"] = mixer.sha256_file(self.plan_path)
        write_json(self.wave_path, wave)
        sfx = copy.deepcopy(self.sfx_contract)
        sfx["inputEvidence"]["editPlan"]["sha256"] = mixer.sha256_file(self.plan_path)
        sfx["inputEvidence"]["dramaticWave"]["sha256"] = mixer.sha256_file(self.wave_path)
        write_json(self.sfx_contract_path, sfx)

    def test_gaps_between_placements_are_silence(self) -> None:
        first, second = copy.deepcopy(self.plan["scorePlacements"])
        first["timelineDuration"] = 1.5
        second["timelineStart"], second["timelineDuration"] = 4.0, 1.0
        self.rewrite_plan(scorePlacements=[first, second])
        self.assertEqual(self.build()["runtimeSeconds"], self.runtime)
        silence = {"movement": "hush", "silence": True, "timelineStart": 1.5,
                   "timelineDuration": 2.5, "reason": "No one plays here."}
        self.rewrite_plan(scorePlacements=[first, silence, second])
        self.build()

    def test_overlap_runtime_escape_and_named_silence_track_are_refused(self) -> None:
        first, second = copy.deepcopy(self.plan["scorePlacements"])
        cases = {
            "overlap": [first, {**second, "timelineStart": 2.5}],
            "runs past": [first, {**second, "timelineDuration": 3.5}],
            "must not name": [first, {"movement": "hush", "silence": True, "timelineStart": 3.0,
                                      "timelineDuration": 1.0, "track": second["track"]}],
        }
        for message, rows in cases.items():
            with self.subTest(message):
                self.rewrite_plan(scorePlacements=rows)
                with self.assertRaisesRegex(mixer.MixError, message):
                    self.build()

    def test_an_empty_score_needs_an_unscored_declaration(self) -> None:
        self.rewrite_plan(scorePlacements=[])
        with self.assertRaisesRegex(mixer.MixError, "unscored: true"):
            self.build()
        self.rewrite_plan(scorePlacements=[], unscored=True)
        self.build()
        self.rewrite_plan(unscored=True)
        with self.assertRaisesRegex(mixer.MixError, "only silence"):
            self.build()

    def test_render_run_must_preserve_semantic_segments_exactly_once(self) -> None:
        path = self.assembly / "picture-render-runs.json"
        runs = json.loads(path.read_text())
        runs.pop()
        write_json(path, runs)
        with self.assertRaisesRegex(mixer.MixError, "semantic segment"):
            self.build()

    def test_malformed_score_notch_is_rejected(self) -> None:
        contract = copy.deepcopy(self.sfx_contract)
        contract["scoreNotchAutomation"][0].pop("scoreGainDb")
        write_json(self.sfx_contract_path, contract)
        with self.assertRaisesRegex(mixer.MixError, "notch gain"):
            self.build()

    def test_saved_contract_is_rebuilt_before_execution(self) -> None:
        contract = self.build()
        saved = self.work / "assembly-v1/absolute-mix-contract.json"
        write_json(saved, contract)
        current = mixer.verify_saved_contract(
            self.episode,
            saved,
            probe=self.probe,
            sfx_revalidator=self.accept_sfx,
        )
        self.assertEqual(current, contract)
        self.voice.write_bytes(b"changed voice")
        with self.assertRaisesRegex(mixer.MixError, "fresh preflight"):
            mixer.verify_saved_contract(
                self.episode,
                saved,
                probe=self.probe,
                sfx_revalidator=self.accept_sfx,
            )

    def test_filter_graph_applies_notches_only_to_music_bus(self) -> None:
        graph = mixer.filter_graph(self.build())
        self.assertTrue(graph.startswith("[0:a]aformat="))
        self.assertIn("asplit=3[voice][mkey][skey]", graph)
        self.assertIn("[music][mkey]sidechaincompress", graph)
        self.assertIn("[sfx][skey]sidechaincompress", graph)
        self.assertEqual(graph.count("aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"),3)
        self.assertNotIn("[voice]volume=", graph)
        self.assertNotIn("shortest", graph)

    def test_filter_graph_parses_in_ffmpeg_when_available(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is unavailable")
        graph = mixer.filter_graph(self.build())
        command = [ffmpeg, "-v", "error"]
        for _ in range(3):
            command.extend(
                ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            )
        command.extend(
            [
                "-filter_complex",
                graph,
                "-map",
                "[out]",
                "-t",
                "0.05",
                "-f",
                "null",
                "-",
            ]
        )
        result = subprocess.run(command, capture_output=True, check=False)
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode("utf-8", errors="replace"),
        )

    def test_body_mux_copies_picture_and_never_uses_shortest(self) -> None:
        command = mixer.body_mux_command(
            self.picture,
            self.work / "premaster.wav",
            self.work / "body.mp4",
            "loudnorm=I=-14:TP=-1.5:LRA=11",
            self.runtime,
        )
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertNotIn("-shortest", command)
        self.assertEqual(command[command.index("-t") + 1], "6.000000000")

    def test_output_guard_refuses_deliverables_and_external_paths(self) -> None:
        with self.assertRaisesRegex(mixer.MixError, "deliverables"):
            mixer.work_output(
                self.episode, "deliverables/mix-contract.json", "contract"
            )
        with self.assertRaisesRegex(mixer.MixError, "escapes"):
            mixer.work_output(self.episode, "../mix-contract.json", "contract")


if __name__ == "__main__":
    unittest.main()
