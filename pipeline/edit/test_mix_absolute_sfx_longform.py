#!/usr/bin/env python3
"""Tests for the hash-bound Stable-Audio long-form mixer contract."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
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

    def test_v1_contract_is_unchanged_without_authored_settings(self) -> None:
        contract = self.build()
        self.assertEqual(contract["settings"], mixer.default_settings())
        self.assertNotIn("settingsVersion", contract["settings"])
        explicit = mixer.build_mix_contract(
            self.episode, sfx_input_path=self.sfx_contract_path, edit_plan_path=self.plan_path,
            picture_path=self.picture, voice_path=self.voice, music_path=self.music,
            probe=self.probe, sfx_revalidator=self.accept_sfx, settings=mixer.default_settings())
        self.assertEqual(json.dumps(contract, sort_keys=True), json.dumps(explicit, sort_keys=True))

    def test_v2_settings_are_bounded_recorded_and_rebuilt(self) -> None:
        authored = {"settingsVersion": 2, "musicGainDb": 3.0, "targetLra": "measured",
                    "musicDuck": {"threshold": 0.05, "ratio": 2.5, "attackMs": 30, "releaseMs": 800},
                    "targetIntegratedLufs": -20, "refuseDynamicMastering": True,
                    "reason": "A quiet film that should stay quiet."}
        contract = mixer.build_mix_contract(
            self.episode, sfx_input_path=self.sfx_contract_path, edit_plan_path=self.plan_path,
            picture_path=self.picture, voice_path=self.voice, music_path=self.music,
            probe=self.probe, sfx_revalidator=self.accept_sfx, settings=authored)
        settings = contract["settings"]
        self.assertEqual(settings["musicDuck"]["ratio"], 2.5)
        self.assertEqual(settings["sfxDuck"], mixer.default_settings()["sfxDuck"])
        self.assertEqual(settings["targetTruePeakDbtp"], -1.5)
        self.assertNotEqual(contract["contractFingerprint"], self.build()["contractFingerprint"])
        saved = self.assembly / "absolute-mix-contract.json"
        write_json(saved, contract)
        self.assertEqual(mixer.verify_saved_contract(
            self.episode, saved, probe=self.probe, sfx_revalidator=self.accept_sfx), contract)
        self.assertIn("ratio=2.5", mixer.filter_graph(contract))
        self.assertEqual(mixer.second_pass_lra(settings, {"input_lra": "17.40"}), 17.4)
        self.assertEqual(mixer.second_pass_lra(settings, {"input_lra": "0.00"}), 1.0)
        self.assertEqual(mixer.measurement_lra(settings), 11.0)

    def test_v2_settings_out_of_bounds_or_unversioned_are_refused(self) -> None:
        cases = {
            "settingsVersion 2": {"musicGainDb": 3.0},
            "-6..6": {"settingsVersion": 2, "musicGainDb": 9.0},
            "ratio 1-12": {"settingsVersion": 2, "sfxDuck": {"threshold": 0.02, "ratio": 20,
                                                             "attackMs": 8, "releaseMs": 360}},
            "-27..-12": {"settingsVersion": 2, "targetIntegratedLufs": -8},
            "-3..-1": {"settingsVersion": 2, "targetTruePeakDbtp": -0.5},
            "measured": {"settingsVersion": 2, "targetLra": "wide"},
            "boolean": {"settingsVersion": 2, "refuseDynamicMastering": "yes"},
            "unknown": {"settingsVersion": 2, "limiter": True},
        }
        for message, settings in cases.items():
            with self.subTest(message), self.assertRaisesRegex(mixer.MixError, message):
                mixer.validate_settings(settings)

    def test_second_pass_type_is_recorded_and_dynamic_names_choices(self) -> None:
        record = mixer.second_pass_record(
            {"normalization_type": "dynamic", "input_lra": "18.20", "output_lra": "2.00",
             "input_i": "-30.1", "output_i": "-14.2"}, 11.0)
        self.assertEqual((record["normalizationType"], record["inputLra"], record["outputLra"]),
                         ("dynamic", 18.2, 2.0))
        message = mixer.dynamic_mastering_message(record)
        for choice in ("measured", "fixedGainMaster", "refuseDynamicMastering off"):
            self.assertIn(choice, message)
        with self.assertRaisesRegex(mixer.MixError, "normalization type"):
            mixer.second_pass_record({"input_lra": "1"}, 11.0)

    def test_info_log_level_lets_loudnorm_report_a_dynamic_fallback(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is unavailable")
        # Short loud bursts over a quiet bed: one gain cannot meet -14 LUFS at -1.5 dBTP.
        source = self.work / "wide.wav"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                        "aevalsrc='if(lt(mod(t,10),0.02),0.95,0.03)*sin(2*PI*440*t)':s=48000:d=40",
                        "-ac", "2", str(source)], check=True)
        settings = mixer.default_settings()
        first = mixer.loudness(source, settings)
        norm = (f"loudnorm=I=-14.0:TP=-1.5:LRA=11.0:measured_I={first['input_i']}:"
                f"measured_TP={first['input_tp']}:measured_LRA={first['input_lra']}:"
                f"measured_thresh={first['input_thresh']}:offset={first['target_offset']}:"
                "linear=true:print_format=json")
        result = subprocess.run([ffmpeg, "-nostats", "-v", "info", "-i", str(source), "-af", norm,
                                 "-f", "null", "-"], capture_output=True, check=True)
        record = mixer.second_pass_record(mixer.parse_loudnorm(result.stderr), 11.0)
        self.assertEqual(record["normalizationType"], "dynamic")

    def test_plan_following_flags_only_contradictions_of_the_plan(self) -> None:
        events = [{"id": f"demo-s00{i}", "shot": i, "kind": "narration", "start": float(i - 1),
                   "end": float(i)} for i in range(1, 5)]
        levels = {"voice": [-20, -20, -20, -20], "music": [-30, -34, -28, -40],
                  "premaster": [-19, -19, -19, -19], "master": [-14, -14, -11, -14]}

        def measure(path: Path):
            values = levels[path.name]
            return [(round(t / 10 - 0.3, 6), round(t / 10 + 0.1, 6), values[min(3, int(t // 10))])
                    for t in range(0, 40)]

        buses = {name: Path(name) for name in levels}
        result = mixer.plan_following(
            events, buses, [{"shots": [4, 4], "level": 0, "reason": "a held breath"}],
            [{"shot": i, "envelopeTarget": v} for i, v in enumerate((0.3, 0.6, 0.6, 0.2), 1)],
            measure=measure)
        self.assertEqual([row["musicLufs"] for row in result["events"]], [-30.0, -34.0, -28.0, -40.0])
        kinds = {(flag["event"], flag["kind"]) for flag in result["flags"]}
        self.assertEqual(kinds, {("demo-s002", "authored-rise-falls"), ("demo-s004", "zero-breath-sounds"),
                                 ("demo-s003", "mastering-moved-event-level")})
        self.assertIsNone(result["listeningVerdict"])

    def test_momentary_loudness_reads_ffmpeg_and_hears_silence(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is unavailable")
        path = self.work / "tone-then-silence.wav"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                        "aevalsrc='if(lt(t,2),0.1*sin(2*PI*440*t),0)':s=48000:d=4",
                        "-ac", "2", str(path)], check=True)
        windows = mixer.momentary_loudness(path)
        tone = mixer.event_loudness(windows, 0.0, 2.0)
        self.assertTrue(-30 < tone < -15, tone)
        self.assertLessEqual(mixer.event_loudness(windows, 2.0, 4.0), mixer.SILENT_LUFS)
        self.assertIsNone(mixer.event_loudness(windows, 1.0, 1.2))

    def real_media(self) -> dict:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or b"libx264" not in subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                                                          capture_output=True).stdout:
            self.skipTest("ffmpeg with libx264 is unavailable")
        quiet = "-v", "error", "-y"
        subprocess.run([ffmpeg, *quiet, "-f", "lavfi", "-i", "color=c=gray:s=1280x720:r=24",
                        "-t", "6", "-c:v", "libx264", "-an", str(self.picture)], check=True)
        for path, expression, codec in ((self.voice, "0.2*sin(2*PI*220*t)", "pcm_s24le"),
                                        (self.music, "0.05*sin(2*PI*330*t)", "pcm_s24le"),
                                        (self.sfx_stem, "0", "pcm_f32le")):
            subprocess.run([ffmpeg, *quiet, "-f", "lavfi", "-i", f"aevalsrc='{expression}':s=48000:d=6",
                            "-ac", "2", "-c:a", codec, str(path)], check=True)
        self.sfx_contract["sfxStem"]["sha256"] = mixer.sha256_file(self.sfx_stem)
        write_json(self.sfx_contract_path, self.sfx_contract)
        return self.build()

    def test_execute_records_the_second_pass_and_refuses_dynamic_only_when_asked(self) -> None:
        contract = self.real_media()
        contract_path = self.assembly / "absolute-mix-contract.json"
        write_json(contract_path, contract)
        warnings = io.StringIO()
        with contextlib.redirect_stderr(warnings):
            audit = mixer.execute_mix(self.episode, contract_path, contract)
        second = audit["loudnormSecondPass"]
        self.assertEqual("dynamic gain" in warnings.getvalue(), second["normalizationType"] == "dynamic")
        self.assertIn(second["normalizationType"], ("linear", "dynamic"))
        self.assertEqual(second["targetLra"], 11.0)
        bus = json.loads((self.assembly / "absolute-mix-bus-audit.json").read_text())
        self.assertEqual([row["event"] for row in bus["planFollowing"]["events"]],
                         ["scene-1", "scene-2", "scene-3"])
        self.assertIsNone(bus["planFollowing"]["listeningVerdict"])
        if second["normalizationType"] == "dynamic":
            master = self.episode / contract["outputs"]["bodyMaster"]
            master.unlink()
            refusing = copy.deepcopy(contract)
            refusing["settings"] = mixer.validate_settings({"settingsVersion": 2, "refuseDynamicMastering": True})
            with self.assertRaisesRegex(mixer.MixError, "dynamic gain"):
                mixer.execute_mix(self.episode, contract_path, refusing)
            self.assertFalse(master.exists())

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
        self.assertEqual(command[:4], ["ffmpeg", "-y", "-v", "error"])
        loud = mixer.body_mux_command(self.picture, self.work / "premaster.wav", self.work / "body.mp4",
                                      "loudnorm=I=-14:TP=-1.5:LRA=11", self.runtime, log_level="info")
        self.assertEqual([part for part in loud if part not in ("-nostats", "info")],
                         [part for part in command if part != "error"])

    def test_output_guard_refuses_deliverables_and_external_paths(self) -> None:
        with self.assertRaisesRegex(mixer.MixError, "deliverables"):
            mixer.work_output(
                self.episode, "deliverables/mix-contract.json", "contract"
            )
        with self.assertRaisesRegex(mixer.MixError, "escapes"):
            mixer.work_output(self.episode, "../mix-contract.json", "contract")


if __name__ == "__main__":
    unittest.main()
