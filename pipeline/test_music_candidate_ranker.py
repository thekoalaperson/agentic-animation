from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

try:
    from pipeline import music_candidate_ranker as ranker
except ImportError:
    import music_candidate_ranker as ranker


SAMPLE_RATE = 22050


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -0.999, 0.999)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())


def tone(frequency: float, seconds: float = 12.0, amplitude: float = 0.35) -> np.ndarray:
    time = np.arange(round(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * math.pi * frequency * time)


def pulse_track(seconds: float = 16.0, bpm: float = 120.0) -> np.ndarray:
    size = round(seconds * SAMPLE_RATE)
    result = np.zeros(size, dtype=np.float64)
    rng = np.random.default_rng(4301)
    interval = 60.0 / bpm
    burst_size = round(0.075 * SAMPLE_RATE)
    envelope = np.exp(-np.linspace(0, 7, burst_size))
    for number, start_seconds in enumerate(np.arange(0.25, seconds - 0.1, interval)):
        start = round(start_seconds * SAMPLE_RATE)
        stop = min(size, start + burst_size)
        burst = rng.normal(0, 1, stop - start)
        # Alternating spectral color lets spectral flux and energy rise both
        # observe the synthetic onset without making the test nondeterministic.
        carrier_frequency = 330.0 if number % 2 == 0 else 660.0
        carrier_time = np.arange(stop - start) / SAMPLE_RATE
        colored = 0.55 * burst + 0.45 * np.sin(
            2 * math.pi * carrier_frequency * carrier_time
        )
        result[start:stop] += 0.70 * colored * envelope[:stop - start]
    return result


def features(audio: np.ndarray) -> dict:
    return ranker.extract_features(
        audio, SAMPLE_RATE, 2048, 512, 12, 6, 180.0
    )


class SyntheticFeatureTests(unittest.TestCase):
    def test_low_frequency_energy_distinguishes_weight_from_brightness(self) -> None:
        low = features(tone(70.0))
        high = features(tone(2200.0))
        self.assertGreater(
            low["summary"]["lowFrequencyEnergyRatioMean"],
            high["summary"]["lowFrequencyEnergyRatioMean"] + 0.75,
        )
        self.assertLess(
            low["summary"]["spectralCentroidHzMean"],
            high["summary"]["spectralCentroidHzMean"],
        )
        rng = np.random.default_rng(19)
        noise = features(0.20 * rng.normal(size=round(12.0 * SAMPLE_RATE)))
        self.assertGreater(
            noise["summary"]["spectralSpreadHzMean"],
            high["summary"]["spectralSpreadHzMean"],
        )

    def test_onsets_and_periodicity_distinguish_pulse_from_drone(self) -> None:
        pulse = features(pulse_track())
        drone = features(tone(220.0, seconds=16.0))
        self.assertGreater(
            pulse["summary"]["onsetDensityPerSecond"],
            drone["summary"]["onsetDensityPerSecond"] + 0.5,
        )
        self.assertGreater(pulse["summary"]["tempoConfidence01"], 0.10)
        self.assertIsNotNone(pulse["summary"]["tempoBpmCandidate"])
        bpm = pulse["summary"]["tempoBpmCandidate"]
        self.assertTrue(55 <= bpm <= 65 or 110 <= bpm <= 130, bpm)

    def test_fixed_rms_mapping_does_not_turn_a_flat_track_into_a_ramp(self) -> None:
        result = features(tone(440.0))
        curve = np.asarray(result["trajectory"]["rmsEnergy"])
        self.assertLess(float(np.max(curve) - np.min(curve)), 0.01)

    def test_breathing_and_ending_tail_are_measured_separately(self) -> None:
        constant = tone(330.0, seconds=20.0)
        breathing = constant.copy()
        for start_seconds in (2.0, 5.0, 8.0, 11.0, 14.0):
            start = round(start_seconds * SAMPLE_RATE)
            breathing[start:start + round(0.65 * SAMPLE_RATE)] = 0.0
        fading = constant.copy()
        fade_samples = round(5.0 * SAMPLE_RATE)
        fading[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
        fading[-round(0.55 * SAMPLE_RATE):] = 0.0
        constant_features = features(constant)
        breathing_features = features(breathing)
        fading_features = features(fading)
        self.assertGreater(
            breathing_features["summary"]["breathingFraction"],
            constant_features["summary"]["breathingFraction"] + 0.05,
        )
        self.assertGreater(
            fading_features["summary"]["endingTailDropDb"],
            constant_features["summary"]["endingTailDropDb"] + 10.0,
        )
        self.assertLess(
            fading_features["summary"]["endingAbruptness01"],
            constant_features["summary"]["endingAbruptness01"],
        )
        self.assertGreater(fading_features["summary"]["terminalSilenceSeconds"], 0.35)

    def test_dynamic_sections_have_more_range_and_contrast_than_stationary_tone(self) -> None:
        stationary = tone(330.0, seconds=16.0, amplitude=0.24)
        dynamic = stationary.copy()
        quarter = len(dynamic) // 4
        dynamic[:quarter] *= 0.06
        dynamic[quarter:2 * quarter] *= 0.95
        dynamic[2 * quarter:3 * quarter] *= 0.18
        dynamic[3 * quarter:] *= np.linspace(0.25, 1.0, len(dynamic) - 3 * quarter)
        stationary_features = features(stationary)
        dynamic_features = features(dynamic)
        self.assertGreater(
            dynamic_features["summary"]["rmsDynamicRangeDb"],
            stationary_features["summary"]["rmsDynamicRangeDb"] + 10.0,
        )
        self.assertGreater(
            dynamic_features["summary"]["sectionContrastRaw"],
            stationary_features["summary"]["sectionContrastRaw"] + 0.05,
        )

    def test_trajectory_fit_prefers_aligned_over_reversed_energy(self) -> None:
        bins = 12
        ramp = np.linspace(0.1, 0.9, bins)
        target = {
            "proxies": {name: ramp.copy() for name in ranker.TRAJECTORY_FEATURES},
            "globals": {name: 0.5 for name in ranker.GLOBAL_FEATURES},
            "ending": {name: 0.5 for name in ranker.ENDING_FEATURES},
        }
        normalized = {name: 0.5 for name in ranker.GLOBAL_FEATURES + ranker.ENDING_FEATURES}
        aligned = {
            "trajectory": {name: ramp.copy() for name in ranker.TRAJECTORY_FEATURES},
            "normalized": normalized,
        }
        reversed_features = {
            "trajectory": {name: ramp[::-1].copy() for name in ranker.TRAJECTORY_FEATURES},
            "normalized": normalized,
        }
        weights = {name: 1.0 for name in (
            ranker.TRAJECTORY_FEATURES + ranker.GLOBAL_FEATURES + ranker.ENDING_FEATURES
        )}
        self.assertGreater(
            ranker.fit_candidate(aligned, target, weights)["technicalFitScore"],
            ranker.fit_candidate(reversed_features, target, weights)["technicalFitScore"] + 10,
        )


class ManifestFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.creative_path = root / "creative.json"
        self.anchors_path = root / "anchors.json"
        self.execution_path = root / "execution.json"
        self.manifest_path = root / "analysis-manifest.json"
        self.payload = {
            "model": "V5_5",
            "customMode": True,
            "instrumental": True,
            "prompt": "",
            "title": "Synthetic movement",
            "duration": 12,
            "callBackUrl": "https://localhost.invalid/callback",
            "negativeTags": "vocals",
            "style": "Synthetic fixture only.",
        }
        common = {
            "model": "V5_5", "customMode": True, "instrumental": True,
            "prompt": "", "callBackUrl": "https://localhost.invalid/callback",
        }
        cue = {
            "id": "m1-synthetic", "shots": [1, 2],
            "emotionalArc": "fixture",
            "payload": {
                "title": "Synthetic movement", "duration": 12,
                "negativeTags": "vocals", "style": "Synthetic fixture only.",
            },
        }
        self.creative = {"version": 1, "commonPayload": common, "cues": [cue]}
        write_json(self.creative_path, self.creative)
        self.creative_hash = ranker.sha_file(self.creative_path)
        values_a = dict(zip(ranker.SEMANTIC_DIMENSIONS,
                            [45, 10, 60, 15, 50, 35, 20, 70]))
        values_b = dict(zip(ranker.SEMANTIC_DIMENSIONS,
                            [70, 5, 75, 5, 80, 60, 55, 50]))
        self.anchors = {
            "version": 1,
            "kind": "authored-semantic-target-not-audio-measurement",
            "dimensions": list(ranker.SEMANTIC_DIMENSIONS),
            "anchors": [
                {"shotId": "test-s001", "values": values_a},
                {"shotId": "test-s002", "values": values_b},
            ],
        }
        write_json(self.anchors_path, self.anchors)
        self.anchors_hash = ranker.sha_file(self.anchors_path)
        frozen_cue = {**cue, "payload": {**common, **cue["payload"]}}
        self.execution = {
            "version": 1, "episode": "TEST",
            "sourceDraftSha256": self.creative_hash,
            "expectedCandidatesPerRequest": 2,
            "cues": [frozen_cue],
        }
        write_json(self.execution_path, self.execution)
        self.execution_hash = ranker.sha_file(self.execution_path)
        profile = {
            "promptBlend": 0.5,
            "curves": {
                "rmsEnergy": [[0.0, 0.2], [1.0, 0.8]],
                "onsetActivity": [[0.0, 0.1], [1.0, 0.6]],
            },
            "globals": {
                "dynamicRange": 0.5, "tempoConfidence": 0.4,
                "breathing": 0.3, "sectionContrast": 0.5,
            },
            "ending": {"tailDrop": 0.3, "terminalSilence": 0.1,
                       "abruptness": 0.1},
        }
        weights = {name: 1.0 for name in (
            ranker.TRAJECTORY_FEATURES + ranker.GLOBAL_FEATURES
            + ranker.ENDING_FEATURES
        )}
        self.manifest = {
            "version": 1, "episode": "TEST",
            "creativeDraft": "creative.json",
            "creativeDraftSha256": self.creative_hash,
            "semanticAnchors": "anchors.json",
            "semanticAnchorsSha256": self.anchors_hash,
            "executionManifest": "execution.json",
            "candidateDirectory": "sources",
            "reportPath": "analysis/report.json",
            "sampling": {
                "sampleRate": SAMPLE_RATE, "frameSize": 2048, "hopSize": 512,
                "trajectoryBins": 12, "sectionBins": 6, "lowFrequencyHz": 180,
            },
            "scoreWeights": weights,
            "homogeneityThresholds": {
                "globalDistance": 0.14, "trajectorySimilarity": 0.93,
                "diversityScale": 0.42,
            },
            "movementProfiles": {"m1-synthetic": profile},
        }
        write_json(self.manifest_path, self.manifest)

    def add_candidate(self, take: int, audio: np.ndarray) -> Path:
        media = self.root / "sources" / f"m1-synthetic-take-{take}.wav"
        sidecar = self.root / "sources" / f"m1-synthetic-take-{take}.source.json"
        write_wav(media, audio)
        request_hash = hashlib.sha256(ranker.canonical_bytes(self.payload)).hexdigest()
        write_json(sidecar, {
            "version": 1,
            "episode": "TEST",
            "cueId": "m1-synthetic",
            "candidateIndex": take,
            "request": self.payload,
            "requestSha256": request_hash,
            "executionManifestSha256": self.execution_hash,
            "sourceDraftSha256": self.creative_hash,
            "file": str(media.resolve()),
            "sha256": ranker.sha_file(media),
        })
        return media


class BindingAndExecutionTests(unittest.TestCase):
    def test_unversioned_paid_draft_requires_explicit_schema_and_exact_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ManifestFixture(Path(tmp))
            del fixture.creative["version"]
            write_json(fixture.creative_path, fixture.creative)
            digest = ranker.sha_file(fixture.creative_path)
            fixture.manifest["creativeDraftSha256"] = digest
            fixture.execution["sourceDraftSha256"] = digest
            write_json(fixture.execution_path, fixture.execution)
            write_json(fixture.manifest_path, fixture.manifest)
            with self.assertRaises(ranker.AnalysisError):
                ranker.load_plan(fixture.manifest_path)
            fixture.manifest["creativeDraftSchema"] = "finite-suno-unversioned-v1"
            write_json(fixture.manifest_path, fixture.manifest)
            self.assertEqual(ranker.load_plan(fixture.manifest_path)["creativeHash"], digest)
            fixture.execution["sourceDraftSha256"] = "0" * 64
            write_json(fixture.execution_path, fixture.execution)
            with self.assertRaises(ranker.AnalysisError):
                ranker.load_plan(fixture.manifest_path)

    def test_unversioned_opt_in_does_not_accept_unknown_explicit_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ManifestFixture(Path(tmp))
            fixture.creative["version"] = 2
            write_json(fixture.creative_path, fixture.creative)
            fixture.manifest["creativeDraftSha256"] = ranker.sha_file(fixture.creative_path)
            fixture.manifest["creativeDraftSchema"] = "finite-suno-unversioned-v1"
            write_json(fixture.manifest_path, fixture.manifest)
            with self.assertRaises(ranker.AnalysisError):
                ranker.load_plan(fixture.manifest_path)

    def test_hard_transition_blocks_semantic_interpolation_bleed(self) -> None:
        left_values = dict(zip(ranker.SEMANTIC_DIMENSIONS,
                               [10, 10, 10, 10, 10, 10, 90, 10]))
        right_values = dict(zip(ranker.SEMANTIC_DIMENSIONS,
                                [10, 10, 10, 10, 10, 10, 5, 10]))
        plan = {
            "parsedAnchors": [
                {"scene": 27, "values": left_values},
                {"scene": 37, "values": right_values},
            ],
            "hardTransitions": [31],
        }
        values = ranker.interpolate_with_hard_barriers(
            plan, "kinetic", np.asarray([30.9, 31.0])
        )
        self.assertAlmostEqual(float(values[0]), 0.90)
        self.assertAlmostEqual(float(values[1]), 0.05)

    def test_missing_candidate_dry_run_is_clean_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            result = ranker.execute(fixture.manifest_path)
            self.assertEqual(result["status"], "MISSING_CANDIDATES")
            self.assertEqual(result["networkCalls"], 0)
            self.assertEqual(result["writes"], 0)
            self.assertEqual(result["missingMovements"], ["m1-synthetic"])
            self.assertFalse((fixture.root / "sources").exists())
            self.assertFalse((fixture.root / "analysis" / "report.json").exists())

    def test_changed_authored_wave_is_refused_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            fixture.anchors["anchors"][0]["values"]["awe"] = 99
            write_json(fixture.anchors_path, fixture.anchors)
            with self.assertRaisesRegex(ranker.AnalysisError, "changed after"):
                ranker.execute(fixture.manifest_path)

    def test_report_path_may_not_enter_deliverables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            fixture.manifest["reportPath"] = "deliverables/report.json"
            write_json(fixture.manifest_path, fixture.manifest)
            with self.assertRaisesRegex(ranker.AnalysisError, "never write below deliverables"):
                ranker.execute(fixture.manifest_path)

    def test_bound_synthetic_candidates_are_analyzed_and_flagged_advisory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            audio = pulse_track(seconds=12.0)
            fixture.add_candidate(1, audio)
            fixture.add_candidate(2, audio.copy())
            # Small synthetic feature tests never decode production media on the Mac.
            with patch.object(ranker, "decode_audio", return_value=audio):
                result = ranker.execute(fixture.manifest_path, do_analyze=True)
            self.assertEqual(result["networkCalls"], 0)
            self.assertEqual(result["candidatesAnalyzed"], 2)
            self.assertTrue(result["advisoryOnly"])
            report_path = fixture.root / "analysis" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("No waveform feature proves an emotion",
                          report["advisory"]["limitations"][0])
            self.assertEqual(
                len(report["rankings"]["m1-synthetic"]["orderedCandidates"]), 2
            )
            self.assertIsNone(report["rankings"]["m1-synthetic"]["structuralLead"])
            self.assertGreaterEqual(len(report["homogeneityFlags"]), 1)
            self.assertTrue(all(candidate["selectionVerdict"].startswith("unreviewed")
                                for candidate in report["candidates"]))

    def test_media_changed_after_sidecar_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            media = fixture.add_candidate(1, tone(330.0))
            with media.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ranker.AnalysisError, "media hash mismatch"):
                ranker.execute(fixture.manifest_path)


def author_dimensions(fixture: ManifestFixture, names: list[str], **manifest_extra) -> None:
    """Rewrite the fixture's anchors with the operator's own axes and rebind the hash."""
    fixture.anchors["dimensions"] = names
    for number, row in enumerate(fixture.anchors["anchors"]):
        row["values"] = {name: 20 + 30 * number + 5 * position
                         for position, name in enumerate(names)}
    write_json(fixture.anchors_path, fixture.anchors)
    fixture.manifest["semanticAnchorsSha256"] = ranker.sha_file(fixture.anchors_path)
    fixture.manifest.update(manifest_extra)
    write_json(fixture.manifest_path, fixture.manifest)


def full_curves(profile: dict) -> dict:
    return {**profile, "promptBlend": 1,
            "curves": {name: [[0.0, 0.3], [1.0, 0.6]] for name in ranker.TRAJECTORY_FEATURES}}


class OpenDimensionTests(unittest.TestCase):
    def test_default_table_is_the_default_reading(self) -> None:
        rng = np.random.default_rng(7)
        semantics = {name: rng.uniform(0, 1, 32) for name in ranker.SEMANTIC_DIMENSIONS}
        tabled = ranker.table_proxies(ranker.DEFAULT_PROXY_WEIGHTS, semantics, 32)
        for name, values in ranker.default_proxies(semantics).items():
            np.testing.assert_allclose(tabled[name], values, rtol=0, atol=1e-12)

    def test_default_eight_record_their_mapping_in_a_version_two_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            audio = pulse_track(seconds=12.0)
            fixture.add_candidate(1, audio)
            fixture.add_candidate(2, audio.copy())
            with patch.object(ranker, "decode_audio", return_value=audio):
                ranker.execute(fixture.manifest_path, do_analyze=True)
            report = json.loads((fixture.root / "analysis/report.json").read_text())
            self.assertEqual(report["version"], 2)
            self.assertEqual(report["expectedCandidatesPerMovement"], 2)
            mapping = report["dimensionMapping"]
            self.assertEqual(mapping["source"], "default-eight-axis")
            self.assertEqual(mapping["sha256"], ranker.sha_bytes(
                ranker.canonical_bytes(ranker.DEFAULT_PROXY_WEIGHTS)))
            self.assertEqual(set(mapping["defaulted"]), set(ranker.SEMANTIC_DIMENSIONS))
            self.assertEqual(mapping["ignored"], [])

    def test_other_axes_need_authored_weights_or_the_movements_own_curves(self) -> None:
        names = ["demo-tension", "demo-distance"]
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            author_dimensions(fixture, names)
            with self.assertRaisesRegex(ranker.AnalysisError, "proxyWeights.*promptBlend 1"):
                ranker.execute(fixture.manifest_path)
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            profile = full_curves(fixture.manifest["movementProfiles"]["m1-synthetic"])
            author_dimensions(fixture, names, movementProfiles={"m1-synthetic": profile})
            plan = ranker.load_plan(fixture.manifest_path)
            self.assertEqual(plan["dimensionMapping"]["source"], "movement-curves-only")
            self.assertEqual(plan["dimensionMapping"]["ignored"], names)
            target = ranker.semantic_target(plan, plan["cues"][0])
            np.testing.assert_allclose(target["proxies"]["lowFrequency"], np.linspace(0.3, 0.6, 12))

    def test_authored_weights_drive_the_targets_and_are_recorded(self) -> None:
        names = ["demo-tension", "demo-distance", "demo-unused"]
        weights = {feature: {"intercept": 0.1, "weights": {"demo-tension": 0.5,
                                                          "demo-distance": -0.2}}
                   for feature in ranker.TRAJECTORY_FEATURES}
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            author_dimensions(fixture, names, proxyWeights=weights)
            plan = ranker.load_plan(fixture.manifest_path)
            mapping = plan["dimensionMapping"]
            self.assertEqual((mapping["source"], mapping["mapped"], mapping["ignored"]),
                             ("authored-proxyWeights", ["demo-distance", "demo-tension"],
                              ["demo-unused"]))
            target = ranker.semantic_target(plan, plan["cues"][0])
            semantics = target["semantics"]
            expected = np.clip(0.1 + 0.5 * semantics["demo-tension"]
                               - 0.2 * semantics["demo-distance"], 0, 1)
            np.testing.assert_allclose(target["proxies"]["lowFrequency"], expected)
            for broken in ({**weights, "lowFrequency": {"intercept": 0, "weights": {"demo-other": 1}}},
                           {k: v for k, v in weights.items() if k != "onsetActivity"},
                           {**weights, "rmsEnergy": {"intercept": 0, "weights": {}, "clip": [0.5, 0.2]}}):
                with self.subTest(broken=sorted(broken)):
                    author_dimensions(fixture, names, proxyWeights=broken)
                    with self.assertRaises(ranker.AnalysisError):
                        ranker.load_plan(fixture.manifest_path)

    def test_axis_names_may_be_any_non_blank_text(self) -> None:
        names = ["demo-tensión", "demo held breath", "демо-даль"]
        weights = {feature: {"intercept": 0.2, "weights": {names[0]: 0.4, names[1]: -0.1}}
                   for feature in ranker.TRAJECTORY_FEATURES}
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            author_dimensions(fixture, names, proxyWeights=weights)
            plan = ranker.load_plan(fixture.manifest_path)
            self.assertEqual(list(plan["dimensions"]), names)
            self.assertEqual(plan["dimensionMapping"]["ignored"], [names[2]])
            for broken in ([*names, names[0]], [names[0], "  "], [names[0], 3]):
                with self.subTest(broken=broken):
                    author_dimensions(fixture, broken, proxyWeights=weights)
                    with self.assertRaisesRegex(ranker.AnalysisError, "non-empty list of unique"):
                        ranker.load_plan(fixture.manifest_path)

    def test_landmarks_place_the_loudest_moment_and_the_tail(self) -> None:
        audio = tone(330.0, seconds=12.0, amplitude=0.05)
        burst = slice(round(6.0 * SAMPLE_RATE), round(6.5 * SAMPLE_RATE))
        audio[burst] = tone(440.0, seconds=0.5, amplitude=0.8)
        landmarks = features(audio)["landmarks"]
        self.assertAlmostEqual(landmarks["peakRmsSeconds"], 6.25, delta=0.3)
        self.assertAlmostEqual(landmarks["tailStartSeconds"], 12.0 - 3.0, delta=0.01)
        self.assertIn("not verified musical phrases", landmarks["basis"])
        onsets = features(pulse_track(seconds=12.0))["landmarks"]["strongestOnsets"]
        times = [onset["seconds"] for onset in onsets]
        self.assertTrue(0 < len(times) <= 5 and times == sorted(times))
        pulses = np.arange(0.25, 12.0, 0.5)
        self.assertTrue(all(np.min(np.abs(pulses - t)) < 0.1 for t in times))

    def test_an_extend_without_a_requested_duration_is_not_flagged_short(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            fixture.creative["cues"][0]["payload"].pop("duration")
            write_json(fixture.creative_path, fixture.creative)
            fixture.creative_hash = ranker.sha_file(fixture.creative_path)
            fixture.payload.pop("duration")
            fixture.execution["sourceDraftSha256"] = fixture.creative_hash
            fixture.execution["cues"][0]["payload"].pop("duration")
            write_json(fixture.execution_path, fixture.execution)
            fixture.execution_hash = ranker.sha_file(fixture.execution_path)
            fixture.manifest["creativeDraftSha256"] = fixture.creative_hash
            write_json(fixture.manifest_path, fixture.manifest)
            audio = pulse_track(seconds=12.0)
            fixture.add_candidate(1, audio)
            with patch.object(ranker, "decode_audio", return_value=audio):
                ranker.execute(fixture.manifest_path, do_analyze=True)
            report = json.loads((fixture.root / "analysis/report.json").read_text())
            candidate = report["candidates"][0]
            self.assertEqual((candidate["requestedDurationSeconds"], candidate["durationFlag"]),
                             (None, "NO_REQUESTED_DURATION"))


class ScoreReviewPageTests(unittest.TestCase):
    def test_review_page_reads_the_take_count_from_the_report(self) -> None:
        try:
            from pipeline import build_score_candidate_review as review
        except ImportError:
            import build_score_candidate_review as review
        for takes in (1, 3):
            with self.subTest(takes=takes), tempfile.TemporaryDirectory() as directory:
                fixture = ManifestFixture(Path(directory))
                fixture.execution["expectedCandidatesPerRequest"] = takes
                write_json(fixture.execution_path, fixture.execution)
                fixture.execution_hash = ranker.sha_file(fixture.execution_path)
                audio = pulse_track(seconds=12.0)
                for take in range(1, takes + 1):
                    fixture.add_candidate(take, audio.copy())
                with patch.object(ranker, "decode_audio", return_value=audio):
                    ranker.execute(fixture.manifest_path, do_analyze=True)
                page = fixture.root / "review/score.html"
                with contextlib.redirect_stdout(io.StringIO()):
                    review.build(fixture.root / "analysis/report.json", fixture.creative_path, page)
                self.assertEqual(page.read_text().count('"sha256"'), takes)


if __name__ == "__main__":
    unittest.main()
