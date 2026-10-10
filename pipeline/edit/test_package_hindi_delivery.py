from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from pipeline.edit import package_hindi_delivery as delivery


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def media_expected(width: int, height: int, target: float) -> dict:
    return {
        "width": width,
        "height": height,
        "fps": "24/1",
        "frames": 240,
        "durationSeconds": 10.0,
        "videoCodec": "h264",
        "pixelFormat": "yuv420p",
        "audioCodec": "aac",
        "audioChannels": 2,
        "audioSampleRate": 48000,
        "maxAvDeltaSeconds": 0.15,
        "loudness": {
            "targetIntegratedLufs": target,
            "toleranceLu": 0.5,
            "maxTruePeakDbtp": -1.0,
        },
    }


class HindiDeliveryPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.episode = Path(self.temporary.name) / "demo-test-film"
        (self.episode / "work/inputs").mkdir(parents=True)
        (self.episode / "story").mkdir()
        (self.episode / "story/VERDICTS.md").write_text("approved fixtures\n", encoding="utf-8")
        files = {
            "long": b"long-video-source",
            "short": b"short-youtube-source",
            "social": b"short-social-source",
            "hi": "1\n00:00:00,000 --> 00:00:00,800\nयह एक परीक्षण है।\n".encode(),
            "en": b"1\n00:00:00,000 --> 00:00:00,800\nThis is a test.\n",
            "short_hi": "1\n00:00:00,000 --> 00:00:00,800\nयह छोटा अंश है।\n".encode(),
            "short_en": b"1\n00:00:00,000 --> 00:00:00,800\nThis is a short excerpt.\n",
            "thumb_base": b"approved-textless-thumbnail-source",
            "thumb": b"approved-thumbnail-source",
        }
        self.sources = {}
        for key, content in files.items():
            path = self.episode / "work/inputs" / key
            path.write_bytes(content)
            self.sources[key] = path
        self.thumbnail_recipe = self.episode / "work/inputs/thumbnail-title-recipe.json"
        self.thumbnail_recipe.write_text(json.dumps({
            "method": "GPU-host FreeType exact Romanized text,2x then Lanczos downsample",
            "variants": [{"id": "C2", "sourceSha256": digest(self.sources["thumb_base"]),
                          "files": {"thumb": digest(self.sources["thumb"])},
                          "lines": [{"text": "TEST TITLE"}]}],
        }), encoding="utf-8")
        self.config = {
            "schemaVersion": 1,
            "layout": "hindi-finals-only-v1",
            "episode": self.episode.name,
            "stagingDir": "work/final-delivery-v1/staged-deliverables-v1",
            "publishingAuthorized": False,
            "publishAt": None,
            "channel": {"handle": "@example-studio", "id": "example-channel-id", "language": "hi", "name": "Example Studio"},
            "approvals": {},
            "assets": {
                "long-hi": {"source": "work/inputs/long", "sha256": digest(self.sources["long"]), "destination": "DEMO_TEST_HI_1080P.mp4", "expected": media_expected(1920, 1080, -14.0)},
                "subtitles-hi": {"source": "work/inputs/hi", "sha256": digest(self.sources["hi"]), "destination": "demo-test-hi.srt", "authored": True},
                "subtitles-en": {"source": "work/inputs/en", "sha256": digest(self.sources["en"]), "destination": "demo-test-en.srt", "authored": True},
                "short-hi": {"source": "work/inputs/short", "sha256": digest(self.sources["short"]), "destination": "shorts/demo_test_short_hi.mp4", "expected": media_expected(1080, 1920, -14.0)},
                "short-hi-social": {"source": "work/inputs/social", "sha256": digest(self.sources["social"]), "destination": "shorts/demo_test_short_hi_social.mp4", "expected": media_expected(1080, 1920, -10.0)},
                "short-subtitles-hi": {"source": "work/inputs/short_hi", "sha256": digest(self.sources["short_hi"]), "destination": "shorts/demo-short-hi.srt", "authored": True},
                "short-subtitles-en": {"source": "work/inputs/short_en", "sha256": digest(self.sources["short_en"]), "destination": "shorts/demo-short-en.srt", "authored": True},
            },
            "thumbnail": {
                "method": "gpu-typography",
                "textlessSource": "work/inputs/thumb_base",
                "textlessSha256": digest(self.sources["thumb_base"]),
                "titleRecipe": "work/inputs/thumbnail-title-recipe.json",
                "titleRecipeSha256": digest(self.thumbnail_recipe),
                "source": "work/inputs/thumb",
                "sha256": digest(self.sources["thumb"]),
                "choice": "C2",
                "jpegMaxBytes": 2_000_000,
            },
            "metadata": {
                "uploadSettings": {"videoLanguage": "Hindi", "aiUse": "Yes", "visibility": "Private",
                                   "madeForKids": "Review before upload", "category": "Film & Animation"},
                "title": "Ek Nayi Kahani",
                "description": "एक पूरी हिंदी कथा।",
                "tags": ["story", "कथा"],
                "hashtags": ["#Story", "#Animation"],
                "pinnedComment": None,
                "chapters": [{"seconds": 0, "title": "आरंभ"}],
                "shortTitle": "Kahani Ki Ek Jhalak",
                "shortDescription": "नई कहानी का पूरा रूप देखें।",
                "shortTags": ["story", "animation"],
            },
            "short": {"humanReview": "approved", "revision": "test-v1", "relatedVideoId": None, "ctaText": "FULL VIDEO ON THE CHANNEL", "ctaSeconds": 4.0},
        }
        approvals_dir = self.episode / "story/approvals"
        approvals_dir.mkdir()
        for approval_id in delivery.APPROVAL_IDS:
            subjects = delivery._approval_subject_hashes(self.config, approval_id)
            evidence_doc = {
                "schemaVersion": 1,
                "approvalId": approval_id,
                "verdict": "approved",
                "approvedAt": "2026-09-05T00:00:00Z",
                "subjectSha256": subjects,
            }
            evidence = approvals_dir / f"{approval_id}.json"
            evidence.write_text(json.dumps(evidence_doc), encoding="utf-8")
            self.config["approvals"][approval_id] = {
                "status": "approved",
                "evidence": evidence.relative_to(self.episode).as_posix(),
                "evidenceSha256": digest(evidence),
                "subjectSha256": subjects,
                "approvedAt": "2026-09-05T00:00:00Z",
            }
        self.config_path = self.episode / "work/final-delivery-v1/config.json"
        self.config_path.parent.mkdir(parents=True)
        self.write_config()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_operator_notes_bind_approval_and_are_not_public_description(self):
        before = delivery._approval_subject_hashes(self.config, 'longForm')
        self.config['metadata']['operatorNotes'] = 'Quiz at 15:57; do not publish without authority.'
        self.assertNotEqual(before, delivery._approval_subject_hashes(self.config, 'longForm'))
        page = delivery.build_upload_kit(self.config, self.episode)
        self.assertIn('operator-notes', page)
        self.assertIn(self.config['metadata']['operatorNotes'], page)
        self.assertNotIn('Quiz at', delivery.description_with_chapters(self.config['metadata']))
        delivery._validate_config_schema(self.config)

    def write_config(self) -> None:
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    @staticmethod
    def fake_media(path: Path, asset_id: str, expected: dict) -> dict:
        payload = "same-short-picture" if asset_id.startswith("short-hi") else "long-picture"
        return {
            "width": expected["width"],
            "height": expected["height"],
            "fps": expected["fps"],
            "frames": expected["frames"],
            "videoDurationSeconds": expected["durationSeconds"],
            "audioDurationSeconds": expected["durationSeconds"],
            "formatDurationSeconds": expected["durationSeconds"],
            "videoStartSeconds": 0.0,
            "audioStartSeconds": 0.0,
            "avStartDeltaSeconds": 0.0,
            "avEndDeltaSeconds": 0.0,
            "frameCadenceMaxErrorSeconds": 0.0,
            "videoTimelineSha256": hashlib.sha256(b"same-timeline").hexdigest(),
            "audioPacketCount": 469,
            "audioPacketContinuityMaxErrorSeconds": 0.0,
            "audioPacketTimelineSha256": hashlib.sha256(b"same-audio-timeline").hexdigest(),
            "videoCodec": expected["videoCodec"],
            "pixelFormat": expected["pixelFormat"],
            "audioCodec": expected["audioCodec"],
            "audioChannels": expected["audioChannels"],
            "audioSampleRate": expected["audioSampleRate"],
            "fullDecode": "passed",
            "videoPayloadSha256": hashlib.sha256(payload.encode()).hexdigest(),
            "loudness": {
                "integratedLufs": expected["loudness"]["targetIntegratedLufs"],
                "truePeakDbtp": -1.1,
                "loudnessRangeLu": 4.0,
                "thresholdLufs": -24.0,
            },
        }

    @staticmethod
    def fake_thumbnail(source: Path, png: Path, jpg: Path, jpeg_max_bytes: int) -> dict:
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"lossless-thumbnail")
        jpg.write_bytes(b"same-thumbnail-jpeg")
        return {
            "source": {"width": 1664, "height": 936, "pixelFormat": "rgb24", "fullDecode": "passed"},
            "png": {"width": 1280, "height": 720, "pixelFormat": "rgb24", "fullDecode": "passed", "sha256": digest(png), "bytes": png.stat().st_size},
            "jpeg": {"width": 1280, "height": 720, "pixelFormat": "yuvj420p", "fullDecode": "passed", "sha256": digest(jpg), "bytes": jpg.stat().st_size, "ffmpegQscale": 2, "derivedFromPngSha256": digest(png)},
            "sameApprovedComposition": True,
        }

    @staticmethod
    def fake_probe(path: Path) -> dict:
        if path.name == "thumb":
            return {"width": 1664, "height": 936, "pixelFormat": "rgb24", "fullDecode": "passed"}
        pixel_format = "yuvj420p" if path.suffix == ".jpg" else "rgb24"
        return {"width": 1280, "height": 720, "pixelFormat": pixel_format, "fullDecode": "passed"}

    def test_readiness_passes_complete_nonpublishing_contract(self) -> None:
        _config, gates = delivery.analyze_config(self.episode, self.config_path)
        self.assertEqual([], gates)

    def set_delegated_review(self) -> None:
        self.config["short"]["humanReview"] = "delegated-studio-approved"
        self.config["short"]["delegatedAuthority"] = {
            "path": "story/VERDICTS.md",
            "sha256": digest(self.episode / "story/VERDICTS.md"),
        }
        approval = self.config["approvals"]["short"]
        subjects = delivery._approval_subject_hashes(self.config, "short")
        evidence = self.episode / approval["evidence"]
        document = json.loads(evidence.read_text())
        document["subjectSha256"] = subjects
        evidence.write_text(json.dumps(document), encoding="utf-8")
        approval["subjectSha256"] = subjects
        approval["evidenceSha256"] = digest(evidence)
        self.write_config()

    def test_delegated_review_requires_unchanged_recorded_authority(self) -> None:
        self.set_delegated_review()
        _config, gates = delivery.analyze_config(self.episode, self.config_path)
        self.assertEqual([], gates)
        (self.episode / "story/VERDICTS.md").write_text("authority changed\n")
        with self.assertRaisesRegex(delivery.DeliveryError, "authority binding changed"):
            delivery.analyze_config(self.episode, self.config_path)

    def test_delegated_review_cannot_omit_authority(self) -> None:
        self.set_delegated_review()
        del self.config["short"]["delegatedAuthority"]
        self.write_config()
        with self.assertRaises(delivery.DeliveryError):
            delivery.analyze_config(self.episode, self.config_path)

    @mock.patch.object(delivery, "_probe_image", new=fake_probe.__func__)
    @mock.patch.object(delivery, "render_thumbnail", new=fake_thumbnail.__func__)
    @mock.patch.object(delivery, "validate_media_file", new=fake_media.__func__)
    def test_delegated_review_survives_packaging_without_a_human_verdict(self) -> None:
        self.set_delegated_review()
        result = delivery.stage_package(self.episode, self.config_path)
        self.assertEqual("staged-and-verified", result["status"])
        manifest = json.loads((self.episode / self.config["stagingDir"] / "FINAL_MANIFEST.json").read_text())
        self.assertEqual("delegated-studio-approved", manifest["short"]["humanReview"])

    def test_publication_authority_is_always_refused(self) -> None:
        self.config["publishingAuthorized"] = True
        self.write_config()
        with self.assertRaises(delivery.DeliveryError):
            delivery.analyze_config(self.episode, self.config_path)

    def test_extra_asset_and_review_named_final_are_refused(self) -> None:
        self.config["assets"]["review-page"] = {}
        self.write_config()
        with self.assertRaisesRegex(delivery.DeliveryError, "unsupported keys"):
            delivery.analyze_config(self.episode, self.config_path)
        del self.config["assets"]["review-page"]
        self.config["assets"]["long-hi"]["destination"] = "DEMO_REVIEW_HI_1080P.mp4"
        self.write_config()
        with self.assertRaisesRegex(delivery.DeliveryError, "Review/variant naming"):
            delivery.analyze_config(self.episode, self.config_path)

    def test_subtitle_parser_requires_authored_language_and_shared_timing(self) -> None:
        hi = delivery.parse_srt(self.sources["hi"], language="hi", media_duration=1.0)
        en = delivery.parse_srt(self.sources["en"], language="en", media_duration=1.0)
        self.assertEqual(hi["timings"], en["timings"])
        with self.assertRaisesRegex(delivery.DeliveryError, "no Devanagari"):
            delivery.parse_srt(self.sources["en"], language="hi", media_duration=1.0)

    def test_schema_hindi_names_and_hashed_approval_evidence_are_enforced(self) -> None:
        self.config["thumbnail"]["choice"] = 42
        self.write_config()
        with self.assertRaises(delivery.DeliveryError):
            delivery.analyze_config(self.episode, self.config_path)
        self.config["thumbnail"]["choice"] = "C2"
        self.config["assets"]["short-hi"]["destination"] = "shorts/english_cut.mp4"
        self.write_config()
        with self.assertRaises(delivery.DeliveryError):
            delivery.analyze_config(self.episode, self.config_path)
        self.config["assets"]["short-hi"]["destination"] = "shorts/demo_test_short_hi.mp4"
        evidence = self.episode / self.config["approvals"]["longForm"]["evidence"]
        evidence.write_text("{}", encoding="utf-8")
        self.write_config()
        with self.assertRaisesRegex(delivery.DeliveryError, "evidence SHA-256 mismatch"):
            delivery.analyze_config(self.episode, self.config_path)

    def test_approvals_bind_long_metadata_and_short_revision(self) -> None:
        original_description = self.config["metadata"]["description"]
        self.config["metadata"]["description"] = "नई पर अस्वीकृत विवरण प्रति।"
        self.write_config()
        with self.assertRaisesRegex(delivery.DeliveryError, "longForm.subjectSha256"):
            delivery.analyze_config(self.episode, self.config_path)
        self.config["metadata"]["description"] = original_description
        self.config["short"]["revision"] = "unapproved-revision"
        self.write_config()
        with self.assertRaisesRegex(delivery.DeliveryError, "short.subjectSha256"):
            delivery.analyze_config(self.episode, self.config_path)

    def test_decoded_frame_timing_rejects_vfr(self) -> None:
        good = mock.Mock(stdout="0.000000\n0.041667\n0.083333\n")
        with mock.patch.object(delivery, "_run", return_value=good):
            report = delivery._frame_timing_report(Path("good.mp4"), 3, Fraction(24, 1))
        self.assertLess(report["maxCadenceErrorSeconds"], 0.00075)
        bad = mock.Mock(stdout="0.000000\n0.020000\n0.083333\n")
        with mock.patch.object(delivery, "_run", return_value=bad):
            with self.assertRaisesRegex(delivery.DeliveryError, "not CFR"):
                delivery._frame_timing_report(Path("vfr.mp4"), 3, Fraction(24, 1))

    def test_frame_timing_bounds_decoder_threads_without_skipping_frames(self) -> None:
        probe = mock.Mock(stdout="0.000000\n0.041667\n0.083333\n")
        with mock.patch.object(delivery, "_run", return_value=probe) as run:
            report = delivery._frame_timing_report(Path("same-frames.mp4"), 3, Fraction(24, 1))
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-threads") + 1], "8")
        self.assertIn("frame=best_effort_timestamp_time", command)
        self.assertNotIn("-skip_frame", command)
        self.assertEqual(report["lastSeconds"], 0.083333)

    def test_aac_packet_timing_requires_monotonic_continuity(self) -> None:
        def packets(rows):
            return mock.Mock(stdout=json.dumps({"packets": [dict(pts_time=a, duration_time=b) for a,b in rows]}))
        good = packets([("-0.021333","0.021333"),("0.000000","0.021333"),("0.021333","0.021334")])
        with mock.patch.object(delivery, "_run", return_value=good):
            report = delivery._audio_packet_timing_report(Path("good.mp4"))
        self.assertEqual(3, report["packetCount"])
        discontinuous = packets([("0.000000","0.021333"),("0.050000","0.021333")])
        with mock.patch.object(delivery, "_run", return_value=discontinuous):
            with self.assertRaisesRegex(delivery.DeliveryError, "gap/overlap"):
                delivery._audio_packet_timing_report(Path("gap.mp4"))
        nonmonotonic = packets([("0.000000","0.021333"),("0.000000","0.021333")])
        with mock.patch.object(delivery, "_run", return_value=nonmonotonic):
            with self.assertRaisesRegex(delivery.DeliveryError, "not strictly monotonic"):
                delivery._audio_packet_timing_report(Path("repeat.mp4"))

    def test_aac_skip_samples_side_data_does_not_corrupt_duration(self):
        data={"packets":[{"pts_time":"-0.021333", "duration_time":"0.021333",
                          "side_data_list":[{"side_data_type":"Skip Samples", "skip_samples":1024}]},
                         {"pts_time":"0.000000", "duration_time":"0.021333"}]}
        with mock.patch.object(delivery, "_run", return_value=mock.Mock(stdout=json.dumps(data))) as run:
            report=delivery._audio_packet_timing_report(Path("primed.mp4"))
        self.assertEqual(report["packetCount"],2)
        self.assertEqual(report["maxContinuityErrorSeconds"],0)
        self.assertIn("json", run.call_args.args[0])
        del data["packets"][0]["duration_time"]
        with mock.patch.object(delivery, "_run", return_value=mock.Mock(stdout=json.dumps(data))):
            with self.assertRaisesRegex(delivery.DeliveryError,"without numeric"):
                delivery._audio_packet_timing_report(Path("missing.mp4"))

    def test_same_process_measurement_cache_requires_exact_bytes_and_contract(self):
        delivery._MEDIA_VALIDATION_CACHE.clear()
        source=self.sources["long"]
        expected=media_expected(1920,1080,-14)
        with mock.patch.object(delivery,"_validate_media_file_uncached",return_value={"fullDecode":"passed"}) as check:
            delivery.validate_media_file(source,"long-hi",expected)
            delivery.validate_media_file(source,"long-hi",expected)
            self.assertEqual(check.call_count,1)
            source.write_bytes(b"changed-source")
            delivery.validate_media_file(source,"long-hi",expected)
            self.assertEqual(check.call_count,2)
            expected["frames"]+=1
            delivery.validate_media_file(source,"long-hi",expected)
            self.assertEqual(check.call_count,3)
        delivery._MEDIA_VALIDATION_CACHE.clear()

    def test_channel_and_upload_settings_are_explicit_and_bound(self) -> None:
        before = delivery._approval_subject_hashes(self.config, "longForm")
        page = delivery.build_upload_kit(self.config, self.episode)
        self.assertIn("@example-studio", page)
        self.assertIn("AI-use declaration</dt><dd>Yes", page)
        self.config["metadata"]["uploadSettings"]["aiUse"] = "No"
        self.assertNotEqual(before, delivery._approval_subject_hashes(self.config, "longForm"))
        self.write_config()
        with self.assertRaisesRegex(delivery.DeliveryError, "longForm.subjectSha256"):
            delivery.analyze_config(self.episode, self.config_path)
        del self.config["metadata"]["uploadSettings"]
        self.write_config()
        with self.assertRaisesRegex(delivery.DeliveryError, "missing required keys"):
            delivery.analyze_config(self.episode, self.config_path)

    @mock.patch.object(delivery, "_probe_image", new=fake_probe.__func__)
    @mock.patch.object(delivery, "render_thumbnail", new=fake_thumbnail.__func__)
    @mock.patch.object(delivery, "validate_media_file", new=fake_media.__func__)
    def test_stage_is_exact_and_clutter_is_rejected(self) -> None:
        source_hashes = {key: digest(path) for key, path in self.sources.items()}
        result = delivery.stage_package(self.episode, self.config_path)
        self.assertEqual("staged-and-verified", result["status"])
        self.assertEqual(11, result["files"])
        stage = self.episode / self.config["stagingDir"]
        self.assertFalse((self.episode / "deliverables").exists())
        self.assertEqual(source_hashes, {key: digest(path) for key, path in self.sources.items()})
        manifest = json.loads((stage / "FINAL_MANIFEST.json").read_text())
        self.assertFalse(manifest["publishingAuthorized"])
        self.assertEqual(set(delivery.ALL_ASSET_IDS), {row["id"] for row in manifest["assets"]})
        page = (stage / "upload_kit.html").read_text()
        for ident in delivery.REQUIRED_KIT_IDS:
            self.assertIn(f'id="{ident}"', page)
        for setting in ("Video language</dt><dd>Hindi", "AI-use declaration</dt><dd>Yes", "Made for kids</dt><dd>Review before upload", "Film &amp; Animation"):
            self.assertIn(setting, page)
        self.assertIn("does not grant permission to upload", page)
        (stage / "DEMO_REVIEW.html").write_text("clutter")
        with self.assertRaisesRegex(delivery.DeliveryError, "Final whitelist mismatch"):
            delivery.validate_package_structure(self.config, stage)

    @mock.patch.object(delivery, "render_thumbnail", new=fake_thumbnail.__func__)
    @mock.patch.object(delivery, "validate_media_file", new=fake_media.__func__)
    def test_stage_copy_must_match_configured_source_hash(self) -> None:
        real_copy = delivery.shutil.copy2

        def corrupt_copy(source: Path, destination: Path):
            result = real_copy(source, destination)
            if Path(source).name == "long":
                Path(destination).write_bytes(Path(destination).read_bytes() + b"corrupt")
            return result

        with mock.patch.object(delivery.shutil, "copy2", side_effect=corrupt_copy):
            with self.assertRaisesRegex(delivery.DeliveryError, "configured source SHA-256"):
                delivery.stage_package(self.episode, self.config_path)
        self.assertFalse((self.episode / "deliverables").exists())

    @mock.patch.object(delivery, "_probe_image", new=fake_probe.__func__)
    @mock.patch.object(delivery, "render_thumbnail", new=fake_thumbnail.__func__)
    @mock.patch.object(delivery, "validate_media_file", new=fake_media.__func__)
    def test_upload_kit_copy_cannot_be_stale_even_with_refreshed_manifest_hash(self) -> None:
        delivery.stage_package(self.episode, self.config_path)
        stage = self.episode / self.config["stagingDir"]
        kit = stage / "upload_kit.html"
        kit.write_text(kit.read_text().replace(self.config["metadata"]["title"], "STALE TITLE"), encoding="utf-8")
        manifest_path = stage / "FINAL_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        row = next(row for row in manifest["assets"] if row["id"] == "upload-kit")
        row["sha256"], row["bytes"] = digest(kit), kit.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(delivery.DeliveryError, "byte-for-byte reproducible"):
            delivery.validate_package_structure(self.config, stage)

    @mock.patch.object(delivery, "_probe_image", new=fake_probe.__func__)
    @mock.patch.object(delivery, "render_thumbnail", new=fake_thumbnail.__func__)
    @mock.patch.object(delivery, "validate_media_file", new=fake_media.__func__)
    def test_thumbnail_must_reproduce_from_approved_source(self) -> None:
        delivery.stage_package(self.episode, self.config_path)
        stage = self.episode / self.config["stagingDir"]
        png = stage / delivery.THUMBNAIL_OUTPUTS["thumbnail-master"]
        png.write_bytes(b"different-but-decodable-thumbnail")
        manifest_path = stage / "FINAL_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        asset = next(row for row in manifest["assets"] if row["id"] == "thumbnail-master")
        asset["sha256"], asset["bytes"] = digest(png), png.stat().st_size
        recorded = manifest["validation"]["thumbnail"]["png"]
        recorded["sha256"], recorded["bytes"] = digest(png), png.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(delivery.DeliveryError, "not reproducible"):
            delivery.validate_staged_package(self.episode, self.config_path)

    @mock.patch.object(delivery, "_probe_image", new=fake_probe.__func__)
    @mock.patch.object(delivery, "render_thumbnail", new=fake_thumbnail.__func__)
    @mock.patch.object(delivery, "validate_media_file", new=fake_media.__func__)
    def test_promote_is_explicit_idempotent_and_never_overwrites(self) -> None:
        delivery.stage_package(self.episode, self.config_path)
        with mock.patch.object(delivery, "_config_snapshot", wraps=delivery._config_snapshot) as snapshots:
            with mock.patch.object(delivery, "_validate_package_root", wraps=delivery._validate_package_root) as validations:
                promoted = delivery.promote_to_deliverables(self.episode, self.config_path)
        self.assertEqual(1, snapshots.call_count)
        self.assertEqual(2, validations.call_count)
        self.assertEqual("promoted-local-finals", promoted["status"])
        stage = self.episode / self.config["stagingDir"]
        finals = self.episode / "deliverables"
        self.assertEqual(delivery._tree_hashes(stage), delivery._tree_hashes(finals))
        again = delivery.promote_to_deliverables(self.episode, self.config_path)
        self.assertEqual("already-promoted-identical", again["status"])
        (finals / "upload_kit.html").write_text("different", encoding="utf-8")
        with self.assertRaisesRegex(delivery.DeliveryError, "Refusing overwrite"):
            delivery.promote_to_deliverables(self.episode, self.config_path)
        self.assertTrue((stage / "upload_kit.html").is_file())

    @mock.patch.object(delivery, "_probe_image", new=fake_probe.__func__)
    @mock.patch.object(delivery, "render_thumbnail", new=fake_thumbnail.__func__)
    @mock.patch.object(delivery, "validate_media_file", new=fake_media.__func__)
    def test_stale_config_and_manifest_authority_are_refused(self) -> None:
        delivery.stage_package(self.episode, self.config_path)
        stage = self.episode / self.config["stagingDir"]
        manifest_path = stage / "FINAL_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["short"]["uploadAllowed"] = True
        manifest["publicationAuthority"] = {"authorized": True}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(delivery.DeliveryError):
            delivery.validate_staged_package(self.episode, self.config_path)

        del manifest["publicationAuthority"]
        manifest["short"]["uploadAllowed"] = False
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.config["metadata"]["description"] = "बदला हुआ विवरण।"
        self.write_config()
        with self.assertRaises(delivery.DeliveryError):
            delivery.validate_staged_package(self.episode, self.config_path)

    def test_promotion_lock_and_atomic_noreplace_refuse_collisions(self) -> None:
        with delivery._exclusive_promotion_lock(self.episode):
            with self.assertRaisesRegex(delivery.DeliveryError, "promotion is already active"):
                with delivery._exclusive_promotion_lock(self.episode):
                    pass
        source = self.episode / "work/rename-source"
        destination = self.episode / "work/rename-destination"
        source.mkdir()
        destination.mkdir()
        with self.assertRaisesRegex(delivery.DeliveryError, "refusing overwrite"):
            delivery._rename_noreplace(source, destination)
        self.assertTrue(source.is_dir())
        self.assertTrue(destination.is_dir())


if __name__ == "__main__":
    unittest.main()
