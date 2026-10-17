#!/usr/bin/env python3
"""Offline tests for still registration from any hand, retakes, derived stills,
reference packs and the identity board. Synthetic images only."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

import _common
import asset_registry
import derive_still
import make_reference_pack
import register_builtin_imagegen as registrar

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "project"

# The exact key order an imagegen sidecar has always had. A default
# registration must keep writing exactly this shape.
HISTORICAL_IMAGEGEN_KEYS = [
    "schema_version", "kind", "provider", "tool", "still_id", "episode", "registered_at",
    "registered_by", "source_image", "source_sha256", "target_output_path", "target_sha256",
    "width", "height", "manifest", "record_fingerprint", "source_fingerprint", "prompt_sha256",
    "refs", "imagegen_refs", "exact_identity_claims", "manifest_identity_coverage",
    "continuity_facts", "execution_coverage_count", "no_visible_identity",
]
HISTORICAL_RECORD_KEYS = {
    "status", "attempts", "generated_at", "generation_provider", "provenance_sidecar",
    "grounding_claims",
}


def png(path: Path, size=(64, 36), color=(40, 90, 160), fmt="PNG", **options) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format=fmt, **options)
    return path


class Project(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name).resolve()
        self.root = base / "project"
        shutil.copytree(EXAMPLE, self.root)
        self.imagegen = base / "imagegen-output"
        self.outside = base / "elsewhere"
        self.manifest = self.root / "story/generation_manifest.json"
        environment = mock.patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        for key in ("AGENTIC_PROFILE", "AGENTIC_PROJECT_ROOT", "AGENTIC_IMAGEGEN_OUTPUT_ROOT"):
            os.environ.pop(key, None)

    def run_cli(self, *argv: str) -> object:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            registrar.main(["story/generation_manifest.json", *argv, "--root", str(self.root)])
        return json.loads(buffer.getvalue())

    def imagegen_register(self, still_id: str, *extra: str) -> object:
        source = png(self.imagegen / f"{still_id}.png")
        return self.run_cli(still_id, str(source), "--source-root", str(self.imagegen),
                            "--policy", "story/still-policy.json", *extra)

    def record(self, still_id: str) -> dict:
        doc = json.loads(self.manifest.read_text(encoding="utf-8"))
        return next(row for row in doc["stills"] if row["still_id"] == still_id)

    def sidecar(self, still_id: str) -> dict:
        return json.loads((self.root / self.record(still_id)["provenance_sidecar"]).read_text())


class ImagegenDefaultTests(Project):
    def test_default_registration_keeps_the_historical_shape(self) -> None:
        result = self.imagegen_register("demo-s002", "--no-visible-identity")
        self.assertEqual(result["status"], "generated")
        record = self.record("demo-s002")
        self.assertEqual(record["generation_provider"], "openai_builtin_imagegen")
        self.assertTrue(record["provenance_sidecar"].endswith(".imagegen.json"))
        self.assertNotIn("still_origin", record)
        self.assertEqual(set(record) - {"still_id", "prompt", "size", "target_output_path", "refs"},
                         HISTORICAL_RECORD_KEYS)
        sidecar = self.sidecar("demo-s002")
        self.assertEqual(list(sidecar), HISTORICAL_IMAGEGEN_KEYS)
        self.assertEqual(sidecar["kind"], "built_in_imagegen_registration")
        self.assertEqual(sidecar["source_image"], "demo-s002.png")
        self.assertEqual([sidecar["width"], sidecar["height"]], [64, 36])

    def test_imagegen_source_must_stay_under_its_output_root(self) -> None:
        stray = png(self.outside / "demo-s002.png")
        with self.assertRaisesRegex(SystemExit, "beneath built-in ImageGen output root"):
            self.run_cli("demo-s002", str(stray), "--source-root", str(self.imagegen))

    def test_supplied_fields_are_refused_for_imagegen(self) -> None:
        source = png(self.imagegen / "demo-s002.png")
        with self.assertRaisesRegex(SystemExit, "--maker apply only"):
            self.run_cli("demo-s002", str(source), "--source-root", str(self.imagegen),
                         "--maker", "demo maker")

    def test_references_must_be_registered_first_and_are_reported_together(self) -> None:
        doc = json.loads(self.manifest.read_text())
        doc["stills"][1]["refs"]["all"].append("art/canon/demo-missing.png")
        self.manifest.write_text(json.dumps(doc))
        with self.assertRaises(SystemExit) as caught:
            self.imagegen_register("demo-s001", "--grounding-fact", "demo-lamp")
        self.assertIn("art/canon/demo-table.png", str(caught.exception))
        self.assertIn("art/canon/demo-missing.png", str(caught.exception))

    def test_plate_skips_line_range_fact_and_records_the_skip(self) -> None:
        self.imagegen_register("demo-table", "--no-visible-identity")
        skips = self.sidecar("demo-table")["continuity_fact_skips"]
        self.assertEqual([item["id"] for item in skips], ["demo-lamp"])
        self.assertEqual(self.record("demo-table")["grounding_claims"]["continuity_fact_skips"], skips)
        # The numbered still that references the plate still needs its fact.
        with self.assertRaisesRegex(SystemExit, "demo-lamp"):
            self.imagegen_register("demo-s001")
        self.imagegen_register("demo-s001", "--grounding-fact", "demo-lamp")
        self.assertEqual(self.sidecar("demo-s001")["continuity_facts"][0]["id"], "demo-lamp")


class SuppliedOriginTests(Project):
    def supplied(self, still_id: str, source: Path, *extra: str) -> object:
        return self.run_cli(still_id, str(source), "--origin", "human-made",
                            "--maker", "demo painter", "--rights-basis", "made for this film",
                            "--provider-upload", "forbidden", *extra)

    def test_human_made_still_records_a_truthful_chain(self) -> None:
        self.imagegen_register("demo-table", "--no-visible-identity")
        original = png(self.root / "incoming/demo-s001-scan.jpg", fmt="JPEG", size=(128, 72))
        source = png(self.root / "incoming/demo-s001.png", size=(128, 72))
        self.supplied("demo-s001", source, "--original", str(original),
                      "--grounding-fact", "demo-lamp", "--policy", "story/still-policy.json",
                      "--digital-source-type", "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture")
        record = self.record("demo-s001")
        self.assertEqual(record["generation_provider"], "human-made")
        self.assertEqual(record["still_origin"], "human-made")
        self.assertTrue(record["provenance_sidecar"].endswith(".still.json"))
        sidecar = self.sidecar("demo-s001")
        self.assertEqual(sidecar["kind"], "supplied_still_registration")
        self.assertEqual(sidecar["provider"], "human-made")
        self.assertEqual(sidecar["maker"], "demo painter")
        self.assertEqual(sidecar["rights"], {"basis": "made for this film", "providerUpload": "forbidden"})
        self.assertEqual(sidecar["source_root"], "project")
        self.assertEqual(sidecar["source_image"], "incoming/demo-s001.png")
        self.assertEqual(sidecar["requested_size"], "1536x1024")
        self.assertEqual(sidecar["original"]["format"], "jpeg")
        self.assertEqual([sidecar["original"]["width"], sidecar["original"]["height"]], [128, 72])
        self.assertEqual(sidecar["original"]["sha256"], _common.sha256_file(original))
        table = self.root / "art/canon/demo-table.png"
        self.assertEqual(sidecar["reference_hashes"],
                         [{"ref": "art/canon/demo-table.png", "sha256": _common.sha256_file(table)}])
        assets = asset_registry.load_registry(self.root)
        self.assertIn("art/stills/demo-s001.png", assets)
        self.assertEqual(asset_registry.downstream(assets, "art/canon/demo-table.png"),
                         ["art/stills/demo-s001.png"])

    def test_imagegen_refs_rows_keep_their_registry_edge(self) -> None:
        self.imagegen_register("demo-table", "--no-visible-identity")
        doc = json.loads(self.manifest.read_text())
        doc["stills"][2]["imagegen_refs"] = ["art/canon/demo-table.png"]
        self.manifest.write_text(json.dumps(doc))
        self.supplied("demo-s002", png(self.root / "incoming/demo-s002.png"), "--no-visible-identity")
        self.assertEqual(self.sidecar("demo-s002")["refs"], ["art/canon/demo-table.png"])
        assets = asset_registry.load_registry(self.root)
        self.assertEqual(asset_registry.downstream(assets, "art/canon/demo-table.png"),
                         ["art/stills/demo-s002.png"])

    def test_required_supplied_fields(self) -> None:
        source = png(self.root / "incoming/demo-s002.png")
        with self.assertRaisesRegex(SystemExit, "--maker or --supplied-by"):
            self.run_cli("demo-s002", str(source), "--origin", "third-party",
                         "--rights-basis", "licensed", "--provider-upload", "allowed")
        with self.assertRaisesRegex(SystemExit, "--rights-basis"):
            self.run_cli("demo-s002", str(source), "--origin", "third-party",
                         "--supplied-by", "demo archive", "--provider-upload", "allowed")
        with self.assertRaisesRegex(SystemExit, "--provider-upload"):
            self.run_cli("demo-s002", str(source), "--origin", "third-party",
                         "--supplied-by", "demo archive", "--rights-basis", "licensed")

    def test_jpeg_target_is_refused_with_the_conversion_route(self) -> None:
        source = png(self.root / "incoming/demo-s002.jpg", fmt="JPEG")
        with self.assertRaisesRegex(SystemExit, "--original"):
            self.supplied("demo-s002", source, "--no-visible-identity")

    def test_other_tool_needs_a_declared_root_for_outside_files(self) -> None:
        source = png(self.outside / "demo-s002.png")
        args = ("demo-s002", str(source), "--origin", "other-tool:demo-painter-app",
                "--supplied-by", "demo operator", "--rights-basis", "tool terms",
                "--provider-upload", "allowed")
        with self.assertRaisesRegex(SystemExit, "inside the project"):
            self.run_cli(*args)
        self.run_cli(*args, "--source-root", str(self.outside))
        sidecar = self.sidecar("demo-s002")
        self.assertEqual(sidecar["provider"], "demo-painter-app")
        self.assertEqual(sidecar["source_root"], "--source-root")
        self.assertEqual(sidecar["source_image"], "demo-s002.png")

    def test_bad_origin_is_refused(self) -> None:
        source = png(self.root / "incoming/demo-s002.png")
        with self.assertRaisesRegex(SystemExit, "--origin must be"):
            self.run_cli("demo-s002", str(source), "--origin", "other-tool:")


class ContinuityTests(unittest.TestCase):
    policy = {"grounding_guard": {"continuity_facts": [
        {"id": "range", "line_range": [10, 12], "value": "range fact"},
        {"id": "span", "from": "demo-s002", "to": "demo-s003", "value": "span fact"},
    ]}}

    def test_suffixes_keep_their_line_number(self) -> None:
        for still_id, line in (("demo-s010", "010"), ("demo-s010b", "010"), ("demo-s010-alt", "010"),
                               ("demo-x-s011", "011")):
            self.assertEqual(registrar.numbered_still(still_id).group("line"), line, still_id)
        self.assertEqual(registrar.numbered_still("demo-x-s011").group("project"), "demo-x")
        self.assertIsNone(registrar.numbered_still("demo-plate"))
        facts = registrar.required_continuity_facts(
            {"grounding_guard": {"continuity_facts": [self.policy["grounding_guard"]["continuity_facts"][0]]}},
            "demo-s011-alt")
        self.assertEqual([fact["id"] for fact in facts], ["range"])

    def test_from_to_resolves_by_storyboard_order(self) -> None:
        order = ["demo-s001", "demo-s003", "demo-s002", "demo-s004"]
        policy = {"grounding_guard": {"continuity_facts": [self.policy["grounding_guard"]["continuity_facts"][1]]}}
        self.assertEqual(registrar.required_continuity_facts(policy, "demo-s001", order), [])
        self.assertEqual(registrar.required_continuity_facts(policy, "demo-s003", order), [])
        reordered = ["demo-s001", "demo-s002", "demo-s004", "demo-s003"]
        self.assertEqual([f["id"] for f in registrar.required_continuity_facts(policy, "demo-s004", reordered)],
                         ["span"])
        required, skipped = registrar.resolve_continuity_facts(policy, "demo-plate", reordered)
        self.assertEqual((required, [item["id"] for item in skipped]), ([], ["span"]))
        with self.assertRaisesRegex(registrar.GroundingError, "--storyboard"):
            registrar.required_continuity_facts(policy, "demo-s002")
        with self.assertRaisesRegex(registrar.GroundingError, "not on the storyboard"):
            registrar.required_continuity_facts(policy, "demo-s002", ["demo-s002"])


class CanonTests(Project):
    def test_canon_reference_resolves_only_under_an_explicit_root(self) -> None:
        canon = self.outside / "series-canon"
        sheet = png(canon / "sheets/demo-sheet.png")
        doc = json.loads(self.manifest.read_text())
        doc["stills"][2]["refs"]["all"] = ["canon:sheets/demo-sheet.png"]
        self.manifest.write_text(json.dumps(doc))
        with self.assertRaisesRegex(SystemExit, "--canon-root"):
            self.imagegen_register("demo-s002", "--no-visible-identity")
        with self.assertRaisesRegex(SystemExit, "escapes canon root"):
            doc["stills"][2]["refs"]["all"] = ["canon:../demo-s002.png"]
            self.manifest.write_text(json.dumps(doc))
            png(self.outside / "demo-s002.png")
            self.imagegen_register("demo-s002", "--no-visible-identity", "--canon-root", str(canon))
        doc["stills"][2]["refs"]["all"] = ["canon:sheets/demo-sheet.png"]
        self.manifest.write_text(json.dumps(doc))
        self.imagegen_register("demo-s002", "--no-visible-identity", "--canon-root", str(canon))
        sidecar = self.sidecar("demo-s002")
        self.assertEqual(sidecar["canon_refs"],
                         [{"ref": "canon:sheets/demo-sheet.png", "sha256": _common.sha256_file(sheet)}])
        assets = asset_registry.load_registry(self.root)
        self.assertEqual(asset_registry.downstream(assets, "canon:sheets/demo-sheet.png"),
                         ["art/stills/demo-s002.png"])


class RetakeAndBatchTests(Project):
    def test_retake_retires_bytes_and_lists_impact(self) -> None:
        self.imagegen_register("demo-table", "--no-visible-identity")
        self.imagegen_register("demo-s001", "--grounding-fact", "demo-lamp")
        table = self.root / "art/canon/demo-table.png"
        digest = _common.sha256_file(table)
        result = self.run_cli("demo-table", "--retake", "--note", "demo retake")
        retired = self.root / "art/retired/demo-table" / digest[:12]
        self.assertTrue((retired / "demo-table.png").is_file())
        self.assertTrue((retired / "demo-table.imagegen.json").is_file())
        self.assertFalse(table.exists())
        self.assertEqual(result["downstream"], ["art/stills/demo-s001.png"])
        record = self.record("demo-table")
        self.assertEqual(record["status"], "needs_changes")
        self.assertNotIn("provenance_sidecar", record)
        self.assertEqual(record["retake_history"][0]["target_sha256"], digest)
        assets = asset_registry.load_registry(self.root)
        self.assertTrue(assets["art/retired/demo-table/%s/demo-table.png" % digest[:12]]["retired"])
        self.assertNotIn("art/canon/demo-table.png", assets)
        # The freed row registers again; the registry then holds both.
        png(self.imagegen / "demo-table-v2.png", color=(200, 10, 10))
        self.run_cli("demo-table", str(self.imagegen / "demo-table-v2.png"),
                     "--source-root", str(self.imagegen), "--no-visible-identity")
        self.assertIn("art/canon/demo-table.png", asset_registry.load_registry(self.root))
        with self.assertRaisesRegex(SystemExit, "only a registered still"):
            self.run_cli("demo-s002", "--retake")

    def test_batch_stops_at_the_first_refusal(self) -> None:
        png(self.imagegen / "demo-table.png")
        png(self.imagegen / "demo-s001.png")
        batch = self.root / "work/batch.json"
        batch.parent.mkdir(parents=True, exist_ok=True)
        batch.write_text(json.dumps({"registrations": [
            {"still_id": "demo-table", "source_image": str(self.imagegen / "demo-table.png"),
             "no_visible_identity": True},
            {"still_id": "demo-s001", "source_image": str(self.imagegen / "demo-s001.png")},
            {"still_id": "demo-s002", "source_image": str(self.imagegen / "demo-s002.png")},
        ]}))
        with self.assertRaisesRegex(SystemExit, r"row 2 \(demo-s001\).*demo-lamp.*\['demo-table'\]"):
            self.run_cli("--batch", str(batch), "--source-root", str(self.imagegen),
                         "--policy", "story/still-policy.json")
        self.assertEqual(self.record("demo-table")["status"], "generated")
        self.assertEqual(self.record("demo-s001")["status"], "generation_ready")
        self.assertEqual(self.record("demo-s002")["status"], "generation_ready")


class DeriveTests(Project):
    def setUp(self) -> None:
        super().setUp()
        self.imagegen_register("demo-table", "--no-visible-identity")
        image = Image.new("RGB", (64, 36), "white")
        image.paste((255, 0, 0), (0, 0, 32, 36))
        image.save(self.root / "incoming-parent.png")
        self.parent = self.root / "incoming-parent.png"

    def derive(self, *argv: str) -> None:
        with redirect_stdout(io.StringIO()):
            derive_still.main(["story/generation_manifest.json", "demo-s002", *argv,
                               "--root", str(self.root)])

    def test_crop_with_explicit_anchor_records_lineage(self) -> None:
        self.derive("art/canon/demo-table.png", "--operation", "crop", "--size", "36x36",
                    "--anchor", "left")
        record = self.record("demo-s002")
        self.assertEqual(record["generation_provider"], "derived")
        sidecar = self.sidecar("demo-s002")
        target = self.root / "art/stills/demo-s002.png"
        self.assertEqual(sidecar["target_sha256"], _common.sha256_file(target))
        self.assertEqual(sidecar["parent"]["path"], "art/canon/demo-table.png")
        self.assertEqual(sidecar["parent"]["sidecar"]["path"], "art/canon/demo-table.imagegen.json")
        self.assertEqual(sidecar["operation"], {"name": "crop", "width": 36, "height": 36,
                                                "parentWidth": 64, "parentHeight": 36,
                                                "anchor": "left", "x": 0, "y": 0})
        self.assertEqual(sidecar["tool"]["sha256"], _common.sha256_file(Path(derive_still.__file__)))
        with Image.open(target) as written:
            self.assertEqual(written.size, (36, 36))
        assets = asset_registry.load_registry(self.root)
        self.assertEqual(asset_registry.downstream(assets, "art/canon/demo-table.png"),
                         ["art/stills/demo-s002.png"])
        with self.assertRaisesRegex(SystemExit, "refusing to overwrite|non-registerable"):
            self.derive("art/canon/demo-table.png", "--operation", "crop", "--size", "36x36",
                        "--anchor", "left")

    def test_crop_needs_a_placement_and_scale_keeps_aspect(self) -> None:
        with self.assertRaisesRegex(SystemExit, "explicit --anchor"):
            self.derive("incoming-parent.png", "--operation", "crop", "--size", "32x32")
        with self.assertRaisesRegex(SystemExit, "aspect ratio"):
            self.derive("incoming-parent.png", "--operation", "scale", "--size", "64x64")
        self.assertFalse((self.root / "art/stills/demo-s002.png").exists())
        self.derive("incoming-parent.png", "--operation", "scale", "--size", "32x18")
        self.assertEqual(self.sidecar("demo-s002")["height"], 18)

    def test_pad_places_parent_at_anchor(self) -> None:
        self.derive("incoming-parent.png", "--operation", "pad", "--size", "64x64",
                    "--anchor", "bottom", "--color", "#000000")
        with Image.open(self.root / "art/stills/demo-s002.png") as written:
            self.assertEqual(written.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(written.getpixel((0, 63)), (255, 0, 0))

    def test_exact_frame_keeps_the_worker_rule(self) -> None:
        clip = self.root / "motion/demo-clip.mp4"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"not decoded in this test")
        with mock.patch.object(_common.sys, "platform", "darwin"):
            with self.assertRaises(_common.MediaWorkerRequired):
                _common.extract_exact_frame(clip, 0, self.root / "unused.png")
        with mock.patch.object(_common.sys, "platform", "darwin"):
            with self.assertRaisesRegex(SystemExit, "qualified GPU worker"):
                derive_still.derive(root=self.root, manifest_path=self.manifest,
                                    still_id="demo-s002", parent=clip, operation="exact-frame",
                                    frame=3)
        self.assertFalse((self.root / "art/stills/demo-s002.png").exists())

        def fake(source, index, destination):
            png(destination, size=(48, 27))
        derive_still.derive(root=self.root, manifest_path=self.manifest, still_id="demo-s002",
                            parent=clip, operation="exact-frame", frame=3, extractor=fake)
        self.assertEqual(self.sidecar("demo-s002")["operation"], {"name": "exact-frame", "frameIndex": 3})

    def test_production_inventory_accepts_a_derived_still(self) -> None:
        self.derive("incoming-parent.png", "--operation", "scale", "--size", "32x18")
        record = self.record("demo-s002")
        still = self.root / record["target_output_path"]
        sidecar = json.loads((self.root / record["provenance_sidecar"]).read_text())
        # prepare_episode_production binds exactly these two fields.
        self.assertEqual(sidecar["target_sha256"], _common.sha256_file(still))


class ReferencePackTests(Project):
    def test_pack_round_trips_through_execution_coverage(self) -> None:
        self.imagegen_register("demo-table", "--no-visible-identity")
        self.imagegen_register("demo-s002", "--no-visible-identity")
        policy = json.loads((self.root / "story/still-policy.json").read_text())
        record = make_reference_pack.make_pack(
            root=self.root, output=Path("art/refs/demo-pack.png"),
            sources=["art/canon/demo-table.png", "art/stills/demo-s002.png=window"], policy=policy)
        self.assertEqual([row["label"] for row in record["sources"]], ["demo table", "window"])
        coverage = registrar.expand_execution_coverage(self.root, ["art/refs/demo-pack.png"])
        self.assertEqual(coverage["art/canon/demo-table.png"]["mode"], "packed")
        self.assertEqual(coverage["art/stills/demo-s002.png"]["mode"], "packed")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            make_reference_pack.make_pack(root=self.root, output=Path("art/refs/demo-pack.png"),
                                          sources=["art/canon/demo-table.png"])

    def test_unregistered_or_changed_sources_are_refused(self) -> None:
        self.imagegen_register("demo-table", "--no-visible-identity")
        png(self.root / "art/stills/loose.png")
        with self.assertRaisesRegex(ValueError, "not a registered asset"):
            make_reference_pack.make_pack(root=self.root, output=Path("art/refs/p.png"),
                                          sources=["art/stills/loose.png"])
        (self.root / "art/canon/demo-table.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "does not match"):
            make_reference_pack.make_pack(root=self.root, output=Path("art/refs/p.png"),
                                          sources=["art/canon/demo-table.png"])


class BoardTests(Project):
    def test_facts_and_board_read_registered_state(self) -> None:
        self.imagegen_register("demo-table", "--no-visible-identity")
        self.imagegen_register("demo-s001", "--grounding-fact", "demo-lamp")
        policy = json.loads((self.root / "story/still-policy.json").read_text())
        facts = asset_registry.active_facts(policy, "demo-s001", None)
        self.assertEqual(facts["facts"], [{"id": "demo-lamp", "value": "demo-s001: the lamp is on the table"}])
        storyboard = json.loads((self.root / "story/storyboard_plan.json").read_text())
        manifest = json.loads(self.manifest.read_text())
        out = self.root / "work/still-board.html"
        page = asset_registry.board_html(self.root, policy, storyboard, manifest, out)
        self.assertIn("<h2>demo table</h2>", page)
        self.assertIn("demo-s001", page)
        self.assertIn("demo-lamp", page)
        self.assertIn('src="../art/stills/demo-s001.png"', page)
        self.assertNotIn("demo-s002", page)
        with mock.patch("sys.argv", ["asset_registry.py", str(self.root), "--board", "--policy",
                                     "story/still-policy.json", "--storyboard",
                                     "story/storyboard_plan.json", "--manifest",
                                     "story/generation_manifest.json", "--out", "story/board.html"]):
            with self.assertRaisesRegex(ValueError, "work/"):
                asset_registry.main()
        with mock.patch("sys.argv", ["asset_registry.py", str(self.root), "--board", "--policy",
                                     "story/still-policy.json"]):
            with self.assertRaisesRegex(ValueError, "--storyboard must be given"):
                asset_registry.main()


class HeaderTests(unittest.TestCase):
    def test_png_jpeg_and_webp_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cases = [("a.png", "PNG", {}), ("a.jpg", "JPEG", {}), ("a.webp", "WEBP", {}),
                     ("b.webp", "WEBP", {"lossless": True})]
            for name, fmt, options in cases:
                path = png(base / name, size=(321, 123), fmt=fmt, **options)
                self.assertEqual(_common.image_dimensions(path)[1:], (321, 123), name)
            (base / "x.bin").write_bytes(b"neither")
            with self.assertRaisesRegex(ValueError, "x.bin"):
                _common.image_dimensions(base / "x.bin")


if __name__ == "__main__":
    unittest.main()
