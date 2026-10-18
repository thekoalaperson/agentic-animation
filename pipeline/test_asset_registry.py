"""Offline tests for the read-only asset registry's case and relocation reports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pipeline import asset_registry as registry


class AssetRegistryTest(unittest.TestCase):
    def project(self, root: Path, refs) -> Path:
        for name in ("art/canon/demo-sheet.png", "art/stills/demo-s001.png"):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(b"offline image")
        (root / "art/canon/demo-sheet.imagegen.json").write_text(json.dumps(
            {"target_output_path": "art/canon/demo-sheet.png", "refs": []}))
        (root / "art/stills/demo-s001.imagegen.json").write_text(json.dumps(
            {"target_output_path": "art/stills/demo-s001.png", "refs": refs}))
        return root

    def test_case_variants_are_reported_not_merged_and_missing_refs_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.project(Path(tmp).resolve(),
                                ["art/canon/Demo-Sheet.png", "art/canon/missing.png"])
            with mock.patch.object(registry, "case_insensitive", return_value=True):
                assets = registry.load_registry(root)
                report = registry.integrity_report(assets, root)
                impact = registry.downstream(assets, "art/canon/demo-sheet.png", True)
            self.assertEqual(len(assets), 2)
            self.assertEqual(report["caseVariants"],
                             [["art/canon/Demo-Sheet.png", "art/canon/demo-sheet.png"]])
            self.assertEqual(report["unresolvedReferences"],
                             [{"asset": "art/stills/demo-s001.png", "reference": "art/canon/missing.png"}])
            self.assertEqual(impact, ["art/stills/demo-s001.png"])

    def test_case_sensitive_disk_reports_no_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.project(Path(tmp).resolve(), ["art/canon/demo-sheet.png"])
            with mock.patch.object(registry, "case_insensitive", return_value=False):
                report = registry.integrity_report(registry.load_registry(root), root)
            self.assertEqual(report, {"caseVariants": [], "unresolvedReferences": []})

    def test_series_canon_references_are_neither_unresolved_nor_case_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.project(Path(tmp).resolve(), ["canon:sheets/demo-sheet.png",
                                                      "canon:sheets/Demo-Sheet.png"])
            with mock.patch.object(registry, "case_insensitive", return_value=True):
                report = registry.integrity_report(registry.load_registry(root), root)
            self.assertEqual(report, {"caseVariants": [], "unresolvedReferences": []})

    def test_relocation_map_resolves_paths_recorded_before_a_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = "/old/studio/demo-project/art/canon/demo-sheet.png"
            root = self.project(Path(tmp).resolve(), [old])
            moved = registry.load_registry(root)
            self.assertEqual(moved["art/stills/demo-s001.png"]["refs"], [old])
            relocation = {"originalEpisodeRoot": "/old/studio/demo-project", "paths": {}}
            mapped = registry.load_registry(root, relocation)
            self.assertEqual(mapped["art/stills/demo-s001.png"]["refs"], ["art/canon/demo-sheet.png"])
            self.assertEqual(registry.integrity_report(mapped, root)["unresolvedReferences"], [])


if __name__ == "__main__":
    unittest.main()
