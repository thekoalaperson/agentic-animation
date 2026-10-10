#!/usr/bin/env python3
"""Unit tests for source-bound image registration."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from register_builtin_imagegen import (
        GroundingError,
        expand_execution_coverage,
        validate_continuity_claims,
        validate_exact_identity_claims,
    )
except ModuleNotFoundError:  # unittest discovery from the repository root
    from pipeline.register_builtin_imagegen import (
        GroundingError,
        expand_execution_coverage,
        validate_continuity_claims,
        validate_exact_identity_claims,
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GroundingGuardTests(unittest.TestCase):
    def test_packed_identity_is_auditable_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "canon.png"
            pack = root / "pack.png"
            source.write_bytes(b"canon")
            pack.write_bytes(b"pack")
            (root / "pack.refpack.json").write_text(
                json.dumps(
                    {
                        "kind": "built_in_imagegen_reference_pack",
                        "pack": "pack.png",
                        "pack_sha256": digest(pack),
                        "sources": [
                            {"source": "canon.png", "source_sha256": digest(source)}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            coverage = expand_execution_coverage(root, ["pack.png"])
            self.assertEqual(coverage["canon.png"]["mode"], "packed")

    def test_uncovered_named_exact_claim_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            other = root / "other.png"
            other.write_bytes(b"other")
            policy = {
                "grounding_guard": {
                    "identity_authorities": [
                        {
                            "name": "Character One",
                            "aliases": ["first character"],
                            "sources": ["character-one.png"],
                        }
                    ]
                }
            }
            with self.assertRaisesRegex(GroundingError, "Character One"):
                validate_exact_identity_claims(
                    root=root,
                    policy=policy,
                    execution_refs=["other.png"],
                    structured_claims=[],
                    free_text="exact archived Character One identity",
                )

    def test_demo_required_facts_are_range_aware(self) -> None:
        policy = {
            "grounding_guard": {
                "continuity_facts": [
                    {
                        "id": "hall",
                        "episode": "DEMO",
                        "line_range": [1, 140],
                        "value": "zones",
                    },
                    {
                        "id": "cloth",
                        "episode": "DEMO",
                        "line_range": [108, 133],
                        "value": "crimson-gold",
                    },
                ]
            }
        }
        with self.assertRaisesRegex(GroundingError, "cloth"):
            validate_continuity_claims(
                policy=policy, still_id="demo-s130", supplied_fact_ids=["hall"]
            )
        claims = validate_continuity_claims(
            policy=policy,
            still_id="demo-s130",
            supplied_fact_ids=["hall", "cloth"],
        )
        self.assertEqual([item["id"] for item in claims], ["hall", "cloth"])



if __name__ == "__main__":
    unittest.main()
