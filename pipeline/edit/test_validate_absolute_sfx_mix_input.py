#!/usr/bin/env python3
"""Source-bound final admission tests using the retained renderer's synthetic fixture."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from pipeline.edit import validate_absolute_sfx_mix_input as adapter
from pipeline.edit import test_render_designed_sfx as fixtures


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


class AbsoluteSfxMixInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = fixtures.DesignedSfxContractTests('test_scene_offset_maps_to_current_clock')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.clock.update(schemaVersion=2, language='hi', timelineFingerprint='a' * 64)
        for index, event in enumerate(self.fixture.clock['events']):
            event.update(id=f'scene-{index}', kind='narration')
        self.fixture.make_final()
        self.fixture.design['cues'].append(self.fixture.cue(id='cue-b', sourceBatch='b', shot=2))
        self.fixture.design['scoreNotchAutomation'] = [{
            'cueId': 'cue-a', 'startSeconds': 0.3, 'endSeconds': 0.9,
            'scoreGainDb': -3, 'attackSeconds': 0.08, 'releaseSeconds': 0.22}]
        self.episode = self.fixture.ep
        self.absolute_design = self.fixture.root / 'design.json'
        self.render_audit = self.fixture.root / 'mix-audit.json'
        self.edit_plan = self.episode / self.fixture.design['clock']['path']
        self.picture_segments = self.episode / self.fixture.design['picture']['path']
        self.review_path = self.fixture.root / 'context-review.json'
        self.regenerate()

    def bind(self, path):
        return {'path': str(path.relative_to(self.episode)), 'sha256': adapter.sha256_file(path)}

    def regenerate(self):
        self.audit, _, _ = self.fixture.render_synthetic()
        self.stem = Path(self.audit['stem'])
        self.review = {'schemaVersion': 1, 'kind': 'sfx-context-review', 'status': 'approved',
            'scope': 'sfx-in-picture-voice-music-context', 'independentSfxOnly': True,
            'reviewedBy': 'Synthetic fixture reviewer', 'reviewerRole': 'user',
            'designSha256': adapter.sha256_file(self.absolute_design),
            'renderAuditSha256': adapter.sha256_file(self.render_audit),
            'stemSha256': adapter.sha256_file(self.stem),
            'approvedCueIds': [cue['id'] for cue in self.fixture.design['cues']], 'inputEvidence': {}}
        for key in ('picture', 'voice', 'music', 'contextPreview'):
            path = self.fixture.root / f'{key}.fixture'
            path.write_bytes(f'SYNTHETIC {key} ONLY'.encode())
            self.review['inputEvidence'][key] = self.bind(path)
        write_json(self.review_path, self.review)

    def build(self, include_review=True):
        return adapter.build_contract(self.episode, self.absolute_design, self.render_audit,
                                      self.edit_plan, self.picture_segments,
                                      self.review_path if include_review else None)

    def test_accepts_only_final_hash_bound_independent_stem(self):
        contract = self.build()
        self.assertEqual(contract['schemaVersion'], 2)
        self.assertEqual(contract['kind'], 'stable-audio-absolute-sfx-final-mix-input')
        self.assertEqual(contract['sourceCount'], 2)
        self.assertEqual(contract['cueCount'], 2)
        self.assertEqual(contract['sfxStem']['frames'], 480_000)
        self.assertTrue(contract['consumerRequirements']['muxMustNotUseShortest'])
        self.assertFalse(self.audit['finalApproved'])  # a renderer flag is not a verdict
        self.assertEqual(contract['inputEvidence']['contextReview']['sha256'], adapter.sha256_file(self.review_path))

    def test_review_is_optional_and_does_not_grant_creative_approval(self):
        contract = self.build(include_review=False)
        self.assertIsNone(contract['contextReview'])
        self.assertNotIn('contextReview', contract['inputEvidence'])
        self.assertEqual(contract['status'], 'technically-validated-sfx-input-review-not-implied')
        # Notes need no prescribed reviewer role, approval vocabulary or cue order.
        write_json(self.review_path, {'status': 'pending', 'note': 'Try a quieter ending.'})
        contract = self.build()
        self.assertEqual(contract['contextReview']['recordedStatus'], 'pending')
        self.assertFalse(contract['contextReview']['bindsCurrentRender'])
        self.assertEqual(contract['status'], 'technically-validated-sfx-input-review-not-implied')

    def test_stale_hash_and_preview_audit_are_refused(self):
        for path, message in ((self.stem, 'rendered SFX stem changed'),
                              (self.episode / self.review['inputEvidence']['voice']['path'], 'reviewed voice changed')):
            original = path.read_bytes()
            path.write_bytes(b'CHANGED')
            with self.subTest(path=path.name), self.assertRaisesRegex(adapter.ContractError, message):
                self.build()
            path.write_bytes(original)
        self.audit['workingAudioOnly'] = True
        write_json(self.render_audit, self.audit)
        with self.assertRaisesRegex(adapter.ContractError, 'final-capable'):
            self.build()

    def test_route_contamination_and_timeline_drift_are_refused(self):
        design = json.loads(self.absolute_design.read_text())
        design['generationApiAllowed'] = True
        write_json(self.absolute_design, design)
        with self.assertRaisesRegex(adapter.ContractError, 'no generation API'):
            self.build()
        self.regenerate()
        self.edit_plan.write_text(self.edit_plan.read_text() + ' ')
        with self.assertRaisesRegex(adapter.ContractError, 'sources or timeline'):
            self.build()

    def test_malformed_notches_and_forged_renderer_receipt_are_refused(self):
        self.fixture.design['scoreNotchAutomation'][0].pop('scoreGainDb')
        self.regenerate()
        with self.assertRaisesRegex(adapter.ContractError, 'scoreGainDb'):
            self.build()
        self.fixture.design['scoreNotchAutomation'][0]['scoreGainDb'] = -3
        self.regenerate()
        self.audit['rendererSha256'] = 'b' * 64
        write_json(self.render_audit, self.audit)
        with self.assertRaisesRegex(adapter.ContractError, 'retained renderer changed'):
            self.build()


if __name__ == '__main__':
    unittest.main()
