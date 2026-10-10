"""Offline safety tests; no model calls or media processing."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location('dialogue_finite_target', Path(__file__).with_name('kie_dialogue_finite.py'))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class DialogueFiniteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.patch = mock.patch.object(RUNNER, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.ep = self.root / 'private-project'
        self.work = self.ep / 'work/dialogue'
        self.work.mkdir(parents=True)
        source = self.ep / 'art/stills/demo-s1001.png'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'not-a-real-image-no-media-processing')
        prompt = self.work / 'prompts/demo-s1001.txt'
        prompt.parent.mkdir()
        prompt.write_text('Exact Hindi: नमस्ते।', encoding='utf-8')
        self.source = source
        self.doc = dict(version=2,episode='DEMO',provider='kie.ai',route='grok15',
                        model='grok-imagine-video-1-5-preview',resolution='720p',
                        aspectRatio='16:9',language='hi-IN',financialApproval=True,
                        financialState='offline fixture',jobCount=1,maximumGeneratedSeconds=7, maximumCredits=35,
                        pricing={'creditsPerSecond':5,'verifiedAtUtc':'2026-01-01T00:00:00Z',
                                 'evidence':'offline test fixture; not a provider quote'})
        for key, file in [('draftManifest',self.work/'draft-manifest.json'),
                          ('newAngleAudit',self.work/'source-audit.json'),
                          ('productionOverrides',self.ep/'story/production-overrides-v1.json'),
                          ('authority',self.ep/'story/GOAL.md')]:
            file.parent.mkdir(parents=True,exist_ok=True)
            file.write_text('{}\n',encoding='utf-8')
            self.doc[key] = str(file.relative_to(self.ep))
            self.doc[key+'Sha256'] = RUNNER.sha256(file)
        self.doc['jobs'] = [dict(id='demo-s1001',speaker='Test character',hindi='नमस्ते।',durationSeconds=7,
            source=str(source.relative_to(self.ep)),sourceSha256=RUNNER.sha256(source),
            prompt=str(prompt.relative_to(self.ep)),promptSha256=RUNNER.sha256(prompt),
            outputStem=str((self.work/'motion/demo-s1001').relative_to(self.ep)))]
        self.manifest = self.work/'execution-manifest.json'
        self.save()
        RUNNER.configure(self.manifest,self.ep)

    def save(self):
        self.manifest.write_text(json.dumps(self.doc,ensure_ascii=False),encoding='utf-8')

    def test_dry_run_is_offline_and_accepts_seven_seconds(self):
        with mock.patch.object(RUNNER.subprocess,'run') as run:
            result = RUNNER.execute(7,False)
        run.assert_not_called()
        self.assertEqual(result['networkCalls'],0)
        self.assertEqual(result['generatedSeconds'],7)

    def test_changed_source_refused(self):
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(RUNNER.PlanError,'source missing or hash changed'):
            RUNNER.execute(7,False)

    def test_explicit_versioned_overrides_remain_hash_bound(self):
        versioned = self.ep/'story/production-v2/overrides.json'
        versioned.parent.mkdir()
        versioned.write_text('{"version":2}\n')
        self.doc['productionOverrides'] = str(versioned.relative_to(self.ep))
        self.doc['productionOverridesSha256'] = RUNNER.sha256(versioned)
        self.save()
        RUNNER.configure(self.manifest, self.ep, versioned)
        self.assertEqual(RUNNER.execute(7,False)['networkCalls'], 0)
        versioned.write_text('{"version":3}\n')
        with self.assertRaisesRegex(RUNNER.PlanError,'productionOverrides changed'):
            RUNNER.execute(7,False)

    def test_explicit_overrides_cannot_escape_episode_story(self):
        with self.assertRaisesRegex(RUNNER.PlanError,'production overrides escapes'):
            RUNNER.configure(self.manifest, self.ep, self.work/'overrides.json')

    def test_wrong_total_refused(self):
        with self.assertRaisesRegex(RUNNER.PlanError,'exactly equal'):
            RUNNER.execute(8,False)

    def test_false_financial_approval_stops_before_compute(self):
        self.doc['financialApproval'] = False
        self.save()
        with mock.patch.object(RUNNER.subprocess,'run') as run:
            with self.assertRaisesRegex(RUNNER.PlanError,'submission is gated'):
                RUNNER.execute(7,True)
        run.assert_not_called()

    def test_mac_submission_refused(self):
        with mock.patch.object(RUNNER.platform,'system',return_value='Darwin'), mock.patch.object(RUNNER.subprocess,'run') as run:
            with self.assertRaisesRegex(RUNNER.PlanError,'qualified GPU worker'):
                RUNNER.execute(7,True)
        run.assert_not_called()

    def test_authority_tamper_refused(self):
        (self.ep/'story/GOAL.md').write_text('changed authority')
        with self.assertRaisesRegex(RUNNER.PlanError,'authority changed'):
            RUNNER.execute(7,False)

    def test_exact_line_required(self):
        self.doc['jobs'][0]['hindi']='बदला संवाद।'
        self.save()
        with self.assertRaisesRegex(RUNNER.PlanError,'exact Hindi missing'):
            RUNNER.execute(7,False)


if __name__ == '__main__':
    unittest.main()
