"""Offline safety tests; no model calls or media processing."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


from pipeline.test_paid_approval import write_approval

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
                        aspectRatio='16:9',language='hi-IN',financialApproval=False,
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

    def approval(self):
        plan = RUNNER.load_plan(self.manifest, 7)
        return write_approval(self.ep/'story/approvals/dialogue.json', episode='DEMO',
                              lane=RUNNER.APPROVAL_LANE, manifest=self.manifest,
                              scope=RUNNER.approval_scope(plan), project_root=self.ep)

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

    def test_manifest_boolean_alone_never_authorizes(self):
        for boolean in (False, True):
            with self.subTest(financialApproval=boolean):
                self.doc['financialApproval'] = boolean
                self.save()
                with mock.patch.object(RUNNER.subprocess,'run') as run:
                    with self.assertRaisesRegex(RUNNER.PlanError,'requires a current approval record'):
                        RUNNER.execute(7,True)
                    dry = RUNNER.execute(7,False)
                run.assert_not_called()
                self.assertEqual((dry['financialApproval'], dry['financialAuthorized']), (boolean, False))
        self.assertTrue(RUNNER.execute(7,False,self.approval())['financialAuthorized'])

    def test_mac_submission_refused(self):
        approval = self.approval()
        with mock.patch.object(RUNNER.platform,'system',return_value='Darwin'), mock.patch.object(RUNNER.subprocess,'run') as run:
            with self.assertRaisesRegex(RUNNER.PlanError,'qualified GPU worker'):
                RUNNER.execute(7,True,approval)
        run.assert_not_called()

    def test_schema_matches_the_validator_and_a_valid_manifest(self):
        shown = RUNNER.schema()
        self.assertEqual(set(shown['rootFields']), set(self.doc))
        self.assertEqual(set(shown['jobFields']), set(self.doc['jobs'][0]))
        for key, value in shown['fixedValues'].items():
            if key != 'episode':
                self.assertEqual(self.doc[key], value)

    def test_print_requests_shows_placeholder_payloads_offline(self):
        real_root = Path(__file__).resolve().parents[1]
        with mock.patch.object(RUNNER, 'ROOT', real_root):
            result = RUNNER.execute(7, False, show_requests=True)
        self.assertEqual(result['networkCalls'], 0)
        [shown] = result['requests']
        self.assertEqual(shown['endpoint'], 'jobs/createTask')
        body = shown['body']['input']
        self.assertEqual(body['prompt'], 'Exact Hindi: नमस्ते।')
        self.assertTrue(all(url.startswith('https://placeholder.invalid/') for url in body['image_urls']))
        self.assertFalse((self.work/'motion/demo-s1001.pending.json').exists())
        self.assertFalse((self.work/'motion').exists())

    def test_landed_charges_stop_only_past_the_ceiling_and_rates_need_receipts(self):
        plan = RUNNER.load_plan(self.manifest, 7)
        within = {'demo-s1001': {'credits': 35, 'creditsSource': 'receipt', 'durationSeconds': 7.0},
                  'demo-s1002': {'credits': 10, 'creditsSource': 'wallet-delta', 'durationSeconds': 7.0}}
        plan['maximumCredits'] = 45
        report = RUNNER.check_charges(plan, within)
        self.assertEqual(report['status'], 'WITHIN_CEILING')
        receipted, delta = report['jobs']
        self.assertEqual(receipted['ratioToDeclared'], 1.0)
        self.assertNotIn('ratioToDeclared', delta)
        plan['maximumCredits'] = 44
        with self.assertRaisesRegex(RUNNER.PlanError, 'exceed maximumCredits'):
            RUNNER.check_charges(plan, within)
        saved = json.loads((self.work/'price-variance.json').read_text())
        self.assertEqual((saved['status'], saved['cumulativeLandedCredits']), ('REFUSED', 45.0))
        plan['maximumCredits'] = 1000
        pricey = {'demo-s1001': {'credits': 70, 'creditsSource': 'receipt', 'durationSeconds': 7.0}}
        report = RUNNER.check_charges(plan, pricey)
        self.assertEqual(report['status'], 'WITHIN_CEILING')
        self.assertTrue(report['jobs'][0]['aboveReportTolerance'])

    def test_dry_run_estimates_usd_only_from_the_profile(self):
        with mock.patch.dict(os.environ):
            os.environ.pop('AGENTIC_PROFILE', None)
            self.assertIsNone(RUNNER.execute(7, False)['estimatedUsd'])
        profile = self.root/'profile.json'
        profile.write_text(json.dumps({'providers': {'kie': {'usdPerCredit': 0.01}}}))
        with mock.patch.dict(os.environ, {'AGENTIC_PROFILE': str(profile)}):
            result = RUNNER.execute(7, False)
        self.assertEqual((result['estimatedCredits'], result['estimatedUsd']), (35, 0.35))

    def test_reconcile_only_polls_recorded_tasks_without_approval(self):
        row_ids = []
        def fake_run(command, **kwargs):
            if command[0] == 'nvidia-smi':
                return mock.Mock(returncode=0)
            row_ids.append(command)
            self.assertEqual(kwargs['env']['AGENTIC_UPLOAD_LOG'], str(self.ep/'work/upload-log.jsonl'))
            return mock.Mock(returncode=0)
        landed = {'id': 'demo-s1001', 'sha256': 'c' * 64, 'taskId': 'recorded-task'}
        with mock.patch.object(RUNNER.platform,'system',return_value='Linux'), \
             mock.patch.object(RUNNER.subprocess,'run',side_effect=fake_run), \
             mock.patch.object(RUNNER,'validate_completed',return_value=landed), \
             mock.patch.object(RUNNER,'validate_pending'):
            nothing = RUNNER.reconcile(7)
            self.assertEqual((nothing['unsubmitted'], nothing['reconciled']), (['demo-s1001'], []))
            self.assertEqual(row_ids, [])
            pending = self.work/'motion/demo-s1001.pending.json'
            pending.parent.mkdir(parents=True, exist_ok=True)
            pending.write_text('{}')
            result = RUNNER.reconcile(7)
        self.assertEqual(result['generationPosts'], 0)
        self.assertEqual(result['reconciled'], ['demo-s1001'])
        self.assertEqual(len(row_ids), 1)
        self.assertIn('--resume-only', row_ids[0])
        self.assertFalse((self.work/'execution-authorization.json').exists())
        self.assertFalse(RUNNER.LOCK.exists())

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
