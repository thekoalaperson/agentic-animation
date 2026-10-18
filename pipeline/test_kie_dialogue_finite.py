"""Offline safety tests; no model calls or media processing."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


from pipeline.test_paid_approval import grant_retries, write_approval

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

    def test_version_two_request_and_record_digests_are_pinned(self):
        # Recorded pending records and sidecars bind these digests; a v2
        # manifest must keep producing them byte for byte.
        plan = RUNNER.load_plan(self.manifest, 7)
        [row] = plan['jobs']
        self.assertEqual(plan['manifestSha256'],
                         '6519700e5780d078c56451ef77709c0374d6f4f52ff856641a1f60a90066af63')
        self.assertEqual(row['executionRequestSha256'],
                         'de2e8a89dbae770913ea8e94d0c737ef7f8f272af0aa590063aa90b04963afa0')
        self.assertEqual(RUNNER.canonical_sha256(RUNNER.expected_record(plan, row)),
                         '5cc7d4950ae8e227963991b9b4c4f96793f99a4e8e9604dac57fd2f453ed1f7e')
        command = RUNNER.kie_video_command(plan, row)
        self.assertEqual(command[6:12], ['--res', '720p', '--dur', '7', '--ar', '16:9'])
        self.assertNotIn('--ref', command)

    def version_three(self, references=2, **header):
        refs = []
        for number in range(references):
            ref = self.ep / f'art/refs/demo-ref-{number}.png'
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_bytes(f'offline reference {number}'.encode())
            refs.append(dict(path=str(ref.relative_to(self.ep)), sha256=RUNNER.sha256(ref)))
        self.doc.update(version=3, **header)
        self.doc['pricing'].update(model=self.doc['model'], resolution=self.doc['resolution'])
        if refs:
            self.doc['jobs'][0]['references'] = refs
        self.save()
        return refs

    def test_version_three_sends_frozen_references_after_the_first_frame(self):
        refs = self.version_three()
        plan = RUNNER.load_plan(self.manifest, 7)
        [row] = plan['jobs']
        self.assertEqual(row['referenceSha256s'], [ref['sha256'] for ref in refs])
        request = RUNNER.execution_request(plan, row)
        self.assertEqual((request['version'], request['referenceSha256s']),
                         (2, row['referenceSha256s']))
        self.assertEqual(RUNNER.record_version(row), 3)
        command = RUNNER.kie_video_command(plan, row)
        self.assertEqual(command[-4:], ['--ref', str(self.ep / refs[0]['path']),
                                        '--ref', str(self.ep / refs[1]['path'])])
        dry = RUNNER.execute(7, False)
        self.assertTrue(any('unqualified' in warning for warning in dry['warnings']))

    def test_version_three_without_references_keeps_the_version_one_request(self):
        self.version_three(references=0)
        plan = RUNNER.load_plan(self.manifest, 7)
        request = RUNNER.execution_request(plan, plan['jobs'][0])
        self.assertEqual(request['version'], 1)
        self.assertNotIn('referenceSha256s', request)
        self.assertNotIn('warnings', RUNNER.execute(7, False))

    def test_version_two_never_accepts_references_or_other_values(self):
        self.doc['jobs'][0]['references'] = [dict(path='art/stills/demo-s1001.png',
                                                  sha256=RUNNER.sha256(self.source))]
        self.save()
        with self.assertRaisesRegex(RUNNER.PlanError, 'fields must be exactly'):
            RUNNER.load_plan(self.manifest, 7)
        del self.doc['jobs'][0]['references']
        self.doc['resolution'] = '1080p'
        self.save()
        with self.assertRaisesRegex(RUNNER.PlanError, "resolution must be '720p'"):
            RUNNER.load_plan(self.manifest, 7)

    def test_version_three_values_and_price_are_bound_together(self):
        refs = self.version_three(resolution='1080p')
        plan = RUNNER.load_plan(self.manifest, 7)
        self.assertEqual(RUNNER.kie_video_command(plan, plan['jobs'][0])[6:8], ['--res', '1080p'])
        warnings = RUNNER.execute(7, False)['warnings']
        self.assertTrue(any('only one image is supported at 1080p' in w for w in warnings))
        self.doc['pricing']['resolution'] = '720p'
        self.save()
        with self.assertRaisesRegex(RUNNER.PlanError, 'different model or resolution'):
            RUNNER.load_plan(self.manifest, 7)
        del self.doc['pricing']['resolution']
        self.save()
        with self.assertRaisesRegex(RUNNER.PlanError, 'model and resolution it was read for'):
            RUNNER.load_plan(self.manifest, 7)
        self.doc['pricing']['resolution'] = '1080p'
        for key, value, message in (('resolution', '4k', 'resolution must be one of'),
                                    ('aspectRatio', 'auto', 'aspectRatio must be one of'),
                                    ('model', 'demo-model', 'model must be one of')):
            with self.subTest(key=key):
                saved = self.doc[key]
                self.doc[key] = value
                self.save()
                with self.assertRaisesRegex(RUNNER.PlanError, message):
                    RUNNER.load_plan(self.manifest, 7)
                self.doc[key] = saved
        self.doc['jobs'][0]['references'] = refs * 4
        self.save()
        with self.assertRaisesRegex(RUNNER.PlanError, 'at most 6 references'):
            RUNNER.load_plan(self.manifest, 7)
        self.doc['jobs'][0]['references'] = refs
        self.save()
        (self.ep / refs[1]['path']).write_bytes(b'changed reference')
        with self.assertRaisesRegex(RUNNER.PlanError, r'references\[1\] missing or hash changed'):
            RUNNER.load_plan(self.manifest, 7)

    def test_landed_geometry_follows_the_frozen_values(self):
        self.assertEqual(RUNNER.minimum_geometry('720p', '16:9'), (1200.0, 675.0))
        self.assertEqual(RUNNER.minimum_geometry('1080p', '9:16'), (1012.5, 1800.0))
        self.assertEqual(RUNNER.minimum_geometry('480p', '1:1'), (450.0, 450.0))

    def add_jobs(self, count):
        for number in range(2, count + 1):
            job_id = f'demo-s100{number}'
            prompt = self.work / f'prompts/{job_id}.txt'
            prompt.write_text(f'Exact Hindi: नमस्ते {number}।', encoding='utf-8')
            self.doc['jobs'].append(dict(self.doc['jobs'][0], id=job_id, hindi=f'नमस्ते {number}।',
                prompt=str(prompt.relative_to(self.ep)), promptSha256=RUNNER.sha256(prompt),
                outputStem=str((self.work/f'motion/{job_id}').relative_to(self.ep))))
        self.doc.update(jobCount=count, maximumGeneratedSeconds=7 * count, maximumCredits=35 * count)
        self.save()
        return 7 * count

    def run_wave(self, outcomes, seconds, concurrency=1, retry_failed=False, retries=None,
                 real_pending=False):
        """Submit with kie_video replaced: 'land', 'fail' (definite) or 'lost' per job id.

        A dict outcome is written as the job's pending record; with
        real_pending the runner's own validate_pending reads it.
        """
        invoked = []
        def fake_run(command, **kwargs):
            if command[0] == 'nvidia-smi':
                return mock.Mock(returncode=0)
            job_id = command[command.index('--job-id') + 1]
            invoked.append(job_id)
            stem = self.work / 'motion' / job_id
            stem.parent.mkdir(parents=True, exist_ok=True)
            outcome = outcomes[job_id].pop(0) if isinstance(outcomes[job_id], list) else outcomes[job_id]
            if outcome == 'land':
                stem.with_suffix('.mp4').write_bytes(b'offline')
                stem.with_suffix('.vid.json').write_text('{}')
                return mock.Mock(returncode=0)
            if isinstance(outcome, dict):
                stem.with_suffix('.pending.json').write_text(json.dumps(outcome))
            if outcome == 'fail':
                stem.with_suffix('.pending.json').write_text(json.dumps({
                    'status': 'GENERATION_FAILED', 'taskId': f'task-{job_id}-{len(invoked)}',
                    'failCode': 'demo-code', 'failMsg': 'demo refusal', 'credits': 2}))
            return mock.Mock(returncode=1)
        landed = lambda plan, row: {'id': row['id'], 'durationSeconds': 7.0, 'credits': None,
                                    'creditsSource': None, 'taskId': f'task-{row["id"]}'}
        fake_common = types.SimpleNamespace(kie=lambda route: 10_000)
        approval = self.approval_for(seconds)
        if retries:
            grant_retries(approval, **retries)
        pending = (mock.patch.object(RUNNER, 'validate_pending', wraps=RUNNER.validate_pending)
                   if real_pending else
                   mock.patch.object(RUNNER, 'validate_pending',
                                     side_effect=lambda plan, row, path, allowed=None: json.loads(path.read_text())))
        with mock.patch.object(RUNNER.platform, 'system', return_value='Linux'), \
             mock.patch.object(RUNNER.subprocess, 'run', side_effect=fake_run), \
             mock.patch.object(RUNNER, 'validate_completed', side_effect=landed), \
             pending, mock.patch.dict(sys.modules, {'_common': fake_common}):
            try:
                return RUNNER.execute(seconds, True, approval, concurrency,
                                      retry_failed=retry_failed), invoked
            except RUNNER.PlanError as exc:
                return exc, invoked

    def approval_for(self, seconds):
        plan = RUNNER.load_plan(self.manifest, seconds)
        path = self.ep/'story/approvals/dialogue.json'
        path.unlink(missing_ok=True)
        return write_approval(path, episode='DEMO', lane=RUNNER.APPROVAL_LANE, manifest=self.manifest,
                              scope=RUNNER.approval_scope(plan), project_root=self.ep)

    def refusals(self):
        log = self.work / 'refusals.jsonl'
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def test_a_definite_failure_is_skipped_logged_and_never_reposted(self):
        seconds = self.add_jobs(3)
        outcomes = {'demo-s1001': 'land', 'demo-s1002': 'fail', 'demo-s1003': 'land'}
        result, invoked = self.run_wave(outcomes, seconds)
        self.assertEqual((result['status'], result['failed'], result['items']),
                         ('PARTIAL', ['demo-s1002'], 2))
        self.assertEqual(invoked, ['demo-s1001', 'demo-s1002', 'demo-s1003'])
        audit = json.loads((self.work/'technical-audit.json').read_text())
        self.assertEqual(audit['status'], 'partial')
        self.assertEqual([f['id'] for f in audit['definiteFailures']], ['demo-s1002'])
        [logged] = self.refusals()
        self.assertEqual((logged['jobId'], logged['taskId'], logged['failCode'], logged['operatorResponse']),
                         ('demo-s1002', 'task-demo-s1002-2', 'demo-code', None))
        result, invoked = self.run_wave(outcomes, seconds)
        self.assertEqual((result['status'], invoked), ('PARTIAL', []))
        self.assertEqual(len(self.refusals()), 1)

    def test_consecutive_definite_failures_stop_the_wave(self):
        seconds = self.add_jobs(4)
        result, invoked = self.run_wave({f'demo-s100{n}': 'fail' for n in range(1, 5)}, seconds)
        self.assertIsInstance(result, RUNNER.PlanError)
        self.assertIn('stopped after definite provider failures', str(result))
        self.assertEqual(invoked, ['demo-s1001', 'demo-s1002', 'demo-s1003'])
        self.assertEqual(len(self.refusals()), 3)
        self.assertFalse(RUNNER.LOCK.exists())

    def test_a_whole_failed_concurrency_group_stops_the_wave(self):
        seconds = self.add_jobs(4)
        outcomes = {'demo-s1001': 'fail', 'demo-s1002': 'fail', 'demo-s1003': 'land', 'demo-s1004': 'land'}
        result, invoked = self.run_wave(outcomes, seconds, concurrency=2)
        self.assertIsInstance(result, RUNNER.PlanError)
        self.assertEqual(sorted(invoked), ['demo-s1001', 'demo-s1002'])

    def test_any_other_failure_still_halts(self):
        seconds = self.add_jobs(2)
        result, invoked = self.run_wave({'demo-s1001': 'lost', 'demo-s1002': 'land'}, seconds)
        self.assertIsInstance(result, RUNNER.PlanError)
        self.assertIn('no recovery state was recorded', str(result))
        self.assertEqual(invoked, ['demo-s1001'])
        self.assertEqual(self.refusals(), [])

    def test_failed_task_charges_count_toward_the_ceiling(self):
        plan = RUNNER.load_plan(self.manifest, 7)
        plan['maximumCredits'] = 10
        with self.assertRaisesRegex(RUNNER.PlanError, 'exceed maximumCredits'):
            RUNNER.check_charges(plan, {}, {'demo-s1001': {'creditsConsumed': 11}})

    def test_reconcile_only_reports_definite_failures_without_polling_them(self):
        pending = self.work/'motion/demo-s1001.pending.json'
        pending.parent.mkdir(parents=True)
        pending.write_text(json.dumps({'status': 'GENERATION_FAILED', 'taskId': 'task-failed',
                                       'lastProviderResponse': {'failCode': 'demo-code',
                                                                'failMsg': 'demo refusal',
                                                                'creditsConsumed': 0}}))
        with mock.patch.object(RUNNER.platform,'system',return_value='Linux'), \
             mock.patch.object(RUNNER.subprocess,'run',return_value=mock.Mock(returncode=0)) as run, \
             mock.patch.object(RUNNER,'validate_pending',
                               side_effect=lambda plan, row, path, allowed=None: json.loads(path.read_text())):
            result = RUNNER.reconcile(7)
        self.assertEqual(result['definiteFailures'], ['demo-s1001'])
        self.assertEqual(run.call_count, 1)  # the GPU check only
        [logged] = self.refusals()
        self.assertEqual((logged['failCode'], logged['creditsConsumed']), ('demo-code', 0))

    def test_retry_failed_needs_a_retry_grant(self):
        seconds = self.add_jobs(2)
        self.run_wave({'demo-s1001': 'land', 'demo-s1002': 'fail'}, seconds)
        result, invoked = self.run_wave({'demo-s1002': 'land'}, seconds, retry_failed=True)
        self.assertIsInstance(result, RUNNER.PlanError)
        self.assertIn('version 3 approval record', str(result))
        self.assertEqual(invoked, [])
        self.assertTrue((self.work/'motion/demo-s1002.pending.json').exists())

    def test_retry_failed_reposts_within_the_granted_budget_and_keeps_the_history(self):
        seconds = self.add_jobs(2)
        grant = {'jobs': 1, 'credits': 35}
        self.run_wave({'demo-s1001': 'land', 'demo-s1002': 'fail'}, seconds, retries=grant)
        # Without --retry-failed the failure stays skipped.
        result, invoked = self.run_wave({}, seconds, retries=grant)
        self.assertEqual((result['status'], invoked), ('PARTIAL', []))
        result, invoked = self.run_wave({'demo-s1002': ['fail']}, seconds, retry_failed=True,
                                        retries=grant)
        self.assertEqual((result['status'], invoked), ('PARTIAL', ['demo-s1002']))
        archived = self.work/'motion/demo-s1002.attempts/attempt-001.pending.json'
        self.assertEqual(json.loads(archived.read_text())['taskId'], 'task-demo-s1002-2')
        self.assertEqual(len(self.refusals()), 2)
        # The budget allowed one retry; a second is refused before anything moves.
        result, invoked = self.run_wave({'demo-s1002': ['land']}, seconds, retry_failed=True,
                                        retries=grant)
        self.assertIsInstance(result, RUNNER.PlanError)
        self.assertIn('retry budget is 1', str(result))
        self.assertEqual(invoked, [])
        self.assertTrue((self.work/'motion/demo-s1002.pending.json').exists())
        variance = json.loads((self.work/'price-variance.json').read_text())
        self.assertEqual((variance['cumulativeLandedCredits'], variance['retryCredits']), (2.0, 2.0))

    def test_a_retry_grant_added_after_the_first_submission_names_the_route(self):
        seconds = self.add_jobs(2)
        self.run_wave({'demo-s1001': 'land', 'demo-s1002': 'fail'}, seconds)
        result, invoked = self.run_wave({'demo-s1002': ['land']}, seconds, retry_failed=True,
                                        retries={'jobs': 1, 'credits': 10})
        self.assertIsInstance(result, RUNNER.PlanError)
        self.assertIn('before a batch\'s first submission', str(result))
        self.assertEqual(invoked, [])

    def test_paid_approval_check_accepts_a_dialogue_retry_grant(self):
        from pipeline import paid_approval
        record = grant_retries(self.approval(), jobs=1, credits=5)
        checked = paid_approval.check_record(record, self.manifest, RUNNER.APPROVAL_LANE, self.ep)
        self.assertEqual((checked['schemaVersion'], checked['retryBudget']), (3, {'jobs': 1, 'credits': 5}))

    def test_retry_credits_budget_is_read_before_anything_is_posted(self):
        seconds = self.add_jobs(2)
        # One 7 s retry at 5 credits a second is priced at 35, past a 10 credit grant.
        grant = {'jobs': 5, 'credits': 10}
        self.run_wave({'demo-s1001': 'land', 'demo-s1002': 'fail'}, seconds, retries=grant)
        result, invoked = self.run_wave({'demo-s1002': ['land']}, seconds, retry_failed=True,
                                        retries=grant)
        self.assertIsInstance(result, RUNNER.PlanError)
        self.assertIn('retry credits budget is 10', str(result))
        self.assertEqual(invoked, [])
        self.assertTrue((self.work/'motion/demo-s1002.pending.json').exists())
        self.assertFalse((self.work/'motion/demo-s1002.attempts').exists())

    def test_retried_charges_are_held_to_the_retry_budget_after_they_land(self):
        plan = RUNNER.load_plan(self.manifest, 7)
        history = {'demo-s1001': [{'creditsConsumed': 2}, {'creditsConsumed': 3}]}
        self.assertEqual(RUNNER.retry_charged({}, {'demo-s1001': {'creditsConsumed': 4}}, history), 7)
        report = RUNNER.check_charges(plan, {}, {'demo-s1001': {'creditsConsumed': 4}}, history, 7)
        self.assertEqual((report['cumulativeLandedCredits'], report['retryCredits']), (2.0, 7.0))
        with self.assertRaisesRegex(RUNNER.PlanError, 'past the approval'):
            RUNNER.check_charges(plan, {}, {'demo-s1001': {'creditsConsumed': 4}}, history, 6)

    def test_a_retry_moved_but_never_posted_needs_retry_failed(self):
        seconds = self.add_jobs(3)
        grant = {'jobs': 2, 'credits': 70}
        self.run_wave({'demo-s1001': 'land', 'demo-s1002': 'fail', 'demo-s1003': 'fail'}, seconds,
                      retries=grant)
        # Both records move into their attempts; the first retry is lost, so the
        # second is never posted.
        result, invoked = self.run_wave({'demo-s1002': ['lost'], 'demo-s1003': ['land']}, seconds,
                                        retry_failed=True, retries=grant)
        self.assertIsInstance(result, RUNNER.PlanError)
        self.assertEqual(invoked, ['demo-s1002'])
        self.assertTrue((self.work/'motion/demo-s1003.attempts/attempt-001.pending.json').exists())
        self.assertFalse((self.work/'motion/demo-s1003.pending.json').exists())
        result, invoked = self.run_wave({'demo-s1002': 'land', 'demo-s1003': 'land'}, seconds,
                                        retries=grant)
        self.assertIsInstance(result, RUNNER.PlanError)
        self.assertIn("['demo-s1002', 'demo-s1003'] were moved into their attempts history", str(result))
        self.assertEqual(invoked, [])
        with mock.patch.object(RUNNER.platform, 'system', return_value='Linux'), \
             mock.patch.object(RUNNER.subprocess, 'run', return_value=mock.Mock(returncode=0)), \
             mock.patch.object(RUNNER, 'validate_completed', return_value={'id': 'demo-s1001'}):
            reconciled = RUNNER.reconcile(seconds)
        self.assertEqual((reconciled['unpostedRetries'], reconciled['unsubmitted']),
                         (['demo-s1002', 'demo-s1003'], []))
        # The moves already counted toward the jobs budget; posting them spends no more of it.
        result, invoked = self.run_wave({'demo-s1002': 'land', 'demo-s1003': 'land'}, seconds,
                                        retry_failed=True, retries=grant)
        self.assertEqual((result['status'], invoked), ('COMPLETED', ['demo-s1002', 'demo-s1003']))

    def failed_record(self, job_id, seconds, **changes):
        """A GENERATION_FAILED record shaped as kie_video writes it for this frozen job."""
        plan = RUNNER.load_plan(self.manifest, seconds)
        row = next(row for row in plan['jobs'] if row['id'] == job_id)
        expected = RUNNER.expected_record(plan, row)
        bound = {key: expected[key] for key in ('model', 'sourceSha256', 'promptSha256',
                                                'executionManifestSha256', 'executionRequestSha256')}
        local = {**bound, 'jobId': job_id, 'resolution': expected['resolution'],
                 'durationSeconds': expected['durationSeconds'], 'aspectRatio': expected['aspectRatio'],
                 'audio': True, 'inputHashes': {'primary': row['sourceSha256'], 'references': [],
                                                'last': None, 'videos': []}}
        request = {**bound, 'input': {'prompt': expected['promptText'], 'resolution': expected['resolution'],
                                      'aspect_ratio': expected['aspectRatio'],
                                      'duration': expected['durationSeconds']}}
        record = {**bound, 'provenanceVersion': RUNNER.record_version(row), 'jobId': job_id,
                  'localRequest': local, 'localRequestSha256': RUNNER.canonical_sha256(local),
                  'request': request, 'requestFingerprint': RUNNER.canonical_sha256(request),
                  'requestSha256': RUNNER.canonical_sha256(request),
                  'status': 'GENERATION_FAILED', 'taskId': f'task-{job_id}',
                  'failCode': 'demo-code', 'failMsg': 'demo refusal', 'credits': 0}
        if 'promptText' in changes:
            request['input']['prompt'] = changes.pop('promptText')
            record['requestFingerprint'] = record['requestSha256'] = RUNNER.canonical_sha256(request)
        record.update(changes)
        return record

    def test_a_failed_record_off_the_frozen_job_halts_instead_of_being_skipped(self):
        seconds = self.add_jobs(2)
        good = self.failed_record('demo-s1001', seconds)
        result, invoked = self.run_wave({'demo-s1001': good, 'demo-s1002': 'land'}, seconds,
                                        real_pending=True)
        self.assertEqual((result['status'], result['failed'], invoked),
                         ('PARTIAL', ['demo-s1001'], ['demo-s1001', 'demo-s1002']))
        self.assertEqual(len(self.refusals()), 1)
        result, invoked = self.run_wave({}, seconds, real_pending=True)
        self.assertEqual((result['status'], invoked), ('PARTIAL', []))
        tampered = {'promptSha256': {'promptSha256': '0' * 64},
                    'empty taskId': {'taskId': ''},
                    'request prompt': {'promptText': 'a different line'}}
        for label, change in tampered.items():
            with self.subTest(label):
                (self.work/'refusals.jsonl').unlink(missing_ok=True)
                for job_id in ('demo-s1001', 'demo-s1002'):
                    for suffix in ('.pending.json', '.mp4', '.vid.json'):
                        (self.work/f'motion/{job_id}{suffix}').unlink(missing_ok=True)
                bad = self.failed_record('demo-s1001', seconds, **change)
                result, invoked = self.run_wave({'demo-s1001': bad, 'demo-s1002': 'land'}, seconds,
                                                real_pending=True)
                self.assertIsInstance(result, RUNNER.PlanError)
                self.assertEqual(invoked, ['demo-s1001'])
                self.assertEqual(self.refusals(), [])

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

    def test_batch_started_on_the_boolean_names_its_recorded_runner(self):
        approval = self.approval()
        (self.work/'execution-authorization.json').write_text(json.dumps({
            'executionManifestSha256': RUNNER.load_plan(self.manifest, 7)['manifestSha256'],
            'approval': None}))
        with mock.patch.object(RUNNER.platform,'system',return_value='Linux'), \
             mock.patch.object(RUNNER.subprocess,'run',return_value=mock.Mock(returncode=0)):
            with self.assertRaisesRegex(RUNNER.PlanError,
                                        '--reconcile-only.*runner that started it'):
                RUNNER.execute(7,True,approval)
        self.assertEqual(RUNNER.attempt_records(self.work, 'credit-preflight'), [])
        self.assertFalse(RUNNER.LOCK.exists())

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

    def test_geometry_warnings_leave_every_serialized_record_unchanged(self):
        from PIL import Image
        Image.new('RGB', (64, 64), 'gray').save(self.source, format='PNG')
        self.doc['jobs'][0]['sourceSha256'] = RUNNER.sha256(self.source)
        self.save()
        warned = RUNNER.load_plan(RUNNER.MANIFEST, 7)
        with mock.patch.object(RUNNER, 'source_geometry_warnings', return_value=[]):
            quiet = RUNNER.load_plan(RUNNER.MANIFEST, 7)
        self.assertEqual([w['jobId'] for w in warned['geometryWarnings']], ['demo-s1001'])
        self.assertEqual(quiet['geometryWarnings'], [])

        def serialized(plan):
            return json.dumps({
                'requests': [RUNNER.execution_request(plan, row) for row in plan['jobs']],
                'requestHashes': [row['executionRequestSha256'] for row in plan['jobs']],
                'expected': [RUNNER.expected_record(plan, row) for row in plan['jobs']],
                'commands': [RUNNER.kie_video_command(plan, row) for row in plan['jobs']],
                'finiteExecution': RUNNER.approval_scope(plan),
                'creditPreflight': RUNNER.credit_preflight(plan, 100.0, 35, 1),
            }, sort_keys=True)
        self.assertEqual(serialized(warned), serialized(quiet))

        with mock.patch.object(RUNNER.subprocess, 'run'):
            loud = RUNNER.execute(7, False)
            with mock.patch.object(RUNNER, 'source_geometry_warnings', return_value=[]):
                silent = RUNNER.execute(7, False)
        self.assertEqual({k: v for k, v in loud.items() if k != 'geometryWarnings'},
                         {k: v for k, v in silent.items() if k != 'geometryWarnings'})


if __name__ == '__main__':
    unittest.main()
