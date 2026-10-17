import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common
import render_timed_animatic as animatic
from render_timed_animatic import validate

FFMPEG = shutil.which('ffmpeg') and shutil.which('ffprobe')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AnimaticContractTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'kind': 'timed-still-animatic-not-final', 'finalApproved': False, 'fps': 24,
                     'scenes': [{'id': 'demo-s001', 'mode': 'narration', 'frames': 240,
                                 'text': 'एक पूरा परीक्षण वाक्य', 'sourceStatus': 'candidate',
                                 'audio': {'durationSeconds': 9.5}}]}

    def test_whole_speech(self):
        validate(self.plan)

    def test_reject_short_coverage(self):
        self.plan['scenes'][0]['frames'] = 220
        with self.assertRaises(ValueError): validate(self.plan)

    def test_account_for_lead(self):
        self.plan['scenes'][0]['audioLeadFrames'] = 20
        with self.assertRaises(ValueError): validate(self.plan)

    def test_scratch_is_silent(self):
        self.plan['scenes'][0]['mode'] = 'scratch-dialogue'
        with self.assertRaises(ValueError): validate(self.plan)
        self.plan['scenes'][0].pop('audio')
        validate(self.plan)

    def test_never_final(self):
        self.plan['finalApproved'] = True
        with self.assertRaises(ValueError): validate(self.plan)

    def test_unique_ids(self):
        self.plan['scenes'].append(copy.deepcopy(self.plan['scenes'][0]))
        with self.assertRaises(ValueError): validate(self.plan)

    def test_finite_clock(self):
        self.plan['scenes'][0]['audio']['durationSeconds'] = float('nan')
        with self.assertRaises(ValueError): validate(self.plan)

    def test_honest_source_label(self):
        self.plan['scenes'][0]['sourceStatus'] = ''
        with self.assertRaises(ValueError): validate(self.plan)

    def test_contiguous_scene_clock(self):
        self.plan['scenes'][0]['startFrame'] = 1
        with self.assertRaises(ValueError): validate(self.plan)

    def test_total_clock(self):
        self.plan['frames'] = 241
        with self.assertRaises(ValueError): validate(self.plan)



class HoldAndTempScoreTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'kind': 'timed-still-animatic-not-final', 'finalApproved': False, 'fps': 24,
                     'scenes': [{'id': 'demo-s001', 'mode': 'hold', 'frames': 60, 'text': '',
                                 'sourceStatus': 'candidate'}]}

    def test_hold_needs_no_words(self):
        validate(self.plan)
        del self.plan['scenes'][0]['text']
        validate(self.plan)

    def test_hold_still_needs_an_honest_status(self):
        self.plan['scenes'][0]['sourceStatus'] = ''
        with self.assertRaises(ValueError): validate(self.plan)

    def test_hold_temporary_audio_must_fit_whole(self):
        self.plan['scenes'][0]['audio'] = {'durationSeconds': 2.4}
        validate(self.plan)
        self.plan['scenes'][0]['audio'] = {'durationSeconds': 2.6}
        with self.assertRaises(ValueError): validate(self.plan)

    def test_narration_and_dialogue_still_need_words(self):
        for mode in ('narration', 'scratch-dialogue'):
            plan = copy.deepcopy(self.plan)
            plan['scenes'][0].update(mode=mode, audio={'durationSeconds': 1.0})
            if mode == 'scratch-dialogue':
                plan['scenes'][0].pop('audio')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate(plan)

    def test_temp_score_is_bound_and_bounded(self):
        self.plan['tempScore'] = {'path': 'audio/score.wav', 'sha256': 'a' * 64, 'gainDb': -12}
        validate(self.plan)
        for bad in ({'gainDb': 3}, {'gainDb': float('nan')}, {'gainDb': True}, {'sha256': 'x'}, {'path': None}):
            plan = copy.deepcopy(self.plan)
            plan['tempScore'].update(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate(plan)

    def test_preview_names_are_labelled(self):
        self.assertEqual(animatic.labelled('timed-animatic-v1.mp4', True), 'timed-animatic-v1-preview-not-final.mp4')
        self.assertEqual(animatic.labelled('segments', True), 'segments-preview-not-final')
        self.assertEqual(animatic.labelled('render-audit.json', False), 'render-audit.json')


class StoryboardPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ep = Path(self.temp.name).resolve()
        records = []
        for key in ('demo-s001', 'demo-s002', 'demo-s003'):
            still = self.ep / 'art/stills' / f'{key}.png'
            still.parent.mkdir(parents=True, exist_ok=True)
            still.write_bytes(b'demo still ' + key.encode())
            sidecar = still.with_suffix('.still.json')
            sidecar.write_text(json.dumps({'target_sha256': digest(still)}))
            records.append({'still_id': key, 'target_output_path': f'art/stills/{key}.png',
                            'provenance_sidecar': f'art/stills/{key}.still.json'})
        self.manifest = {'episode': 'demo', 'stills': records}
        narration = self.ep / 'audio/narration-v1/demo-s001-hi.mp3'
        narration.parent.mkdir(parents=True)
        narration.write_bytes(b'demo narration')
        self.board = {'episode': 'demo', 'stills': [
            {'still_id': 'demo-s001', 'speaker': '', 'narration_hi': 'demo line one'},
            {'still_id': 'demo-s002', 'speaker': '-', 'narration_hi': ''},
            {'still_id': 'demo-s003', 'speaker': 'demo-a', 'narration_hi': 'demo line two'}]}

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, **kwargs):
        return animatic.plan_from_storyboard(self.ep, self.board, self.manifest,
                                             measure=lambda path: 2.01, **kwargs)

    def test_refuses_and_names_every_untimed_row(self):
        with self.assertRaises(ValueError) as caught:
            self.plan()
        message = str(caught.exception)
        self.assertIn('demo-s002: no estimatedSeconds for this hold row', message)
        self.assertIn('demo-s003: no estimatedSeconds for this scratch-dialogue row', message)
        self.assertNotIn('demo-s001', message)

    def test_measured_narration_and_authored_estimates(self):
        self.board['stills'][1]['estimatedSeconds'] = 3.5
        plan = self.plan(estimates={'demo-s003': 4})
        modes = [(s['id'], s['mode'], s['frames'], s['timingBasis']) for s in plan['scenes']]
        self.assertEqual(modes, [('demo-s001', 'narration', 49, 'measured narration'),
                                 ('demo-s002', 'hold', 84, 'estimatedSeconds'),
                                 ('demo-s003', 'scratch-dialogue', 96, 'estimatedSeconds')])
        self.assertEqual(plan['scenes'][0]['audio']['durationSeconds'], 2.01)
        self.assertEqual([s['startFrame'] for s in plan['scenes']], [0, 49, 133])
        self.assertEqual(plan['frames'], 229)
        self.assertNotIn('audio', plan['scenes'][2])

    def test_missing_narration_needs_an_estimate(self):
        (self.ep / 'audio/narration-v1/demo-s001-hi.mp3').unlink()
        with self.assertRaisesRegex(ValueError, 'demo-s001: no measured narration file'):
            self.plan(estimates={'demo-s002': 2, 'demo-s003': 2})
        plan = self.plan(estimates={'demo-s001': 2, 'demo-s002': 2, 'demo-s003': 2})
        self.assertNotIn('audio', plan['scenes'][0])
        self.assertEqual((plan['scenes'][0]['mode'], plan['scenes'][0]['text']), ('hold', 'demo line one'))
        self.assertIn('narration not yet made', plan['scenes'][0]['sourceStatus'])

    def test_conflicting_or_unknown_estimates_are_refused(self):
        self.board['stills'][1]['estimatedSeconds'] = 3
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.plan(estimates={'demo-s002': 4, 'demo-s003': 2})
        with self.assertRaisesRegex(ValueError, 'not on the storyboard'):
            self.plan(estimates={'demo-s099': 2})

    def test_written_plan_is_bound_and_never_overwritten(self):
        (self.ep / 'audio/narration-v1/demo-s001-hi.mp3').unlink()
        estimates = self.ep / 'estimates.json'
        estimates.write_text(json.dumps({'demo-s001': 2, 'demo-s002': 1.5, 'demo-s003': 2}))
        (self.ep / 'story').mkdir()
        (self.ep / 'story/storyboard_plan.json').write_text(json.dumps(self.board))
        (self.ep / 'story/generation_manifest.json').write_text(json.dumps(self.manifest))
        target = self.ep / 'work/animatic-v1/plan.json'
        argv = [str(self.ep), str(target), '--from-storyboard', '--estimates', str(estimates)]
        with mock.patch('sys.stdout'):
            animatic.main(argv)
        plan = json.loads(target.read_text())
        validate(plan)
        self.assertEqual(plan['writtenFrom']['estimatesSha256'], digest(estimates))
        self.assertEqual(plan['writtenFrom']['storyboardSha256'], digest(self.ep / 'story/storyboard_plan.json'))
        with self.assertRaisesRegex(ValueError, 'choose a new version'):
            animatic.main(argv)

    def test_unregistered_or_changed_still_is_refused(self):
        (self.ep / 'art/stills/demo-s003.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'demo-s003: still bytes do not match'):
            self.plan(estimates={'demo-s002': 2, 'demo-s003': 2})


class GateAndTypographyTests(unittest.TestCase):
    def test_worker_render_keeps_its_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            ep = Path(temporary).resolve()
            plan = ep / 'work/anim/plan.json'
            plan.parent.mkdir(parents=True)
            plan.write_text(json.dumps({'kind': 'timed-still-animatic-not-final', 'finalApproved': False, 'fps': 24,
                'scenes': [{'id': 'demo-s001', 'mode': 'hold', 'frames': 24, 'sourceStatus': 'candidate',
                            'still': {'path': 'x.png', 'sha256': 'a' * 64}}]}))
            with mock.patch.object(animatic.platform, 'system', return_value='Darwin'), \
                    self.assertRaisesRegex(RuntimeError, 'qualified GPU host'):
                animatic.main([str(ep), str(plan), '--jobs', '1'])
            with self.assertRaises(SystemExit):
                animatic.main([str(ep), str(plan), '--jobs', '1', '--no-cards'])

    def test_complex_script_without_a_shaper_is_refused_with_a_way_out(self):
        row = {'id': 'demo-s001', 'mode': 'narration', 'sourceStatus': 'candidate',
               'text': '\u0915\u094d\u0937'}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, '--no-cards'):
                animatic._plain_raster(row, {}, Path(temporary) / 'type.png')

    def test_complex_script_detection(self):
        self.assertTrue(_common.needs_complex_shaping('demo \u0915'))
        self.assertFalse(_common.needs_complex_shaping('demo line, caf\u00e9'))
        self.assertFalse(_common.needs_complex_shaping(''))


@unittest.skipUnless(FFMPEG, 'ffmpeg is not installed')
class PreviewRenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ep = Path(self.temp.name).resolve()
        (self.ep / 'art').mkdir()
        (self.ep / 'audio').mkdir()
        (self.ep / 'work/anim').mkdir(parents=True)
        run = lambda *args: subprocess.run(['ffmpeg', '-v', 'error', *args], check=True, cwd=self.ep)
        run('-f', 'lavfi', '-i', 'color=c=0x334455:s=160x90', '-frames:v', '1', 'art/demo.png')
        run('-f', 'lavfi', '-i', 'sine=f=440:d=1.0', '-ar', '48000', 'audio/line.wav')
        run('-f', 'lavfi', '-i', 'sine=f=220:d=4', '-ar', '48000', 'audio/score.wav')
        still = {'path': 'art/demo.png', 'sha256': digest(self.ep / 'art/demo.png')}
        self.plan = {'kind': 'timed-still-animatic-not-final', 'finalApproved': False, 'fps': 24, 'scenes': [
            {'id': 'demo-s001', 'mode': 'narration', 'frames': 30, 'text': 'demo line', 'sourceStatus': 'candidate',
             'still': still, 'audio': {'path': 'audio/line.wav', 'sha256': digest(self.ep / 'audio/line.wav'),
                                       'durationSeconds': 1.0}},
            {'id': 'demo-s002', 'mode': 'hold', 'frames': 36, 'sourceStatus': 'candidate', 'still': still}],
            'tempScore': {'path': 'audio/score.wav', 'sha256': digest(self.ep / 'audio/score.wav'), 'gainDb': -18}}
        self.path = self.ep / 'work/anim/plan.json'
        self.path.write_text(json.dumps(self.plan))

    def tearDown(self):
        self.temp.cleanup()

    def render(self, *extra):
        with mock.patch('sys.stdout'):
            animatic.main([str(self.ep), str(self.path), '--jobs', '2', '--preview', *extra])
        return json.loads((self.ep / 'work/anim/render-audit-preview-not-final.json').read_text())

    def test_preview_without_cards_is_labelled_and_never_final(self):
        audit = self.render('--no-cards')
        root = self.ep / 'work/anim'
        self.assertTrue((root / 'timed-animatic-v1-preview-not-final.mp4').is_file())
        self.assertFalse((root / 'timed-animatic-v1.mp4').exists())
        self.assertFalse((root / 'render-audit.json').exists())
        self.assertFalse((root / 'segments').exists())
        self.assertIs(audit['finalApproved'], False)
        self.assertIs(audit['preview'], True)
        self.assertEqual(audit['frames'], 66)
        self.assertEqual(audit['holdCount'], 1)
        self.assertNotIn('qualified GPU worker; no Mac', audit['encoder'])
        self.assertIn(audit['hostRecord']['computeHost'], audit['encoder'])
        self.assertTrue(audit['hostRecord']['ffmpeg'].startswith('ffmpeg version'))
        self.assertIsNone(audit['hostRecord']['font'])
        self.assertEqual(audit['tempScore']['gainDb'], -18)
        self.assertEqual(audit['tempScore']['mixSha256'],
                         digest(root / 'animatic-mix-with-temp-score-preview-not-final.wav'))
        voice = json.loads((root / 'voice-clock-audit-preview-not-final.json').read_text())
        self.assertEqual([e['id'] for e in voice['events']], ['demo-s001'])

    def test_preview_with_plain_cards(self):
        audit = self.render()
        self.assertEqual(audit['typography'], 'unqualified off the worker')
        cards = sorted((self.ep / 'work/anim/segments-preview-not-final').glob('*.type.json'))
        self.assertEqual(len(cards), 2)
        self.assertIs(json.loads(cards[0].read_text())['visualApproval'], False)


if __name__ == '__main__': unittest.main()
