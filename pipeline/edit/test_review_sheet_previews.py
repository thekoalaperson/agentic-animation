"""Review sheets that run anywhere ffmpeg runs, and the sampling they use."""
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
import prepare_episode_production as production

FFMPEG = shutil.which('ffmpeg') and shutil.which('ffprobe')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SamplingTests(unittest.TestCase):
    def test_samples_span_each_clip_whatever_its_length(self):
        self.assertEqual(production.sample_times(3.0 + 1 / 24), [0.0, 0.6, 1.2, 1.8, 2.4, 3.0])
        long = production.sample_times(12.0)
        self.assertEqual(long[0], 0.0)
        self.assertAlmostEqual(long[-1], 12.0 - 1 / 24, places=3)
        self.assertEqual(len(long), 6)

    def test_unmeasured_duration_is_refused(self):
        for bad in (0, -1, float('nan')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                production.sample_times(bad)

    def test_cut_pairs_follow_story_order(self):
        rows = [{'id': 'demo-s001'}, {'id': 'demo-s002'}, {'id': 'demo-s003'}]
        self.assertEqual([(a['id'], b['id']) for a, b in production.cut_pairs(rows)],
                         [('demo-s001', 'demo-s002'), ('demo-s002', 'demo-s003')])
        self.assertEqual(production.cut_pairs(rows[:1]), [])


class EpisodeFixture:
    def __init__(self, root):
        self.ep = Path(root).resolve()
        stills = []
        board = []
        for index, (key, speaker) in enumerate((('demo-s001', ''), ('demo-s002', 'demo-a'))):
            still = self.ep / 'art/stills' / f'{key}.png'
            still.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'color=c=0x22{index}{index}44:s=160x90',
                            '-frames:v', '1', str(still)], check=True)
            sidecar = still.with_suffix('.still.json')
            sidecar.write_text(json.dumps({'target_sha256': digest(still)}))
            stills.append({'still_id': key, 'target_output_path': f'art/stills/{key}.png',
                           'provenance_sidecar': f'art/stills/{key}.still.json',
                           'staging': {'screen_side': 'left'}})
            board.append({'still_id': key, 'speaker': speaker, 'beat': 1,
                          'narration_hi': 'demo line', 'narration_en': 'demo line'})
        (self.ep / 'story').mkdir()
        (self.ep / 'story/storyboard_plan.json').write_text(json.dumps({'episode': 'demo', 'stills': board}))
        (self.ep / 'story/generation_manifest.json').write_text(json.dumps({'episode': 'demo', 'stills': stills}))
        self.clips = self.ep / 'candidates'
        self.clips.mkdir()
        for key, seconds in (('demo-s001', 2), ('demo-s002', 3)):
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'testsrc=s=160x90:r=24:d={seconds}',
                            '-pix_fmt', 'yuv420p', str(self.clips / f'{key}.mp4')], check=True)


class GateTests(unittest.TestCase):
    def test_preview_needs_sheets(self):
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            production.main(['/nonexistent', '--preview'])

    def test_full_pass_keeps_the_worker_gate(self):
        with mock.patch.object(production.platform, 'system', return_value='Darwin'), \
                mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            production.main(['/nonexistent', '--sheets'])


@unittest.skipUnless(FFMPEG, 'ffmpeg is not installed')
class PreviewSheetTests(unittest.TestCase):
    def test_preview_sheets_are_labelled_and_write_no_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EpisodeFixture(temporary)
            with mock.patch.object(production.platform, 'system', return_value='Darwin'), mock.patch('sys.stdout'):
                production.main([str(fixture.ep), '--sheets', '--preview', '--h3-sources', str(fixture.clips)])
            work = fixture.ep / 'work/production-intake-v1'
            sheets = work / 'sheets-preview-not-final'
            names = sorted(p.name for p in sheets.iterdir())
            self.assertEqual(names, ['cuts-001-001.jpg', 'motion-001-002.jpg', 'sheet-audit.json',
                                     'stills-001-002.jpg'])
            self.assertFalse((work / 'inventory.json').exists())
            self.assertFalse((work / 'silent-h3').exists())
            self.assertFalse((work / 'sheets').exists())
            audit = json.loads((sheets / 'sheet-audit.json').read_text())
            self.assertIs(audit['finalApproved'], False)
            self.assertEqual(audit['kind'], 'review-sheets-preview-not-final')
            self.assertTrue(audit['ffmpeg'].startswith('ffmpeg version'))
            self.assertEqual({s['path'] for s in audit['sheets']},
                             {'cuts-001-001.jpg', 'motion-001-002.jpg', 'stills-001-002.jpg'})
            for sheet in audit['sheets']:
                self.assertEqual(sheet['sha256'], digest(sheets / sheet['path']))

    def test_last_frame_is_the_clip_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EpisodeFixture(temporary)
            source = fixture.clips / 'demo-s002.mp4'
            last = production.frame_at(source, last=True, size=(32, 18))
            first = production.frame_at(source, 0, size=(32, 18))
            self.assertEqual(last.size, (32, 18))
            self.assertNotEqual(last.tobytes(), first.tobytes())


if __name__ == '__main__':
    unittest.main()
