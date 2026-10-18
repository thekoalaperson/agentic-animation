"""The coverage inventory reads the optional delivery column of a pipe script."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import audit_story_coverage as coverage


class DeliveryColumnTest(unittest.TestCase):
    def test_fifth_column_stays_out_of_the_text_and_picks_the_lane(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'demo-project'
            (root / 'story').mkdir(parents=True)
            script = root / 'story/production-script-v2.tsv'
            script.write_text('1|Narrator|demo hi one|A lamp on a table.\n'
                              '2|-|||wordless\n'
                              '3|Character One|demo hi three|I am here.|unsynced\n'
                              '4|Character One|demo hi four|I am here.\n')
            output = root / 'work/coverage.json'
            argv = ['coverage', str(root), '--output', str(output)]
            with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.print'):
                coverage.main()
            shots = json.loads(output.read_text())['shots']
            self.assertEqual([s['english'] for s in shots],
                             ['A lamp on a table.', '', 'I am here.', 'I am here.'])
            self.assertEqual([s.get('delivery') for s in shots], [None, 'wordless', 'unsynced', None])
            self.assertEqual([s['requiredLane'] for s in shots],
                             ['h3-base', 'h3-base', 'h3-base', 'grok-dialogue'])



def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StillKeyedLayoutTest(unittest.TestCase):
    """Files belong to rows by still ID, found where today's runners put them."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'demo-project'
        (self.root / 'story').mkdir(parents=True)
        (self.root / 'story/script.txt').write_text(
            '1 | Character One | demo hi one | I am here.\n'
            '2 | Character One | demo hi two | Still here.\n'
            '3 | - | | | wordless\n')
        (self.root / 'story/storyboard_plan.json').write_text(json.dumps({'episode': 'demo', 'stills': [
            {'still_id': 'demo-s010'}, {'still_id': 'demo-s010b'}, {'still_id': 'demo-s011'}]}))

    def tearDown(self):
        self.temp.cleanup()

    def take(self, directory, still):
        motion = self.root / directory / 'motion'
        motion.mkdir(parents=True, exist_ok=True)
        video = motion / f'{still}.mp4'
        video.write_bytes(still.encode())
        (motion / f'{still}.vid.json').write_text(json.dumps({'sha256': sha(video), 'durationSeconds': 4}))

    def run_audit(self, *extra):
        output = self.root / 'work/coverage.json'
        argv = ['coverage', str(self.root), '--output', str(output), *extra]
        with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.print'):
            coverage.main()
        return {s['id']: s for s in json.loads(output.read_text())['shots']}

    def test_split_still_ids_and_the_grok_dialogue_layout(self):
        self.take('work/grok-dialogue-v1', 'demo-s010')
        self.take('work/grok-dialogue-v1', 'demo-s010b')
        silent = self.root / 'work/h3-v1/silent'
        silent.mkdir(parents=True)
        (silent / 'demo-s011-cont1.mp4').write_bytes(b'picture')
        (silent / 'ingest-receipt.json').write_text(json.dumps({'items': [
            {'id': 'demo-s011-cont1', 'silent': 'demo-s011-cont1.mp4', 'audioStreams': 0,
             'silentSha256': sha(silent / 'demo-s011-cont1.mp4'), 'frames': 96}]}))
        shots = self.run_audit()
        self.assertEqual([c['path'] for c in shots['demo-s010']['candidates']],
                         ['work/grok-dialogue-v1/motion/demo-s010.mp4'])
        self.assertEqual([c['path'] for c in shots['demo-s010b']['candidates']],
                         ['work/grok-dialogue-v1/motion/demo-s010b.mp4'])
        self.assertFalse(shots['demo-s011']['hasVerifiedBaseCandidate'])  # a continuation is not a base
        self.assertTrue(shots['demo-s010']['hasVerifiedBaseCandidate'])
        self.assertEqual(shots['demo-s011']['candidates'][0]['kind'], 'h3-continuation')
        self.assertEqual(shots['demo-s010']['speaker'], 'Character One')

    def test_declared_execution_manifests_replace_the_layout_search(self):
        self.take('work/grok-dialogue-v1', 'demo-s010')
        self.take('work/pilot-a', 'demo-s010b')
        manifest = self.root / 'work/pilot-a/execution-manifest.json'
        manifest.write_text(json.dumps({'jobs': [
            {'id': 'demo-s010b', 'outputStem': 'work/pilot-a/motion/demo-s010b'},
            {'id': 'demo-s011', 'outputStem': 'work/pilot-a/motion/demo-s011'}]}))
        shots = self.run_audit('--execution-manifest', str(manifest))
        self.assertEqual(shots['demo-s010']['candidates'], [])
        self.assertEqual([c['path'] for c in shots['demo-s010b']['candidates']],
                         ['work/pilot-a/motion/demo-s010b.mp4'])


if __name__ == '__main__':
    unittest.main()
