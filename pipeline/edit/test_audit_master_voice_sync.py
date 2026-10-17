import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import audit_master_voice_sync as sync


class VoiceSyncPictureEventTest(unittest.TestCase):
    def test_wordless_picture_events_are_listed_not_correlated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / 'edit-plan.json'
            plan.write_text(json.dumps({'runtimeSeconds': 6.0, 'events': [
                {'id': 'demo-s001', 'kind': 'narration', 'start': 0.0, 'end': 3.0},
                {'id': 'demo-s002', 'kind': 'picture', 'start': 3.0, 'end': 4.0, 'audio': None},
                {'id': 'demo-s003', 'kind': 'narration', 'start': 4.0, 'end': 6.0}]}))
            for name in ('master.mp4', 'voice.wav'):
                (root / name).write_bytes(name.encode())
            signal = np.random.default_rng(7).standard_normal(6 * 8000).astype(np.float32)
            output = root / 'sync.json'
            argv = ['sync', str(root / 'master.mp4'), str(root / 'voice.wav'), str(plan), str(output)]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(sync, 'load', return_value=signal), \
                    mock.patch('builtins.print'):
                sync.main()
            result = json.loads(output.read_text())
            self.assertEqual([row['event'] for row in result['results']], ['demo-s001', 'demo-s003'])
            self.assertEqual(result['events'], 2)
            self.assertEqual(result['pictureEventsWithoutVoice'], ['demo-s002'])
            self.assertTrue(result['passed'])


class ProcessedFidelityPictureEventTest(unittest.TestCase):
    def test_wordless_picture_events_have_no_window_to_compare(self):
        import audit_processed_audio_fidelity as fidelity
        with tempfile.TemporaryDirectory() as temp:
            ep = Path(temp) / 'demo-project'
            (ep / 'work/master').mkdir(parents=True)
            master = ep / 'work/master/final.mp4'
            master.write_bytes(b'master')
            master.with_suffix('.controlled.wav').write_bytes(b'controlled')
            receipt = ep / 'work/master/receipt.json'
            receipt.write_text(json.dumps({
                'output': 'work/master/final.mp4', 'outputSha256': sync.sha(master),
                'socialCompression': {'controlledSha256': sync.sha(master.with_suffix('.controlled.wav'))},
                'normalizer': 'anull', 'encodedCorrections': [{'filter': 'anull'}], 'runtimeSeconds': 6.0}))
            plan = ep / 'work/master/edit-plan.json'
            plan.write_text(json.dumps({'events': [
                {'id': 'demo-s001', 'kind': 'narration', 'start': 0.0, 'subtitleAudioAt': 0.0,
                 'subtitleAudioDuration': 3.0},
                {'id': 'demo-s002', 'kind': 'picture', 'start': 3.0, 'audio': None}]}))
            signal = np.random.default_rng(3).standard_normal(6 * 8000).astype(np.float32)

            def fake_run(command, **_):
                if command[0] == 'ffmpeg':
                    Path(command[-1]).write_bytes(b'expected')
            argv = ['fidelity', str(ep), 'work/master/receipt.json', 'work/master/edit-plan.json', 'work/fidelity-v1']
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(fidelity.platform, 'system', return_value='Linux'), \
                    mock.patch.object(fidelity.subprocess, 'run', side_effect=fake_run), \
                    mock.patch.object(sync, 'load', return_value=signal), mock.patch('builtins.print'):
                fidelity.main()
            audit = json.loads((ep / 'work/fidelity-v1/audit.json').read_text())
            self.assertEqual([row['event'] for row in audit['results']], ['demo-s001'])
            self.assertTrue(audit['passed'])


if __name__ == '__main__':
    unittest.main()
