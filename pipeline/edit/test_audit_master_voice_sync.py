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


if __name__ == '__main__':
    unittest.main()
