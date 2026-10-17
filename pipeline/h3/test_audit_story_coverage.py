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


if __name__ == '__main__':
    unittest.main()
