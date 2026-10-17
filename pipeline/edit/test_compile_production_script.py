import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import compile_production_script as compiler


class CompileProductionScriptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ep = Path(self.temp.name) / 'demo-project'
        (self.ep / 'story').mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def compile(self, lines, speakers=None, out='work/script-v1', extra=(), rows=None,
                board_path='story/storyboard_plan.json'):
        speakers = speakers or [None] * len(lines)
        board = {'episode': 'demo', 'stills': [
            {'still_id': f'demo-s{i:03d}', 'beat': 'b', 'speaker': s, **((rows or {}).get(i, {}))}
            for i, s in enumerate(speakers, 1)]}
        (self.ep / board_path).write_text(json.dumps(board))
        script = self.ep / 'story/script.txt'
        script.write_text('\n'.join(lines) + '\n')
        argv = ['compile', str(self.ep), str(script), str(self.ep / out), *extra]
        with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.print'):
            compiler.main()
        root = self.ep / out
        return (json.loads((root / 'overrides.json').read_text()), json.loads((root / 'audit.json').read_text()),
                (root / 'performed-hi.md').read_text())

    def test_four_column_script_keeps_its_original_shape(self):
        overrides, audit, performed = self.compile([
            '001 | Narrator | कमरे में एक दीपक जलता है। | A lamp burns in the room.',
            '002 | Character One | मैं यहाँ हूँ। | I am here.'], [None, 'Character One'])
        reason = 'Authored production rewrite; literal supplied text retained. No inherited image approval.'
        self.assertEqual(overrides['items'], {
            'demo-s001': {'speaker': None, 'narration_hi': 'कमरे में एक दीपक जलता है।',
                          'narration_en': 'A lamp burns in the room.', 'reason': reason},
            'demo-s002': {'speaker': 'Character One', 'narration_hi': 'मैं यहाँ हूँ।',
                          'narration_en': 'I am here.', 'reason': reason}})
        self.assertEqual(list(audit), ['episode', 'sceneCount', 'dialogueCount', 'narrationCharacters',
                                       'roleChanges', 'sourceSha256', 'overridesSha256',
                                       'mediaGenerationPerformed', 'visualApprovalClaimed'])
        self.assertEqual((audit['dialogueCount'], audit['narrationCharacters']), (1, len('कमरे में एक दीपक जलता है।')))
        self.assertEqual(performed, '001. कमरे में एक दीपक जलता है।\n\n002. मैं यहाँ हूँ।\n')

    def test_delivery_column_marks_wordless_and_voice_over_rows(self):
        overrides, audit, performed = self.compile([
            '1 | - | | | wordless',
            '2 | - | |',
            '3 | Character One | मैं यहाँ हूँ। | I am here. | unsynced',
            '4 | Character One | तुम कहाँ हो? | Where are you? | synced',
            '5 | Narrator | दीपक बुझता है। | The lamp goes out. | unsynced'],
            [None, None, 'Character One', 'Character One', None])
        items = overrides['items']
        self.assertEqual({k: items['demo-s001'][k] for k in ('speaker', 'narration_hi', 'narration_en', 'delivery')},
                         {'speaker': None, 'narration_hi': '', 'narration_en': '', 'delivery': 'wordless'})
        self.assertEqual(items['demo-s002']['delivery'], 'wordless')
        self.assertEqual((items['demo-s003']['speaker'], items['demo-s003']['voice'], items['demo-s003']['delivery']),
                         (None, 'Character One', 'unsynced'))
        self.assertNotIn('delivery', items['demo-s004'])
        self.assertNotIn('delivery', items['demo-s005'])
        self.assertEqual((audit['wordlessCount'], audit['voiceOverCount'], audit['dialogueCount']), (2, 1, 1))
        self.assertEqual(audit['narrationCharacters'], len('मैं यहाँ हूँ।') + len('दीपक बुझता है।'))
        self.assertEqual(audit['roleChanges'], [{'id': 'demo-s003', 'before': 'Character One', 'after': None}])
        self.assertEqual(performed.split('\n\n')[0], '3. मैं यहाँ हूँ।')

    def test_errors_name_the_row(self):
        cases = [
            (['1 | - | कुछ | Something'], 'row 1 .*empty'),
            (['1 | - | | | synced'], 'row 1 .*wordless'),
            (['1 | Character One | | | wordless'], 'row 1 .*wordless'),
            (['1 | Narrator | दीपक। | Lamp. | synced'], 'row 1 .*Narrator'),
            (['1 | Narrator | दीपक। | Lamp. | whispered'], 'row 1 .*unknown delivery'),
            (['1 | Narrator | दीपक। |'], 'row 1 .*non-empty'),
            (['1 | Narrator | दीपक।'], 'row 1: expected four columns'),
            (['2 | Narrator | दीपक। | Lamp.'], 'row 1: ordinal'),
            (['1 | | दीपक। | Lamp.'], 'row 1 .*speaker'),
        ]
        for index, (lines, message) in enumerate(cases):
            with self.subTest(lines=lines):
                with self.assertRaisesRegex(compiler.ScriptError, message):
                    self.compile(lines, out=f'work/script-v{index}')
        with self.assertRaisesRegex(compiler.ScriptError, '2 script rows for 1 storyboard'):
            board = {'episode': 'demo', 'stills': [{'still_id': 'demo-s001'}]}
            (self.ep / 'story/storyboard_plan.json').write_text(json.dumps(board))
            script = self.ep / 'story/script.txt'
            script.write_text('1 | - | |\n2 | - | |\n')
            with mock.patch.object(sys, 'argv', ['compile', str(self.ep), str(script), str(self.ep / 'work/x')]):
                compiler.main()

    def test_alternate_storyboard_is_recorded_and_the_default_is_not(self):
        overrides, _, _ = self.compile(['1 | Narrator | दीपक। | Lamp.'])
        self.assertNotIn('storyboard', overrides)
        overrides, _, _ = self.compile(['1 | Narrator | दीपक। | Lamp.'], out='work/script-alt',
                                       board_path='story/storyboard-alt.json',
                                       extra=['--storyboard', 'story/storyboard-alt.json'])
        self.assertEqual(overrides['storyboard'], 'story/storyboard-alt.json')
        with self.assertRaises(ValueError):
            self.compile(['1 | Narrator | दीपक। | Lamp.'], out='work/script-out',
                         extra=['--storyboard', '../outside.json'])

    def test_source_script_reports_authored_lines_by_containment(self):
        (self.ep / 'story/demo.fountain').write_text(
            'INT. DEMO ROOM - DAY\n\nA lamp on a table.\n\nDEMO ONE\n(softly)\n'
            'मैं यहाँ हूँ, दीपक जलता है।\n', encoding='utf-8')
        lines = ['1 | DEMO ONE | मैं यहाँ हूँ। | I am here.',
                 '2 | DEMO ONE | दीपक जलता है | The lamp burns.',
                 '3 | Narrator | रात गहरी है। | The night is deep.',
                 '4 | - | | | wordless']
        speakers = ['DEMO ONE', 'DEMO ONE', None, None]
        _, plain, _ = self.compile(lines, speakers)
        self.assertNotIn('authoredBeyondSource', plain)
        _, audit, _ = self.compile(lines, speakers, out='work/script-v2',
                                   extra=['--source-script', str(self.ep / 'story/demo.fountain')])
        beyond = audit['authoredBeyondSource']
        self.assertEqual(beyond['source'], 'story/demo.fountain')
        self.assertEqual(beyond['rows'], [{'id': 'demo-s003', 'text': 'रात गहरी है।'}])
        self.assertEqual([k for k in audit if k != 'authoredBeyondSource'], list(plain))

    def test_speech_density_counts_without_a_rate_and_estimates_with_one(self):
        lines = ['1 | DEMO ONE | मैं यहाँ हूँ। | I am here.', '2 | Narrator | दीपक जलता है। | The lamp burns.',
                 '3 | - | | | wordless']
        speakers = ['DEMO ONE', None, None]
        self.compile(lines, speakers)
        self.assertFalse((self.ep / 'work/script-v1/speech-density.json').exists())
        self.compile(lines, speakers, out='work/script-v2', extra=['--speech-density'],
                     rows={2: {'holdSeconds': 0.5}})
        counts = json.loads((self.ep / 'work/script-v2/speech-density.json').read_text())
        self.assertIsNone(counts['charactersPerSecond'])
        self.assertEqual([(r['id'], r['characters']) for r in counts['rows']],
                         [('demo-s001', len('मैं यहाँ हूँ।')), ('demo-s002', len('दीपक जलता है।')), ('demo-s003', 0)])
        self.assertFalse(any('estimatedSpeechSeconds' in r for r in counts['rows']))
        self.compile(lines, speakers, out='work/script-v3', rows={2: {'holdSeconds': 0.5}},
                     extra=['--characters-per-second', '10', '--clip-cap', '1'])
        rows = json.loads((self.ep / 'work/script-v3/speech-density.json').read_text())['rows']
        self.assertEqual(rows[0]['estimatedSpeechSeconds'], round(len('मैं यहाँ हूँ।') / 10, 3))
        self.assertTrue(rows[0]['estimateExceedsClipCap'])
        self.assertEqual(rows[1]['pictureSeconds'], 0.5)
        self.assertTrue(rows[1]['estimateExceedsPicture'])
        self.assertNotIn('estimateExceedsClipCap', rows[1])  # the cap is a lip-sync clip limit
        for row in rows:
            self.assertFalse({'suggestedClipSeconds', 'split', 'suggestedSplit'} & set(row))

    def test_ordinals_compare_as_integers(self):
        overrides, _, _ = self.compile(['1 | Narrator | दीपक। | Lamp.', '02 | Narrator | रात। | Night.'])
        self.assertEqual(list(overrides['items']), ['demo-s001', 'demo-s002'])


if __name__ == '__main__':
    unittest.main()
