"""The screenplay reader inventories source units and reports how rows use them."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import read_screenplay as reader


DEMO = """Title: Demo
Credit: written by
Author: Demo Writer

# Part one

= A lamp is lit.

INT. DEMO ROOM - NIGHT #1#

A lamp on a table.

[TIME 21:00]

DEMO ONE (V.O.) (CONT'D)
(quietly)
मैं यहाँ हूँ।
दीपक जलता है।

DEMO TWO
Where?

DEMO ONE ^
Here.

@demo three
तुम कौन हो?

[[A note for the director.]]

/* An old line
that was cut. */

डेमो
यह संकेत नहीं है।

CUT TO:

.MEMORY

~La la

> THE END <

!LOUD NOISE
Then quiet.
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


class ReadScreenplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name) / 'demo-project'
        (self.project / 'story').mkdir(parents=True)
        self.screenplay = write(self.project / 'story/demo.fountain', DEMO)

    def tearDown(self):
        self.temp.cleanup()

    def read(self, *extra):
        with mock.patch('builtins.print'):
            self.assertEqual(reader.main([str(self.project), '--screenplay', str(self.screenplay), *extra]), 0)

    def units(self, version=1):
        return json.loads((self.project / f'story/source-units-v{version}.json').read_text(encoding='utf-8'))

    def test_every_element_type_becomes_a_unit_without_shots(self):
        self.read()
        doc = self.units()
        self.assertEqual(doc['titlePage'], {'Title': 'Demo', 'Credit': 'written by', 'Author': 'Demo Writer'})
        self.assertEqual(doc['source'], 'story/demo.fountain')
        self.assertEqual(doc['sourceSha256'], reader.sha256_bytes(self.screenplay.read_bytes()))
        kinds = [(u['id'], u['kind']) for u in doc['units']]
        self.assertEqual(kinds, [
            ('s000.e001', 'section'), ('s000.e002', 'synopsis'),
            ('s001.e001', 'heading'), ('s001.e002', 'action'), ('s001.e003', 'action'),
            ('s001.e004', 'dialogue'), ('s001.e005', 'dialogue'), ('s001.e006', 'dialogue'),
            ('s001.e007', 'dialogue'), ('s001.e008', 'note'), ('s001.e009', 'boneyard'),
            ('s001.e010', 'action'), ('s001.e011', 'transition'),
            ('s002.e001', 'heading'), ('s002.e002', 'lyric'), ('s002.e003', 'action'), ('s002.e004', 'action')])
        by_id = {u['id']: u for u in doc['units']}
        self.assertEqual(by_id['s001.e001']['sceneNumber'], '1')
        first = by_id['s001.e004']
        self.assertEqual((first['character'], first['extensions']), ('DEMO ONE', ['V.O.', "CONT'D"]))
        self.assertEqual(first['text'], 'मैं यहाँ हूँ।\nदीपक जलता है।')
        self.assertEqual([line['kind'] for line in first['lines']], ['parenthetical', 'dialogue', 'dialogue'])
        self.assertEqual(first['textSha256'], reader.text_sha(first['text']))
        self.assertEqual(by_id['s001.e005']['dual'], {'side': 'left', 'pair': 's001.e006'})
        self.assertEqual(by_id['s001.e006']['dual'], {'side': 'right', 'pair': 's001.e005'})
        self.assertEqual((by_id['s001.e007']['character'], by_id['s001.e007']['forcedCue']), ('demo three', True))
        self.assertEqual(by_id['s001.e009']['text'], 'An old line\nthat was cut.')
        self.assertTrue(by_id['s002.e003']['centered'])
        self.assertEqual(by_id['s002.e004']['text'], 'LOUD NOISE\nThen quiet.')
        self.assertEqual(doc['flags']['bracketLines'], ['s001.e003'])
        self.assertEqual(doc['flags']['notes'], ['s001.e008'])
        # A Devanagari line before text is not a cue; it is listed, never guessed.
        self.assertEqual([c['text'] for c in doc['flags']['uncertainCues']], ['डेमो'])
        self.assertEqual(by_id['s001.e010']['kind'], 'action')
        for unit in doc['units']:
            self.assertFalse({'still_id', 'shot', 'durationSeconds', 'holdSeconds'} & set(unit))
        self.assertTrue((self.project / 'story/source-units-v1.md').is_file())

    def test_rereading_the_same_file_writes_nothing_new(self):
        self.read()
        before = sorted(p.name for p in (self.project / 'story').iterdir())
        self.read()
        self.assertEqual(sorted(p.name for p in (self.project / 'story').iterdir()), before)

    def test_a_changed_reader_reads_the_same_file_again(self):
        self.read()
        doc = self.units()
        doc['readerSha256'] = '0' * 64
        (self.project / 'story/source-units-v1.json').write_text(json.dumps(doc), encoding='utf-8')
        self.read()
        diff = json.loads((self.project / 'story/source-units-v2.diff.json').read_text(encoding='utf-8'))
        self.assertEqual([diff[k] for k in ('moved', 'changed', 'added', 'removed', 'renumbered')], [[]] * 5)

    def test_a_revision_is_a_new_version_with_a_diff(self):
        self.read()
        v1 = (self.project / 'story/source-units-v1.json').read_bytes()
        revised = (DEMO.replace('A lamp on a table.\n', 'A lamp on a table.\n\nA window opens.\n')
                   .replace('Where?', 'Where now?').replace('~La la\n\n', ''))
        revised = revised.replace('@demo three\nतुम कौन हो?\n\n', '') + '\n@demo three\nतुम कौन हो?\n'
        write(self.screenplay, revised)
        self.read()
        self.assertEqual((self.project / 'story/source-units-v1.json').read_bytes(), v1)
        diff = json.loads((self.project / 'story/source-units-v2.diff.json').read_text(encoding='utf-8'))
        self.assertEqual(diff['previous']['sha256'], reader.sha256_bytes(v1))
        self.assertEqual([a['text'] for a in diff['added']], ['A window opens.'])
        self.assertEqual([(c['old'], c['before'], c['after']) for c in diff['changed']],
                         [('s001.e005', 'Where?', 'Where now?')])
        self.assertEqual([r['text'] for r in diff['removed']], ['La la'])
        self.assertEqual([(m['old'], m['kind']) for m in diff['moved']], [('s001.e007', 'dialogue')])
        self.assertIn({'old': 's001.e004', 'new': 's001.e005'}, diff['renumbered'])
        self.assertEqual(self.units(2)['previous']['path'], 'story/source-units-v1.json')

    def test_existing_output_is_never_replaced(self):
        target = write(self.project / 'story/source-units-v1.json', '{}\n')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(reader.main([str(self.project), '--screenplay', str(self.screenplay),
                                          '--output', 'story/source-units-v1.json']), 2)
        self.assertEqual(target.read_text(), '{}\n')

    def test_normalisation_separates_spelling_from_change(self):
        self.assertEqual(reader.compare('मैं यहाँ हूँ।', 'मैं  यहाँ हूँ'), 'verbatim')
        self.assertEqual(reader.compare('क़लम', 'कलम'), 'nukta-or-joiner')
        self.assertEqual(reader.compare('क्‍ष', 'क्ष'), 'nukta-or-joiner')
        self.assertEqual(reader.compare('मैं यहाँ हूँ।', 'मैं वहाँ हूँ।'), 'changed')
        self.assertTrue(reader.contains('मैं यहाँ हूँ। दीपक जलता है।', 'दीपक जलता है'))


class UnitsReportTest(unittest.TestCase):
    """--units lists how rows use the units; it holds no totals and gates nothing."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name) / 'demo-project'
        write(self.project / 'story/demo.fountain', """INT. DEMO ROOM - DAY

A lamp on a table.

DEMO ONE
मैं यहाँ हूँ। दीपक जलता है।

DEMO TWO
तुम कौन हो?

DEMO ONE
क़लम यहाँ है।

DEMO TWO
अब चलो।

A window opens.
""")
        with mock.patch('builtins.print'):
            reader.main([str(self.project), '--screenplay', str(self.project / 'story/demo.fountain')])

    def tearDown(self):
        self.temp.cleanup()

    def report(self, stills, *extra, overrides=None):
        write(self.project / 'story/storyboard_plan.json',
              json.dumps({'episode': 'demo', 'stills': stills}, ensure_ascii=False))
        args = [str(self.project), '--units', 'story/source-units-v1.json', '--output', 'work/units.json']
        if overrides is not None:
            write(self.project / 'work/script-v1/overrides.json',
                  json.dumps({'items': overrides}, ensure_ascii=False))
            args += ['--overrides', 'work/script-v1/overrides.json']
        out = self.project / 'work/units.json'
        if out.exists():
            out.unlink()
        with mock.patch('builtins.print'):
            self.assertEqual(reader.main(args + list(extra)), 0)
        return json.loads(out.read_text(encoding='utf-8'))

    def test_dispositions_split_merge_reorder_and_spelling(self):
        stills = [
            {'still_id': 'demo-s001', 'speaker': None, 'delivery': 'wordless', 'source_units': ['s001.e002']},
            {'still_id': 'demo-s002', 'speaker': 'DEMO ONE', 'narration_hi': 'मैं यहाँ हूँ।',
             'narration_en': 'I am here.', 'source_units': ['s001.e003']},
            {'still_id': 'demo-s003', 'speaker': 'DEMO ONE', 'narration_hi': 'दीपक जलता है।',
             'narration_en': 'The lamp burns.', 'source_units': ['s001.e003']},
            {'still_id': 'demo-s004', 'speaker': None, 'voice': 'DEMO ONE', 'delivery': 'unsynced',
             'narration_hi': 'कलम यहाँ है।', 'narration_en': 'The pen is here.', 'source_units': ['s001.e005']},
            {'still_id': 'demo-s005', 'speaker': 'DEMO TWO', 'narration_hi': 'तुम कौन हो? अब चलो।',
             'narration_en': 'Who are you? Let us go.', 'source_units': ['s001.e004', 's001.e006']},
            {'still_id': 'demo-s006', 'speaker': 'DEMO TWO', 'narration_hi': 'यह नया है।',
             'narration_en': 'This is new.'},
        ]
        write(self.project / 'story/reasons.json', json.dumps({'s001.e007': 'Left for the next cut.'}))
        report = self.report(stills, '--reasons', 'story/reasons.json')
        units = {u['id']: u for u in report['sourceUnits']}
        self.assertEqual(units['s001.e002']['use'], 'pictured without words')
        self.assertEqual((units['s001.e003']['use'], units['s001.e003']['text']), ('spoken on camera', 'verbatim'))
        self.assertEqual(units['s001.e003']['splitAcross'], ['demo-s002', 'demo-s003'])
        self.assertEqual((units['s001.e005']['use'], units['s001.e005']['text']),
                         ('heard over picture', 'nukta-or-joiner'))
        self.assertEqual(units['s001.e004']['mergedWith'], ['s001.e006'])
        self.assertEqual(units['s001.e004']['text'], 'verbatim')
        self.assertTrue(units['s001.e004']['reordered'])  # cited after a later unit was already used
        self.assertEqual((units['s001.e007']['use'], units['s001.e007']['reason']),
                         ('not used', 'Left for the next cut.'))
        self.assertEqual(report['spokenTextInNoUnit'], [{'still': 'demo-s006', 'text': 'यह नया है।'}])
        self.assertEqual(report['rowsCitingNoUnit'], ['demo-s006'])
        flat = json.dumps(report)
        for word in ('total', 'percent', 'score', 'Count'):
            self.assertNotIn(word, flat)

    def test_changed_text_and_override_text_are_used(self):
        stills = [{'still_id': 'demo-s001', 'speaker': 'DEMO ONE', 'narration_hi': 'पुराना',
                   'narration_en': 'Old', 'source_units': ['s001.e003']}]
        report = self.report(stills, overrides={'demo-s001': {
            'speaker': 'DEMO ONE', 'narration_hi': 'मैं वहाँ हूँ।', 'narration_en': 'I am there.'}})
        unit = {u['id']: u for u in report['sourceUnits']}['s001.e003']
        self.assertEqual(unit['text'], 'changed')
        self.assertEqual(unit['spoken'], [{'still': 'demo-s001', 'text': 'मैं वहाँ हूँ।'}])
        self.assertIn('overrides', report)

    def test_frozen_dialogue_lines_and_subtitles_are_compared(self):
        prompt = write(self.project / 'work/grok-dialogue-v1/prompts/demo-s001.txt',
                       'A lamp on a table. DEMO ONE says: "मैं यहाँ हूँ।" Then: "कुछ और।"')
        write(self.project / 'work/grok-dialogue-v1/execution-manifest.json', json.dumps({
            'jobs': [{'id': 'demo-s001', 'hindi': 'मैं यहाँ हूँ।',
                      'prompt': str(prompt.relative_to(self.project))}]}, ensure_ascii=False))
        stills = [{'still_id': 'demo-s001', 'speaker': 'DEMO ONE', 'narration_hi': 'मैं यहाँ हूँ। दीपक जलता है।',
                   'narration_en': 'I am here. The lamp burns.', 'source_units': ['s001.e003']}]
        report = self.report(stills, '--dialogue-manifest', 'work/grok-dialogue-v1/execution-manifest.json')
        [line] = report['dialogueJobLines']
        self.assertEqual(line['jobAgainstUnits'], 'verbatim')
        self.assertEqual([reader.normalize(t) for t in line['promptLinesNotInUnits']], ['कुछ और'])
        self.assertEqual(report['subtitleCarriesUnspokenWords'], [{
            'still': 'demo-s001', 'subtitle': 'मैं यहाँ हूँ। दीपक जलता है।', 'spoken': 'मैं यहाँ हूँ।'}])

    def test_identical_english_on_split_halves_is_listed(self):
        stills = [
            {'still_id': 'demo-s001', 'speaker': 'DEMO ONE', 'narration_hi': 'मैं यहाँ हूँ।',
             'narration_en': 'I am here and the lamp burns.', 'source_units': ['s001.e003']},
            {'still_id': 'demo-s002', 'speaker': 'DEMO ONE', 'narration_hi': 'दीपक जलता है।',
             'narration_en': 'I am here and the lamp burns.', 'source_units': ['s001.e003']}]
        report = self.report(stills)
        self.assertEqual([r['unit'] for r in report['splitRowsWithIdenticalEnglish']], ['s001.e003'])

    def test_episode_range_limits_the_report_and_records_the_units_hash(self):
        stills = [{'still_id': 'demo-s001', 'speaker': 'DEMO TWO', 'narration_hi': 'तुम कौन हो?',
                   'narration_en': 'Who are you?', 'source_units': ['s001.e004']},
                  {'still_id': 'demo-s002', 'speaker': None, 'delivery': 'wordless', 'source_units': ['s001.e007']}]
        report = self.report(stills, '--episode-range', 's001.e002', 's001.e004')
        self.assertEqual([u['id'] for u in report['sourceUnits']], ['s001.e002', 's001.e003', 's001.e004'])
        units_bytes = (self.project / 'story/source-units-v1.json').read_bytes()
        self.assertEqual(report['episodeRange'], {'first': 's001.e002', 'last': 's001.e004',
                                                  'unitsSha256': reader.sha256_bytes(units_bytes)})
        self.assertEqual(report['rowsCitingUnitsOutsideRange'], [{'still': 'demo-s002', 'units': ['s001.e007']}])

    def test_unknown_citations_are_named(self):
        write(self.project / 'story/storyboard_plan.json', json.dumps({'episode': 'demo', 'stills': [
            {'still_id': 'demo-s001', 'speaker': None, 'delivery': 'wordless', 'source_units': ['s009.e001']}]}))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = reader.main([str(self.project), '--units', 'story/source-units-v1.json',
                                '--output', 'work/units.json'])
        self.assertEqual(code, 2)
        self.assertIn('s009.e001', stderr.getvalue())
        self.assertFalse((self.project / 'work/units.json').exists())


if __name__ == '__main__':
    unittest.main()
