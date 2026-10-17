"""Wordless and voice-over rows in the text and inventory tools."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prepare_episode_production as production
import prepare_subtitle_manuscript as manuscript
from render_working_picture import ownership_contracts

REASON = 'Authored production rewrite; literal supplied text retained. No inherited image approval.'
ITEMS = {
    'demo-s001': dict(speaker=None, narration_hi='दीपक जलता है।', narration_en='A lamp burns.', reason=REASON),
    'demo-s002': dict(speaker=None, narration_hi='', narration_en='', reason=REASON, delivery='wordless'),
    'demo-s003': dict(speaker=None, narration_hi='मैं यहाँ हूँ।', narration_en='I am here.', reason=REASON,
                      delivery='unsynced', voice='Character One'),
    'demo-s004': dict(speaker='Character One', narration_hi='तुम कहाँ हो?', narration_en='Where are you?',
                      reason=REASON),
}


class WordlessProductionInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ep = Path(self.temp.name) / 'demo-project'
        (self.ep / 'story').mkdir(parents=True)
        self.overrides = self.ep / 'work/script-v1/overrides.json'
        self.overrides.parent.mkdir(parents=True)
        self.overrides.write_text(json.dumps({'episode': 'demo', 'items': ITEMS}, ensure_ascii=False))

    def tearDown(self):
        self.temp.cleanup()

    def test_manuscript_skips_wordless_rows(self):
        edits = self.ep / 'work/review/edits.json'
        edits.parent.mkdir(parents=True)
        edits.write_text('{}')
        output = self.ep / 'work/review/manuscript.json'
        with mock.patch('builtins.print'):
            manuscript.prepare(self.ep, self.overrides, edits, output)
        items = json.loads(output.read_text())['items']
        self.assertEqual([item['id'] for item in items], ['demo-s001', 'demo-s003', 'demo-s004'])
        edits.write_text(json.dumps({'demo-s002': {'en': 'Words.', 'editReason': 'x'}}))
        with self.assertRaisesRegex(ValueError, 'wordless'):
            manuscript.prepare(self.ep, self.overrides, edits, self.ep / 'work/review/manuscript-2.json')

    def write_board(self, name='story/storyboard_plan.json'):
        stills, records = [], []
        for scene in ITEMS:
            image = self.ep / f'art/{scene}.png'
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(scene.encode())
            sidecar = image.with_suffix('.json')
            sidecar.write_text(json.dumps({'target_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}))
            stills.append({'still_id': scene, 'beat': 'b', 'speaker': ITEMS[scene]['speaker'],
                           'narration_hi': 'पुराना।', 'narration_en': 'Old.'})
            records.append({'still_id': scene, 'target_output_path': str(image.relative_to(self.ep)),
                            'provenance_sidecar': str(sidecar.relative_to(self.ep))})
        (self.ep / name).write_text(json.dumps({'episode': 'demo', 'stills': stills}))
        (self.ep / 'story/generation_manifest.json').write_text(json.dumps({'episode': 'demo', 'stills': records}))

    def test_alternate_storyboard_and_passage_links_are_read(self):
        self.write_board('story/storyboard-alt.json')
        items = json.loads(self.overrides.read_text())
        items['items']['demo-s004'] = dict(items['items']['demo-s004'], passage='demo-s003')
        self.overrides.write_text(json.dumps(items, ensure_ascii=False))
        argv = ['prepare', str(self.ep), '--inventory-only', '--voice-id', 'demo-voice', '--work', 'work/alt',
                '--overrides', str(self.overrides), '--storyboard', 'story/storyboard-alt.json']
        with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.print'):
            production.main()
        inventory = json.loads((self.ep / 'work/alt/inventory.json').read_text())
        self.assertEqual(inventory['storyboard'], 'story/storyboard-alt.json')
        self.assertEqual(inventory['boardSha256'],
                         hashlib.sha256((self.ep / 'story/storyboard-alt.json').read_bytes()).hexdigest())
        draft = json.loads((self.ep / 'work/alt/dialogue-draft.json').read_text())['items']
        self.assertNotIn('passage', draft[0])  # the paid draft shape is unchanged

    def test_inventory_routes_each_delivery(self):
        stills, records = [], []
        for scene in ITEMS:
            image = self.ep / f'art/{scene}.png'
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(scene.encode())
            sidecar = image.with_suffix('.json')
            sidecar.write_text(json.dumps({'target_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}))
            stills.append({'still_id': scene, 'beat': 'b', 'speaker': ITEMS[scene]['speaker'],
                           'narration_hi': 'पुराना।', 'narration_en': 'Old.'})
            records.append({'still_id': scene, 'target_output_path': str(image.relative_to(self.ep)),
                            'provenance_sidecar': str(sidecar.relative_to(self.ep))})
        (self.ep / 'story/storyboard_plan.json').write_text(json.dumps({'episode': 'demo', 'stills': stills}))
        (self.ep / 'story/generation_manifest.json').write_text(json.dumps({'episode': 'demo', 'stills': records}))
        argv = ['prepare', str(self.ep), '--inventory-only', '--voice-id', 'demo-voice',
                '--overrides', str(self.overrides)]
        with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.print'):
            production.main()
        work = self.ep / 'work/production-intake-v1'
        narration = json.loads((work / 'narration-plan.json').read_text())['items']
        self.assertEqual([(n['shotId'], n.get('speaker')) for n in narration],
                         [('demo-s001', None), ('demo-s003', 'Character One')])
        dialogue = json.loads((work / 'dialogue-draft.json').read_text())['items']
        self.assertEqual([d['id'] for d in dialogue], ['demo-s004'])
        inventory = json.loads((work / 'inventory.json').read_text())
        self.assertEqual((inventory['count'], inventory['wordlessCount']), (4, 1))
        self.assertEqual(inventory['items'][1]['delivery'], 'wordless')
        self.assertNotIn('storyboard', inventory)

    def test_working_picture_needs_no_ledger_for_an_all_dialogue_film(self):
        self.assertEqual(ownership_contracts(self.ep, {'documents': {}}), [])


if __name__ == '__main__':
    unittest.main()
