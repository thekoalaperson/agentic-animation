import json
from pathlib import Path
import tempfile
import unittest

from pipeline.edit import package_hindi_delivery as full
from pipeline.edit import test_package_hindi_delivery as fixtures

digest = fixtures.digest

from pipeline.edit.package_hindi_longform import shift_srt, build_kit

class LongformPackagingTests(unittest.TestCase):
    def test_subtitle_shift_preserves_words_and_duration(self):
        source = '1\n00:00:59,990 --> 00:01:00,900\nयह एक परीक्षण है।\n'
        changed = shift_srt(source, 21)
        self.assertEqual(changed, '1\n00:01:00,011 --> 00:01:00,921\nयह एक परीक्षण है।\n')

    def test_builtin_provenance_accepts_bound_prompt_and_rejects_tampering(self):
        fixture = fixtures.HindiDeliveryPackageTests('test_readiness_passes_complete_nonpublishing_contract')
        fixture.setUp()
        try:
            prompt = fixture.episode / 'work/inputs/prompt.txt'
            prompt.write_text('Exact selected text')
            thumb = fixture.config['thumbnail']
            thumb['method'] = 'builtin-imagegen-edit'
            recipe = {'schemaVersion': 1, 'method': thumb['method'], 'renderer': 'image_gen.imagegen',
                      'textlessSource': thumb['textlessSource'], 'textlessSha256': thumb['textlessSha256'],
                      'output': thumb['source'], 'outputSha256': thumb['sha256'],
                      'prompt': 'work/inputs/prompt.txt', 'promptSha256': digest(prompt),
                      'titleLines': ['SHAKTI BHI.', 'KARUNA BHI.']}
            fixture.thumbnail_recipe.write_text(json.dumps(recipe))
            thumb['titleRecipeSha256'] = digest(fixture.thumbnail_recipe)
            config_path = fixture.episode / 'work/config-check.json'
            config_path.write_text(json.dumps(fixture.config))
            _, gates = full.analyze_config(fixture.episode, config_path)
            self.assertEqual(gates, [])
            prompt.write_text('Changed prompt')
            with self.assertRaisesRegex(full.DeliveryError, 'prompt provenance'):
                full.analyze_config(fixture.episode, config_path)
        finally:
            fixture.tearDown()

    def test_kit_has_selected_copy_and_no_short_player(self):
        cfg = {'episode': 'sample-film', 'channel': {'handle': '@example-studio'},
               'metadata': {'title': 'Selected & title', 'description': 'कथा', 'tags': ['story'],
                            'chapters': [], 'hashtags': ['#Story'],
                            'uploadSettings': {'videoLanguage': 'Hindi', 'aiUse': 'Yes', 'visibility': 'Private',
                                               'madeForKids': 'Review before upload', 'category': 'Film & Animation'}}}
        files = {'long-hi': 'master.mp4', 'subtitles-hi': 'hi.srt', 'subtitles-en': 'en.srt',
                 'thumbnail-master': 'thumbnails/a.png', 'thumbnail-hi': 'thumbnails/a.jpg'}
        page = build_kit(cfg, files, '../work/ending.mp4', 7.0)
        parsed = full.KitAudit(); parsed.feed(page)
        self.assertEqual(parsed.copy_text['hi-title'], 'Selected & title')
        self.assertEqual(set(parsed.copy_targets), {'hi-title', 'hi-description', 'common-tags'})
        self.assertNotIn('<video', page)
        self.assertIn('7-second', page)
        self.assertIn('AI-use declaration</dt><dd>Yes', page)
        self.assertIn('Selected &amp; title', page)
        self.assertIn('Publication not authorized', page)


if __name__ == '__main__':
    unittest.main()
