import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import derive_subtitle_pairs as derive


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeriveSubtitlePairsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ep = Path(self.temp.name) / 'demo-project'
        work = self.ep / 'work/assembly-v1'
        work.mkdir(parents=True)

        def speech(scene, start, hi, en):
            return dict(id=scene, kind='narration', start=start, duration=3.0, end=start + 3.0,
                        subtitleAudioAt=0.0, subtitleAudioDuration=2.5, text=hi,
                        subtitleTextHi=hi, subtitleTextEn=en)
        events = [speech('demo-s001', 0.0, 'दीपक जलता है।', 'A lamp burns.'),
                  dict(id='demo-s002', kind='picture', start=3.0, duration=2.0, end=5.0,
                       audio=None, picture=None, text=''),
                  speech('demo-s003', 5.0, 'रात आती है।', 'Night falls.')]
        self.plan = work / 'edit-plan.json'
        self.plan.write_text(json.dumps(dict(renderable=True, finalApproved=False, timelineFingerprint='f' * 64,
                                             runtimeSeconds=8.0, events=events), ensure_ascii=False))
        timing = []
        for language in ('hi', 'en'):
            path = work / f'subtitle-{language}-timing.json'
            shots = [dict(id=e['id'], start=e['start'], dur=e['duration'], audioAt=0.0, audioDuration=2.5,
                          text=e['subtitleText' + language.title()]) for e in events if e['kind'] != 'picture']
            path.write_text(json.dumps(dict(language=language, shots=shots), ensure_ascii=False))
            timing.append(path)
        manuscript = self.ep / 'work/review/pairs.json'
        manuscript.parent.mkdir(parents=True)
        manuscript.write_text(json.dumps(dict(
            kind='source-bound-authored-subtitle-pairs', authority='demo translation review',
            sourceBindings=[dict(path=str(p.relative_to(self.ep)), sha256=sha(p)) for p in timing],
            items=[dict(id='demo-s001', hi=['दीपक जलता है।'], en=['A lamp burns.']),
                   dict(id='demo-s003', hi=['रात आती है।'], en=['Night falls.'])]), ensure_ascii=False))
        self.recipe = self.ep / 'work/review/recipe.json'
        self.recipe.write_text(json.dumps(dict(
            kind='source-bound-subtitle-derivation', outputDirectory='work/assembly-v2',
            sourcePlan=dict(path=str(self.plan.relative_to(self.ep)), sha256=sha(self.plan)),
            manuscript=dict(path=str(manuscript.relative_to(self.ep)), sha256=sha(manuscript)))))

    def tearDown(self):
        self.temp.cleanup()

    def test_wordless_picture_event_needs_no_manuscript_item(self):
        with mock.patch.object(sys, 'argv', ['derive', str(self.ep), str(self.recipe.relative_to(self.ep))]), \
                mock.patch('builtins.print'):
            derive.main()
        out = self.ep / 'work/assembly-v2'
        plan = json.loads((out / 'edit-plan.json').read_text())
        self.assertEqual(plan['events'][1], json.loads(self.plan.read_text())['events'][1])
        self.assertEqual([e.get('subtitleCuePairs') is not None for e in plan['events']], [True, False, True])
        receipt = json.loads((out / 'derivation-audit.json').read_text())
        self.assertEqual((receipt['sceneCount'], receipt['cueCount']), (2, 2))


if __name__ == '__main__':
    unittest.main()
