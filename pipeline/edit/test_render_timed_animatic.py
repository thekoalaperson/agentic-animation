import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_timed_animatic import validate


class AnimaticContractTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'kind': 'timed-still-animatic-not-final', 'finalApproved': False, 'fps': 24,
                     'scenes': [{'id': 'demo-s001', 'mode': 'narration', 'frames': 240,
                                 'text': 'एक पूरा परीक्षण वाक्य', 'sourceStatus': 'candidate',
                                 'audio': {'durationSeconds': 9.5}}]}

    def test_whole_speech(self):
        validate(self.plan)

    def test_reject_short_coverage(self):
        self.plan['scenes'][0]['frames'] = 220
        with self.assertRaises(ValueError): validate(self.plan)

    def test_account_for_lead(self):
        self.plan['scenes'][0]['audioLeadFrames'] = 20
        with self.assertRaises(ValueError): validate(self.plan)

    def test_scratch_is_silent(self):
        self.plan['scenes'][0]['mode'] = 'scratch-dialogue'
        with self.assertRaises(ValueError): validate(self.plan)
        self.plan['scenes'][0].pop('audio')
        validate(self.plan)

    def test_never_final(self):
        self.plan['finalApproved'] = True
        with self.assertRaises(ValueError): validate(self.plan)

    def test_unique_ids(self):
        self.plan['scenes'].append(copy.deepcopy(self.plan['scenes'][0]))
        with self.assertRaises(ValueError): validate(self.plan)

    def test_finite_clock(self):
        self.plan['scenes'][0]['audio']['durationSeconds'] = float('nan')
        with self.assertRaises(ValueError): validate(self.plan)

    def test_honest_source_label(self):
        self.plan['scenes'][0]['sourceStatus'] = ''
        with self.assertRaises(ValueError): validate(self.plan)

    def test_contiguous_scene_clock(self):
        self.plan['scenes'][0]['startFrame'] = 1
        with self.assertRaises(ValueError): validate(self.plan)

    def test_total_clock(self):
        self.plan['frames'] = 241
        with self.assertRaises(ValueError): validate(self.plan)


if __name__ == '__main__': unittest.main()
