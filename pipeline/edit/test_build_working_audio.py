"""Pure metadata checks: no media, subprocess, network or cloud execution."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_working_audio import validate_working_clock


class WorkingClockTests(unittest.TestCase):
    def setUp(self):
        self.clock=dict(kind='measured-speech-clock-not-renderable-edit-plan',renderable=False,
                        episode='DEMO',runtimeSeconds=5,eventCount=2,events=[
                            dict(id='demo-s001',shot=1,start=0,end=2,duration=2),
                            dict(id='demo-s005',shot=5,start=2,end=5,duration=3)])

    def test_episode_specific_scene_count_and_omissions_are_valid(self):
        validate_working_clock(self.clock,'DEMO')

    def test_rejects_missing_or_duplicate_events(self):
        for change in ('count','id','shot'):
            c=copy.deepcopy(self.clock)
            if change=='count':c['eventCount']=144
            else:c['events'][1][change]=c['events'][0][change]
            with self.subTest(change=change),self.assertRaises(RuntimeError):validate_working_clock(c,'DEMO')

    def test_rejects_gap_nonfinite_and_wrong_duration(self):
        for key,value in [('start',2.1),('end',float('nan')),('duration',4)]:
            c=copy.deepcopy(self.clock);c['events'][1][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):validate_working_clock(c,'DEMO')

    def test_rejects_wrong_episode_and_final_clock(self):
        with self.assertRaises(RuntimeError):validate_working_clock(self.clock,'OTHER')
        self.clock['renderable']=True
        with self.assertRaises(RuntimeError):validate_working_clock(self.clock,'DEMO')


if __name__=='__main__':unittest.main()
