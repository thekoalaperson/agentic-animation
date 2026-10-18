"""Pure metadata checks: no media, subprocess, network or cloud execution."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_working_audio import resolve_working_placements, validate_working_clock


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



class WorkingScoreAnchorTests(unittest.TestCase):
    def setUp(self):
        self.clock=dict(runtimeSeconds=5,events=[
            dict(id='demo-s001',shot=1,start=0,end=2,duration=2,subtitleAudioAt=0.5,subtitleAudioDuration=1.0),
            dict(id='demo-s005',shot=5,start=2,end=5,duration=3)])
        self.absolute=dict(movement='demo-m1',take=1,timelineStart=0.25,timelineDuration=1.5)

    def test_absolute_placements_pass_unchanged_and_record_nothing(self):
        placements,resolved=resolve_working_placements([self.absolute],self.clock)
        self.assertEqual((placements,resolved),([self.absolute],[]))

    def test_anchored_placement_resolves_on_the_working_clock(self):
        anchored=dict(movement='demo-m2',take=1,anchor=dict(event='demo-s001',at='speechEnd',offsetSeconds=0.25),
                      until=dict(event='demo-s005',at='end'))
        placements,resolved=resolve_working_placements([self.absolute,anchored],self.clock)
        self.assertEqual(placements[0],self.absolute)
        self.assertEqual((placements[1]['timelineStart'],placements[1]['timelineDuration']),(1.75,3.25))
        self.assertEqual(resolved[0]['movement'],'demo-m2')
        self.assertIn('basis',resolved[0]['resolvedFrom'])
        self.assertNotIn('timelineStart',anchored)

    def test_bad_anchors_refuse_with_a_sentence_not_a_key_error(self):
        for anchor in (dict(event='demo-s009',at='start'),dict(event='demo-s005',at='speechStart'),
                       dict(event='demo-s001',at='middle')):
            with self.subTest(anchor=anchor),self.assertRaisesRegex(RuntimeError,'Working score placement'):
                resolve_working_placements([dict(movement='demo-m2',take=1,anchor=anchor,timelineDuration=1)],self.clock)
        with self.assertRaisesRegex(RuntimeError,'after the film'):
            resolve_working_placements([dict(movement='demo-m2',take=1,anchor=dict(event='demo-s005',at='start'),
                                             timelineDuration=4)],self.clock)


if __name__=='__main__':unittest.main()
