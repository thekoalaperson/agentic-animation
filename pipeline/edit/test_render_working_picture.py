import copy
import unittest
from render_working_picture import validate


class WorkingPictureTest(unittest.TestCase):
    def setUp(self):
        self.c=dict(kind='measured-speech-clock-not-renderable-edit-plan',renderable=False,
                    episode='DEMO',runtimeSeconds=2,eventCount=1,
                    events=[dict(id='a',shot=1,start=0,end=2,duration=2)])
        self.r=dict(episode='DEMO',mode='measurement-only-score-gates-preserved',
                    errors=[],missing=[dict(type='score-admission')],measuredClock=dict(sha256='h'),
                    pictureBlocks=[dict(start=0,duration=2,status='ready-native-picture-audio-sync')])

    def test_only_score_may_remain_pending(self):
        validate(self.r,self.c,'h')
        for change in [dict(errors=[{'type':'dialogue-admission'}]),
                       dict(missing=[{'type':'approved-h3-coverage'}]),
                       dict(measuredClock={'sha256':'wrong'}),
                       dict(exactCoverageDeficits=[{'frames':1}])]:
            r=copy.deepcopy(self.r);r.update(change)
            with self.assertRaises(RuntimeError):validate(r,self.c,'h')

    def test_gap_and_short_picture_rejected(self):
        for key,value in [('start',1),('duration',1),('status','pending')]:
            r=copy.deepcopy(self.r);r['pictureBlocks'][0][key]=value
            with self.assertRaises(RuntimeError):validate(r,self.c,'h')


if __name__=='__main__':unittest.main()
