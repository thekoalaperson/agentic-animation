import unittest
from map_emotion_clock import map_clock


class EmotionClockTest(unittest.TestCase):
    def test_independent_dimensions_and_hard_transition(self):
        clock={'runtimeSeconds':30,'events':[{'id':str(i),'shot':i,'start':i*5,'end':i*5+5} for i in range(6)]}
        anchors={'dimensions':['a','b'],'hardTransitions':[4], 'anchors':[
            {'shotId':'0','values':{'a':0,'b':100}},
            {'shotId':'2','values':{'a':100,'b':0}},
            {'shotId':'4','values':{'a':10,'b':80}}]}
        rows=map_clock(clock,anchors)['shots']
        self.assertEqual(rows[1]['values'],{'a':50,'b':50})
        self.assertEqual(rows[3]['values'],{'a':100,'b':0})
        self.assertEqual(rows[4]['values'],{'a':10,'b':80})


if __name__=='__main__':unittest.main()
