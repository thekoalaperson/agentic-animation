"""Small synthetic arrays and metadata only, never production media."""
import copy
import unittest
import numpy as np
from short_audio_contract import question_text, cta_filter, verify_whole_voice


class ShortContractTests(unittest.TestCase):
    def setUp(self):
        self.events = [dict(id=f's{i}', start=i/6, end=(i+1)/6, audio=f'{i}.wav', audioSha256=str(i),
                            audioSourceStart=0, audioSourceDuration=.05, text=f'line{i}')
                       for i in range(2)]
        self.short = copy.deepcopy(self.events)
        self.short[1]['start'] = 2/24
        self.spec = dict(totalFrames=6, bodyFrames=4, beats=[
            dict(id=f's{i}',sourceStartFrame=i*4,startFrame=i*2,frames=2,
                 audioDuration=.05,spokenTextHi=f'line{i}') for i in range(2)])
        self.source = np.zeros((16000,2),np.float32)
        self.source[0:2400] = .125
        self.source[8000:10400] = .25
        self.voice = np.zeros((12000,2),np.float32)
        self.voice[:4000] = self.source[:4000]
        self.voice[4000:8000] = self.source[8000:12000]

    def verify(self):
        return verify_whole_voice(self.spec,self.events,self.short,self.source,self.voice)

    def test_complete_voice_and_silent_cta(self):
        rows,error=self.verify()
        self.assertEqual(len(rows),2)
        self.assertEqual(error,0)

    def test_changed_or_cta_voice_rejected(self):
        for index in (100,9000):
            old=self.voice[index,0];self.voice[index,0]+=.01
            with self.assertRaises(ValueError):self.verify()
            self.voice[index,0]=old

    def test_trimmed_performance_rejected(self):
        self.spec['beats'][0]['frames']=1
        with self.assertRaises(ValueError):self.verify()

    def test_changed_identity_and_clock_rejected(self):
        self.short[0]['audioSha256']='wrong'
        with self.assertRaises(ValueError):self.verify()

    def test_reversed_source_order_rejected(self):
        self.spec['beats'].reverse()
        for i,b in enumerate(self.spec['beats']):b['startFrame']=i*2
        self.short.reverse()
        for i,e in enumerate(self.short):e['start']=i*2/24
        self.voice[:4000]=self.source[8000:12000]
        self.voice[4000:8000]=self.source[:4000]
        with self.assertRaisesRegex(ValueError, 'first complete performance'):self.verify()

    def test_source_window_spill_rejected(self):
        self.spec['beats'][0]['frames']=5
        self.spec['beats'][1]['startFrame']=5
        self.spec['bodyFrames']=7
        self.spec['totalFrames']=9
        self.short[1]['start']=5/24
        self.voice=np.zeros((18000,2),np.float32)
        self.voice[:10000]=self.source[:10000]
        self.voice[10000:14000]=self.source[8000:12000]
        with self.assertRaisesRegex(ValueError, 'crosses into'):self.verify()
        self.short=copy.deepcopy(self.events);self.short[1]['start']=2/24
        self.spec['beats'][1]['sourceStartFrame']=5
        with self.assertRaises(ValueError):self.verify()

    def test_question_omission_is_explicit(self):
        self.assertIsNone(question_text({}))
        self.assertIsNone(question_text({'question':None}))
        self.assertEqual(question_text({'question':'Why?'}),'Why?')
        for value in ('', ' ', 12):
            with self.assertRaises(ValueError):question_text({'question':value})

    def test_full_duration_cta_has_no_filter(self):
        self.assertIsNone(cta_filter(dict(ctaVisibility='full-duration',ctaFrames=156)))
        self.assertIn('fade',cta_filter(dict(ctaFrames=156)))
        with self.assertRaises(ValueError):cta_filter(dict(ctaFrames=96))


if __name__=='__main__':unittest.main()
