"""Pure working-clock and receipt checks; no production media processing."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import build_score_sfx_revision as sound


class WorkingRevisionTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(episode='DEMO', workingAudioOnly=True, finalApproved=False)
        self.clock = dict(episode='DEMO', kind='measured-speech-clock-not-renderable-edit-plan',
                          renderable=False, deliveryFps=24, runtimeSeconds=2,
                          eventCount=1, events=[dict(id='s1', shot=1, start=0, end=2, duration=2)])

    def test_working_and_final_are_separate(self):
        self.assertTrue(sound.validate_clock(self.config,self.clock))
        for field in ('workingAudioOnly','finalApproved'):
            c=copy.deepcopy(self.config);c.pop(field)
            with self.subTest(field=field), self.assertRaises(RuntimeError):sound.validate_clock(c,self.clock)
        c=copy.deepcopy(self.clock);c['renderable']=True
        with self.assertRaises(RuntimeError):sound.validate_clock(self.config,c)
        c['kind']='renderable-edit-plan'
        self.assertFalse(sound.validate_clock(dict(episode='DEMO'),c))

    def test_invalid_clock_is_rejected(self):
        for field,value in [('deliveryFps',25),('episode','OTHER'),('runtimeSeconds',2.001)]:
            c=copy.deepcopy(self.clock);c[field]=value
            with self.subTest(field=field), self.assertRaises(RuntimeError):sound.validate_clock(self.config,c)
        c=copy.deepcopy(self.clock);c['events'][0]['start']=.1
        with self.assertRaises(RuntimeError):sound.validate_clock(self.config,c)

    def test_stems_cannot_be_silently_truncated(self):
        for length in (1998,2002,4000):
            with self.subTest(length=length),self.assertRaises(RuntimeError):sound.exact_stem(np.zeros((length,2)),2000,'voice')
        for length in (1999,2000,2001):
            self.assertEqual(len(sound.exact_stem(np.zeros((length,2)),2000,'voice')),2000)

    def test_stale_receipts_are_rejected(self):
        c=dict(inputs={k:dict(path=k,sha256=k+'-hash') for k in ('clock','voice','sfx')})
        v=dict(kind='working-audio-stems-not-final',finalApproved=False,clockSha256='clock-hash',outputs=dict(voice=dict(sha256='voice-hash')))
        s=dict(workingAudioOnly=True,finalApproved=False,clockSha256='clock-hash',sha256='sfx-hash')
        for key in (None,'voice','sfx'):
            rows={'v':copy.deepcopy(v),'s':copy.deepcopy(s)}
            if key:rows['v' if key=='voice' else 's']['clockSha256']='stale'
            with patch.object(sound,'read',side_effect=lambda p: rows[p]):
                if key:
                    with self.assertRaises(RuntimeError):sound.validate_working_stems(c,dict(voiceAudit='v',sfxAudit='s'))
                else:sound.validate_working_stems(c,dict(voiceAudit='v',sfxAudit='s'))


if __name__=='__main__':unittest.main()
