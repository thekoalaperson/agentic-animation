import unittest
from unittest.mock import patch
import numpy as np
from pipeline.edit import assemble_longform as music

class MusicEnvelopeTests(unittest.TestCase):
    def inputs(self, breaths):
        plan={'events':[{'shot':i+1,'id':str(i+1),'kind':'narration','start':float(i),'end':float(i+1)} for i in range(4)],'musicBreaths':breaths}
        wave={'shots':[{'shot':i+1,'music_energy':float(i)} for i in range(4)]}
        return plan,wave
    def test_linear_smoothing_matches_convolution(self):
        for rate in [20,24,30]:
            plan,wave=self.inputs([])
            with patch.object(music,'SR',rate): actual,_=music.wave_envelope(plan,wave,rate*4)
            raw=np.concatenate([np.full(rate,0.14+0.82*(i/3)**1.55,np.float32) for i in range(4)])
            n=round(.3*rate)
            expected=np.convolve(raw,np.ones(n,np.float32)/n,mode='same')
            np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=1e-7)
    def test_absolute_vacuum_has_no_boundary_leak(self):
        plan,wave=self.inputs([{'shots':[2,2],'level':0,'reason':'test vacuum'}])
        with patch.object(music,'SR',24): actual,_=music.wave_envelope(plan,wave,96)
        self.assertTrue(np.all(actual[24:48]==0))
        self.assertGreater(float(actual[23]),0)
        self.assertGreater(float(actual[48]),0)

if __name__=='__main__': unittest.main()
