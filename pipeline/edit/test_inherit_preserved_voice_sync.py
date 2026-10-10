import unittest
from pipeline.edit.inherit_preserved_voice_sync import uniform_shift, derive_rows


class PreservedVoiceProofTests(unittest.TestCase):
    def test_uniform_priming_shift_and_extra_tail(self):
        self.assertAlmostEqual(uniform_shift([0,1,2],[.021,1.021,2.021,3])['min'],.021)

    def test_drift_and_missing_packets_fail(self):
        for after in ([0,1.1,2],[0,1]):
            with self.assertRaises(ValueError):
                uniform_shift([0,1,2],after)

    def test_inherited_values_remain_explicit(self):
        ref={'results':[{'event':'s1','lagSeconds':.021,'correlation':.9,'pass':True}],
             'maxAllowedLagSeconds':.05,'minimumAllowedCorrelation':.85}
        row=derive_rows(ref,{'min':-.00001,'max':.00001})[0]
        self.assertTrue(row['correlationIsInherited'])
        self.assertTrue(row['pass'])
        self.assertEqual(row['correlation'],.9)
        self.assertFalse(derive_rows(ref,{'min':.04,'max':.04})[0]['pass'])


if __name__ == '__main__':
    unittest.main()
