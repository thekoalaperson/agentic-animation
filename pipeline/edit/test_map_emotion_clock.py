import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import map_emotion_clock
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


def demo_clock():
    return {'episode':'DEMO','runtimeSeconds':30,'events':[
        {'id':f'demo-s00{i}','shot':i,'start':i*5,'end':i*5+5,'kind':'dialogue' if i%2 else 'narration'}
        for i in range(6)]}


def demo_anchors(**extra):
    return {'dimensions':['demo-pull','demo-calm'],'anchors':[
        {'shotId':'demo-s000','values':{'demo-pull':0,'demo-calm':100},'musicEnergy':10},
        {'shotId':'demo-s004','values':{'demo-pull':80,'demo-calm':20},'musicEnergy':90}],**extra}


class MusicEnergyTest(unittest.TestCase):
    def test_without_a_spec_the_output_is_unchanged(self):
        result=map_clock(demo_clock(),demo_anchors())
        self.assertEqual(set(result),{'kind','episode','dimensions','runtimeSeconds','anchors','shots',
                                      'method','listeningVerdict'})
        self.assertEqual(set(result['shots'][0]),{'id','shot','start','end','values'})
        self.assertEqual(set(result['anchors'][0]),{'time','shot','values'})

    def test_music_energy_follows_a_dimension_or_its_own_anchors(self):
        rows=map_clock(demo_clock(),demo_anchors(musicEnergy={'fromDimension':'demo-pull'}))['shots']
        self.assertEqual([r['music_energy'] for r in rows],[r['values']['demo-pull'] for r in rows])
        rows=map_clock(demo_clock(),demo_anchors(musicEnergy={'perAnchor':True}))['shots']
        self.assertEqual((rows[0]['music_energy'],rows[2]['music_energy'],rows[5]['music_energy']),(10,50,90))
        for spec in ({'fromDimension':'demo-other'},{'perAnchor':False},'demo-pull'):
            with self.subTest(spec=spec),self.assertRaises(ValueError):
                map_clock(demo_clock(),demo_anchors(musicEnergy=spec))

    def test_the_written_wave_passes_the_mixer_and_feeds_the_assembler(self):
        import assemble_longform
        import mix_absolute_sfx_longform as mixer
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan={**demo_clock(),'scorePlacements':[]}
            files={name:root/f'{name}.json' for name in ('editPlan','anchors','dramaticWave',
                   'voiceLevels','musicConform','musicWaveAudit','pictureSegments','pictureRenderRuns')}
            files['editPlan'].write_text(json.dumps(plan))
            files['anchors'].write_text(json.dumps(demo_anchors(musicEnergy={'perAnchor':True})))
            argv=['map_emotion_clock.py',str(files['editPlan']),str(files['anchors']),str(files['dramaticWave'])]
            with mock.patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                map_emotion_clock.main()
            wave=json.loads(files['dramaticWave'].read_text())
            self.assertEqual(wave['clockSha256'],hashlib.sha256(files['editPlan'].read_bytes()).hexdigest())
            _,audit=assemble_longform.wave_envelope(plan,wave,30*48000)
            files['musicWaveAudit'].write_text(json.dumps(audit))
            files['voiceLevels'].write_text(json.dumps([{'event':e['id']} for e in plan['events']]))
            files['musicConform'].write_text('[]')
            files['pictureSegments'].write_text(json.dumps([{'id':'demo-p1'}]))
            files['pictureRenderRuns'].write_text(json.dumps([{'memberSegmentIds':['demo-p1'],'frames':720}]))
            mixer.validate_assembly_evidence(root,plan,30.0,720,files)
            # The mixer still refuses a wave bound to another clock or out of shot order.
            for broken in ({**wave,'clockSha256':'0'*64},{**wave,'shots':wave['shots'][::-1]}):
                files['dramaticWave'].write_text(json.dumps(broken))
                with self.assertRaises(mixer.MixError):
                    mixer.validate_assembly_evidence(root,plan,30.0,720,files)


if __name__=='__main__':unittest.main()
