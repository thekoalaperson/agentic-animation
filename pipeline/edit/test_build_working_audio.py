"""Pure metadata checks: no media, subprocess, network or cloud execution."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import unittest.mock

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



class AuditionTests(unittest.TestCase):
    def test_audition_receipts_get_their_own_kinds(self):
        import build_working_audio as working
        for kind in ('working-audio-stems-not-final','working-audio-context-not-final'):
            record=working.label_audition({'kind':kind},True,{'computeHost':'demo'})
            self.assertTrue(record['kind'].endswith('-audition-not-final'))
            self.assertIs(record['admissible'],False)
            self.assertEqual(working.label_audition({'kind':kind},False,None),{'kind':kind})
        self.assertEqual(working.audition_output('work/mix-v1',True),'work/mix-v1-audition-not-final')

    def test_admission_steps_refuse_audition_receipts(self):
        import tempfile, json
        import build_working_audio as working
        import build_score_sfx_revision as revision
        import short_audio_contract as short
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            stems=working.label_audition({'kind':'working-audio-stems-not-final','finalApproved':False,
                                          'outputs':{'voice':{'sha256':'a'*64}}},True,{})
            (root/'voice.json').write_text(json.dumps(stems));(root/'sfx.json').write_text('{}')
            config={'inputs':{'clock':{'sha256':'c'*64},'voice':{'sha256':'a'*64}}}
            with self.assertRaisesRegex(RuntimeError,'working admission provenance'):
                revision.validate_working_stems(config,{'voiceAudit':root/'voice.json','sfxAudit':root/'sfx.json'})
            mix=working.label_audition({'kind':'working-audio-context-not-final','voiceGain':1,
                                        'unchangedVoiceRecoveryMaxError':0.0},True,{})
            for name,value in (('mix',mix),('voice',stems),('source',{})):
                (root/f'{name}.json').write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'linear whole-voice provenance'):
                short.prepared_audio({'inputs':{}},{'mixAudit':root/'mix.json','voiceAudit':root/'voice.json',
                                                    'sourceVoiceAudit':root/'source.json'},None)

    def test_working_mix_refuses_audition_stems(self):
        import build_working_audio as working
        config={'inputs':{k:{'sha256':k[0]*64} for k in ('voice','music','sfx','clock')}}
        sfx={'workingAudioOnly':True,'finalApproved':False,'sha256':'s'*64,'clockSha256':'c'*64}
        stems={'kind':'working-audio-stems-not-final','finalApproved':False,
               'outputs':{'voice':{'sha256':'v'*64},'music':{'sha256':'m'*64}}}
        working.check_working_receipts(config,stems,sfx)
        audition=working.label_audition(copy.deepcopy(stems),True,{})
        with self.assertRaisesRegex(RuntimeError,'Only explicitly working stems'):
            working.check_working_receipts(config,audition,sfx)

    def test_short_refuses_an_audition_voice_beside_a_real_mix(self):
        import tempfile, json
        import build_working_audio as working
        import short_audio_contract as short
        mix={'kind':'working-audio-context-not-final','voiceGain':1,'unchangedVoiceRecoveryMaxError':0.0}
        stems={'kind':'working-audio-stems-not-final','finalApproved':False,'clockSha256':'c'*64,
               'outputs':{'voice':{'sha256':'v'*64}}}
        short_voice={'clockSha256':'c'*64,'outputs':{'voice':{'sha256':'v'*64}},'finalApproved':False}
        audition=working.label_audition(copy.deepcopy(stems),True,{})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def call(voice,source):
                for name,value in (('mix',mix),('voice',voice),('source',source)):
                    (root/f'{name}.json').write_text(json.dumps(value))
                return short.prepared_audio({'inputs':{}},{'mixAudit':root/'mix.json','voiceAudit':root/'voice.json',
                                                           'sourceVoiceAudit':root/'source.json'},None)
            for voice,source,name in ((audition,stems,'voice'),(short_voice,audition,'source voice')):
                with self.subTest(name=name),self.assertRaisesRegex(ValueError,f'Short {name} receipt is an audition'):
                    call(voice,source)
            # A production pair, including the Short writer's kind-less voice
            # receipt, gets past the audition check to the binding check.
            with self.assertRaises(KeyError):
                call(short_voice,stems)

    def test_production_run_keeps_its_gate(self):
        import build_working_audio as working
        with unittest.mock.patch.object(working.sys,'platform','darwin'),self.assertRaisesRegex(RuntimeError,'qualified GPU worker'):
            working.main(['/nonexistent','/nonexistent.json'])

    def test_working_music_keeps_its_gate(self):
        import tempfile, json
        import render_working_music as music
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'config.json';config.write_text(json.dumps({'finalApproved':False}))
            with unittest.mock.patch.object(music.platform,'system',return_value='Darwin'),self.assertRaisesRegex(RuntimeError,'GCP'):
                music.main([d,str(config)])
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
