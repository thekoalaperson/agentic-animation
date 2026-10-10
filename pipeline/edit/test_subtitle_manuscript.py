import copy
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.edit.subtitle_manuscript import apply_manuscript, digest, text_digest


class SubtitleManuscriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.source=self.root/'source.json'; self.source.write_text('{}')
        self.path=self.root/'manuscript.json'
        self.effective={'s001':{'hi':'वह हठ था।','en':'It was persistence.'}}
        self.doc={'schemaVersion':1,'kind':'subtitle-text-manuscript',
            'sourceBindings':[{'path':'source.json','sha256':digest(self.source)}],
            'items':[{'id':'s001','sourceTextSha256':text_digest('वह हठ था।','It was persistence.'),
                'hi':'वह हठ था।','en':'It was obstinacy.','editReason':'Preserve हठ.',
                'cuePairs':[{'hi':'वह हठ था।','en':'It was obstinacy.'}],'cueWeights':[3]}]}

    def tearDown(self): self.tmp.cleanup()

    def apply(self,doc=None):
        self.path.write_text(json.dumps(doc or self.doc,ensure_ascii=False))
        return apply_manuscript(self.root,self.path,digest(self.path),self.effective)

    def test_subtitle_only_change_preserves_hindi_and_source(self):
        output=self.apply()
        self.assertEqual(output['s001']['en'],'It was obstinacy.')
        self.assertEqual(output['s001']['hi'],self.effective['s001']['hi'])
        self.assertEqual(self.effective['s001']['en'],'It was persistence.')

    def test_rejects_hindi_drift_empty_english_missing_reason_and_bad_pairs(self):
        mutations=[('hi','वह बदल गया।'),('en',''),('editReason',''),
            ('cuePairs',[{'hi':'वह हठ','en':'It was obstinacy.'}]),('cueWeights',[0]),
            ('cueWeights',None),('sourceTextSha256','a'*64)]
        for key,value in mutations:
            doc=copy.deepcopy(self.doc); doc['items'][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError): self.apply(doc)

    def test_rejects_missing_duplicate_and_stale_source(self):
        for items in ([],self.doc['items']*2):
            doc=copy.deepcopy(self.doc); doc['items']=items
            with self.assertRaisesRegex(ValueError,'every scene exactly once'): self.apply(doc)
        self.source.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'source changed'): self.apply()

    def test_rejects_manuscript_hash_and_source_escape(self):
        self.path.write_text(json.dumps(self.doc))
        with self.assertRaisesRegex(ValueError,'hash changed'):
            apply_manuscript(self.root,self.path,'0'*64,self.effective)
        doc=copy.deepcopy(self.doc); doc['sourceBindings'][0]['path']='../not-in-episode'
        with self.assertRaises(ValueError): self.apply(doc)


if __name__=='__main__': unittest.main()
