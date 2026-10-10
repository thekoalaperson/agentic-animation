"""Contract logic tests; these are not evidence that actual finals exist."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

if __package__:
    from . import verify_hd_delivery as delivery
else:
    import verify_hd_delivery as delivery


class DeliveryContractTest(unittest.TestCase):
    def setUp(self):
        self.streams = [
            {'codec_type':'video','width':1920,'height':1080,'avg_frame_rate':'48/1',
             'nb_read_frames':'480','duration':'10.0'},
            {'codec_type':'audio','codec_name':'aac','sample_rate':'48000',
             'channels':2,'duration':'10.0'}]

    def check(self, streams=None, file_hash='expected'):
        with patch.object(delivery,'sha',return_value=file_hash), \
             patch.object(delivery,'output',return_value=json.dumps({'streams':streams or self.streams})), \
             patch.object(delivery.subprocess,'run') as decode:
            result = delivery.verify_media(Path('/tmp/test.mp4'),'expected',[1920,1080],480,48,10)
            self.assertIn('-xerror',decode.call_args.args[0])
            return result

    def test_complete_contract(self):
        self.assertEqual(self.check()['decode'],'passed')

    def test_hash_mismatch(self):
        with self.assertRaises(ValueError): self.check(file_hash='different')

    def test_video_contracts(self):
        for field,value in [('width',1280),('height',720),('avg_frame_rate','24/1'),
                            ('nb_read_frames','479'),('duration','9.9')]:
            with self.subTest(field=field):
                streams=copy.deepcopy(self.streams);streams[0][field]=value
                with self.assertRaises(ValueError): self.check(streams)

    def test_audio_contracts(self):
        for field,value in [('codec_name','mp3'),('channels',1),('sample_rate','44100'),('duration','9.8')]:
            with self.subTest(field=field):
                streams=copy.deepcopy(self.streams);streams[1][field]=value
                with self.assertRaises(ValueError): self.check(streams)

    def test_missing_or_extra_stream(self):
        for streams in [self.streams[:1],self.streams+[{'codec_type':'subtitle'}]]:
            with self.assertRaises(ValueError): self.check(streams)


if __name__=='__main__': unittest.main()
