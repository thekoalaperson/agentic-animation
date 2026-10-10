"""Pure stream-contract tests, no media or GPU operations."""
import unittest
from unittest.mock import patch
import hd_audio_remux as h


class PictureOnlyTests(unittest.TestCase):
    def test_silent_is_explicit_and_no_audio_hash_attempted(self):
        with patch.object(h,'streams',return_value=[{'codec_type':'video'}]), patch.object(h,'packet_hash') as digest:
            self.assertIsNone(h.source_audio_identity('input',True))
            digest.assert_not_called()

    def test_silent_cannot_accidentally_enter_audio_preserving_lane(self):
        with patch.object(h,'streams',return_value=[{'codec_type':'video'}]):
            with self.assertRaises(ValueError):h.source_audio_identity('input')

    def test_existing_audio_cannot_be_dropped(self):
        rows=[{'codec_type':'video'},{'codec_type':'audio'}]
        with patch.object(h,'streams',return_value=rows),patch.object(h,'packet_hash',return_value='verified'):
            with self.assertRaises(ValueError):h.source_audio_identity('input',True)
            self.assertEqual(h.source_audio_identity('input'),'verified')

    def test_extra_or_duplicate_streams_fail(self):
        for rows in [[{'codec_type':'video'}]*2,[{'codec_type':'video'},{'codec_type':'subtitle'}]]:
            with patch.object(h,'streams',return_value=rows):
                with self.assertRaises(ValueError):h.source_audio_identity('input',True)

if __name__=='__main__':unittest.main()
