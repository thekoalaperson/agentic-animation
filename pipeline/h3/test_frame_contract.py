"""Frame-contract mistakes halt, never enter automatic generation retries."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('finite_client', Path(__file__).parent / 'box' / 'finite_client.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

class FrameContractTest(unittest.TestCase):
    def test_matching_frames(self):
        client.verify_frame_contract({'nb_read_frames': '277'}, {'expectedFrames': 277})

    def test_mismatch_stops_instead_of_retry(self):
        with self.assertRaises(client.StopRun):
            client.verify_frame_contract({'nb_read_frames': '294'}, {'expectedFrames': 301})
        self.assertFalse(issubclass(client.StopRun, client.RejectedTake))

if __name__ == '__main__':
    unittest.main()
