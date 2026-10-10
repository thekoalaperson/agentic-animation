"""OOM responses require operator recovery, not an automatic request retry."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'finite_client', Path(__file__).parent / 'box' / 'finite_client.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class OOMRecoveryTest(unittest.TestCase):
    def test_nested_scheduler_error(self):
        self.assertTrue(client.requires_server_recovery({
            'error': {'message': 'Error executing request: CUDA out of memory.'}}))

    def test_exception_name(self):
        self.assertTrue(client.requires_server_recovery({
            'error': 'torch.OutOfMemoryError: allocation failed'}))

    def test_other_failure_keeps_normal_policy(self):
        self.assertFalse(client.requires_server_recovery({'error': 'invalid video'}))
        self.assertFalse(client.requires_server_recovery({'error': None}))


if __name__ == '__main__':
    unittest.main()
