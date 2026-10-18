"""Offline request-envelope regressions and faithful test fixtures."""
import copy
import hashlib
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dialogue_provenance import canonical_sha, validate_dialogue_request


def bind_fixture(record, prompt_text, references=()):
    """Build the nested fields emitted by kie_video.py, without any network."""
    local = {key: record[key] for key in (
        'jobId', 'model', 'sourceSha256', 'promptSha256', 'executionManifestSha256',
        'executionRequestSha256', 'resolution', 'aspectRatio', 'durationSeconds', 'audio')}
    local.update(version=1, referenceOnly=False, multiClip=False,
                 inputHashes=dict(primary=record['sourceSha256'], last=None, references=list(references),
                                     videos=[]))
    record['localRequest'] = local
    record['localRequestSha256'] = canonical_sha(local)
    request = {key: record[key] for key in ('model', 'sourceSha256', 'promptSha256',
        'executionManifestSha256', 'executionRequestSha256', 'localRequestSha256')}
    request.update(version=1, promptTextSha256=hashlib.sha256(prompt_text.strip().encode()).hexdigest(),
        input=dict(prompt=prompt_text.strip(), duration=record['durationSeconds'],
                   resolution=record['resolution'], aspect_ratio=record['aspectRatio']))
    record.update(request=request, requestSha256=canonical_sha(request),
                  requestFingerprint=canonical_sha(request))
    return record


class DialogueEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(jobId='demo-s117', model='grok-imagine-video-1-5-preview',
            sourceSha256='a'*64, promptSha256='b'*64, executionManifestSha256='c'*64,
            executionRequestSha256='d'*64, resolution='720p', aspectRatio='16:9',
            durationSeconds=10, audio=True)
        self.record = bind_fixture(copy.deepcopy(self.expected), 'A corrected action.\n')

    def test_valid_and_missing_request(self):
        validate_dialogue_request(self.record, self.expected, 'A corrected action.\n')
        del self.record['request']
        with self.assertRaisesRegex(ValueError, 'nested request'):
            validate_dialogue_request(self.record, self.expected, 'A corrected action.\n')

    def test_retopping_old_request_is_rejected(self):
        old = copy.deepcopy(self.expected)
        old['executionManifestSha256'] = 'e'*64
        old['promptSha256'] = 'f'*64
        record = bind_fixture(old, 'Old action.')
        record.update(self.expected)
        with self.assertRaisesRegex(ValueError, 'nested request mismatch'):
            validate_dialogue_request(record, self.expected, 'A corrected action.\n')

    def test_rehashed_wrong_provider_prompt_is_rejected(self):
        self.record['request']['input']['prompt'] = 'Wrong action.'
        digest = canonical_sha(self.record['request'])
        self.record.update(requestSha256=digest, requestFingerprint=digest)
        with self.assertRaisesRegex(ValueError, 'provider input mismatch: prompt'):
            validate_dialogue_request(self.record, self.expected, 'A corrected action.\n')

    def test_frozen_references_must_match_in_order(self):
        refs = ['e'*64, 'f'*64]
        record = bind_fixture(copy.deepcopy(self.expected), 'A corrected action.\n', refs)
        validate_dialogue_request(record, {**self.expected, 'referenceSha256s': refs},
                                  'A corrected action.\n')
        for wrong in ([], refs[::-1], refs[:1]):
            with self.subTest(expected=wrong), self.assertRaisesRegex(ValueError, 'references'):
                validate_dialogue_request(record, {**self.expected, 'referenceSha256s': wrong},
                                          'A corrected action.\n')
        with self.assertRaisesRegex(ValueError, 'references'):
            validate_dialogue_request(self.record, {**self.expected, 'referenceSha256s': refs},
                                      'A corrected action.\n')


if __name__ == '__main__':
    unittest.main()
