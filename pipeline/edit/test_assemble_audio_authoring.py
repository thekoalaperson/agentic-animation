"""Authored audio choices in the long-form assembler: silence, voice gain, music shaping."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.edit import assemble_longform as assembly


class AudioAuthoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.ep = Path(self.temporary.name) / "demo-film"
        self.work = self.ep / "work/assembly-v1"
        self.work.mkdir(parents=True)
        self.written = {}

    def tearDown(self):
        self.temporary.cleanup()

    def capture(self, path, samples):
        self.written[Path(path).name] = samples.copy()
        Path(path).write_bytes(b"captured")

    def events(self, kinds=("narration", "dialogue", "narration", "narration")):
        return [{"shot": i + 1, "id": f"demo-s{i + 1:03d}", "kind": kind,
                 "start": float(i), "end": float(i + 1), "duration": 1.0}
                for i, kind in enumerate(kinds)]

    def wave(self):
        return {"shots": [{"shot": i + 1, "music_energy": float(i)} for i in range(4)]}

    def test_silence_placements_decode_nothing_and_keep_one_conform_row_each(self):
        plan = {"runtimeSeconds": 4.0, "events": self.events(), "musicBreaths": [],
                "scorePlacements": [{"movement": "hush", "silence": True, "timelineStart": 0.0,
                                     "timelineDuration": 4.0, "reason": "The film opens unscored."}]}
        with patch.object(assembly, "SR", 24), patch.object(assembly, "write_audio", self.capture), \
                patch.object(assembly, "audio", side_effect=AssertionError("nothing to decode")):
            assembly.build_music(self.ep, self.work, plan, self.wave())
        self.assertFalse(np.any(self.written["music-wave-48k24.wav"]))
        conform = json.loads((self.work / "music-conform.json").read_text())
        self.assertEqual(conform, [{"movement": "hush", "silence": True, "timelineStart": 0.0,
                                    "timelineDuration": 4.0, "reason": "The film opens unscored.",
                                    "repriseJoins": []}])


if __name__ == "__main__":
    unittest.main()
