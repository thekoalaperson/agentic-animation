#!/usr/bin/env python3
"""Still reuse is listed as information, and staging notes caption the sheets."""

import unittest

import build_episode_timeline as timeline
import prepare_episode_production as production


class StillReuseTests(unittest.TestCase):
    def test_reuse_is_information_with_a_consecutive_flag(self):
        a, b = "a" * 64, "b" * 64
        dialogues = {
            "demo-s001": {"source": "art/stills/one.png", "sourceSha256": a},
            "demo-s002": {"source": "art/stills/one.png", "sourceSha256": a},
            "demo-s003": {"source": "art/stills/two.png", "sourceSha256": b},
            "demo-s004": {"source": "art/stills/three.png", "sourceSha256": "c" * 64},
            "demo-s005": {"source": "art/stills/two.png", "sourceSha256": b},
        }
        reuse = timeline._still_reuse(dialogues)
        self.assertEqual(reuse, [
            {"source": "art/stills/one.png", "sourceSha256": a,
             "jobs": ["demo-s001", "demo-s002"], "consecutive": True},
            {"source": "art/stills/two.png", "sourceSha256": b,
             "jobs": ["demo-s003", "demo-s005"], "consecutive": False},
        ])
        self.assertTrue(all("reason" not in row for row in reuse))
        self.assertEqual(timeline._still_reuse({"demo-s001": {"source": "x.png"}}), [])


class StagingCaptionTests(unittest.TestCase):
    def test_caption_names_speaker_and_staging(self):
        row = {"id": "demo-s003", "speaker": "Demo One"}
        staging = {"line": "keep", "screen_side": "left", "gaze": "to the door", "extra": ""}
        first, second = production.sheet_caption(row, staging)
        self.assertEqual(first, "demo-s003 | Demo One")
        self.assertEqual(second, "screen side: left / gaze: to the door / line: keep")
        self.assertEqual(production.sheet_caption({"id": "demo-s004", "speaker": ""}),
                         ("demo-s004", ""))
        crossing = production.staging_summary({"line": {"cross": "yes", "reason": "the turn"}})
        self.assertEqual(crossing, "line: yes the turn")
        self.assertTrue(production.staging_summary({"gaze": "x" * 200}).endswith("…"))
        self.assertEqual(production.staging_summary("not an object"), "")


if __name__ == "__main__":
    unittest.main()
