from pathlib import Path
import unittest

import kie_video

SHEET = Path(__file__).resolve().parents[1] / "docs" / "VIDEO-MODELS.md"


class VideoModelSheetTest(unittest.TestCase):
    def test_every_kie_video_route_is_on_the_sheet(self):
        text = SHEET.read_text(encoding="utf-8")
        missing = [alias for alias in kie_video.MODELS if f"`{alias}`" not in text]
        self.assertEqual(missing, [], "add a dated line for each route to docs/VIDEO-MODELS.md")


if __name__ == "__main__":
    unittest.main()
