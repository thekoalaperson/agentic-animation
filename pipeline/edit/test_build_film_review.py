import json
import unittest
from pathlib import Path

import build_film_review as review

TEMPLATE = (Path(__file__).resolve().parent / "film_review_template.html").read_text()


def working_cut(**extra):
    return {"episode": "DEMO", "title": "demo working cut", "reviewKey": "demo-1",
            "reviewStatus": "Working cut; not final", "events": [{"id": "demo-s001", "start": 0, "end": 2}],
            "files": {"720": {"path": "demo-720.mp4", "sha256": "a" * 64}}, **extra}


class FilmReviewTest(unittest.TestCase):
    def test_a_working_cut_alone_is_enough(self):
        page = review.render(working_cut(), TEMPLATE)
        self.assertIn('"reviewKey": "demo-1"', page)
        self.assertNotIn("__FILM_REVIEW_DATA__", page)

    def test_coverage_lines_travel_with_the_page(self):
        data = working_cut(reviewSummary={"dialogue": {"line": "dialogue: 1 admitted"}},
                           checked=[{"what": "every line against the script", "method": "transcript"}],
                           notChecked=["lip sync at full speed"])
        page = review.render(data, TEMPLATE)
        payload = json.loads(page.split("const DATA=", 1)[1].split(";\n", 1)[0])
        self.assertEqual(payload["notChecked"], ["lip sync at full speed"])

    def test_refusals_name_what_is_missing(self):
        cases = {
            "working cut": working_cut(files={"short": {"path": "s.mp4", "sha256": "b" * 64}}),
            "shortEvents": working_cut(files={"720": {"path": "a.mp4", "sha256": "a" * 64},
                                              "short": {"path": "s.mp4", "sha256": "b" * 64}}),
            "sha256": working_cut(files={"1080": {"path": "a.mp4", "sha256": "short"}}),
            "needs title": {k: v for k, v in working_cut().items() if k != "title"},
            "unknown review files": working_cut(files={"720": {"path": "a.mp4", "sha256": "a" * 64},
                                                       "4k": {"path": "b.mp4", "sha256": "b" * 64}}),
        }
        for message, data in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                review.render(data, TEMPLATE)


if __name__ == "__main__":
    unittest.main()
