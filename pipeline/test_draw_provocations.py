import contextlib
import io
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import draw_provocations as tool


class DeckTest(unittest.TestCase):
    def test_committed_deck_is_large_and_has_no_repeats(self):
        cards = tool.read_deck()
        self.assertGreaterEqual(len(cards), 25)
        self.assertEqual(len(cards), len(set(cards)))
        self.assertFalse(any(card.startswith("#") for card in cards))

    def test_draw_is_distinct_cards_from_the_deck(self):
        cards = tool.read_deck()
        drawn = tool.draw(cards, 2, random.Random(7))
        self.assertEqual(len(drawn), 2)
        self.assertEqual(len(set(drawn)), 2)
        self.assertTrue(set(drawn) <= set(cards))

    def test_default_source_is_the_system_random(self):
        seen = []

        class Recorder(random.SystemRandom):
            def sample(self, population, k):
                seen.append(k)
                return super().sample(population, k)

        with mock.patch.object(tool.random, "SystemRandom", Recorder):
            tool.draw(["a", "b", "c"])
        self.assertEqual(seen, [2])

    def test_count_outside_the_deck_is_refused(self):
        for count in (0, 4):
            with self.assertRaises(ValueError):
                tool.draw(["a", "b", "c"], count)

    def test_repeated_card_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            deck = Path(directory) / "deck.txt"
            deck.write_text("# note\nsame\n\nsame\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                tool.read_deck(deck)

    def test_main_prints_only_the_draw_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            deck = Path(directory) / "deck.txt"
            deck.write_text("one\ntwo\nthree\n", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(tool.main(["--deck", str(deck), "--count", "3"]), 0)
            self.assertEqual(sorted(out.getvalue().splitlines()), ["one", "three", "two"])
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["deck.txt"])

    def test_committed_axes_are_wide_and_cover_every_group(self):
        axes = tool.read_axes()
        groups = {name.split(":", 1)[0] for name in axes}
        self.assertEqual(groups, {"look", "camera and cutting", "performance", "music and sound"})
        for name, values in axes.items():
            self.assertGreaterEqual(len(values), 8, name)
            self.assertEqual(len(values), len(set(values)), name)

    def test_axes_draw_one_value_per_axis_and_a_seed_repeats_it(self):
        axes = tool.read_axes()
        first = tool.draw_axes(axes, random.Random(11))
        self.assertEqual([name for name, _ in first], list(axes))
        for name, value in first:
            self.assertIn(value, axes[name])
        self.assertEqual(tool.draw_axes(axes, random.Random(11)), first)

    def test_axes_default_source_is_the_system_random(self):
        seen = []

        class Recorder(random.SystemRandom):
            def choice(self, seq):
                seen.append(len(seq))
                return super().choice(seq)

        with mock.patch.object(tool.random, "SystemRandom", Recorder):
            tool.draw_axes({"a: x": ["1", "2"], "a: y": ["3", "4", "5"]})
        self.assertEqual(seen, [2, 3])

    def test_malformed_axes_files_are_refused(self):
        bad = {
            "value before heading": "loose\n[a: x]\n1\n2\n",
            "heading without group": "[x]\n1\n2\n",
            "repeated heading": "[a: x]\n1\n2\n[a: x]\n3\n4\n",
            "repeated value": "[a: x]\n1\n1\n",
            "single value": "[a: x]\n1\n",
            "empty": "# only a note\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "axes.txt"
            for label, text in bad.items():
                path.write_text(text, encoding="utf-8")
                with self.subTest(label), self.assertRaises(ValueError):
                    tool.read_axes(path)

    def test_main_axes_prints_one_line_per_axis_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "axes.txt"
            path.write_text("# note\n[look: medium]\nink\nclay\n\n[music: density]\nsparse\nfull\n",
                            encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(tool.main(["--axes", "--axes-file", str(path)]), 0)
            lines = out.getvalue().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertRegex(lines[0], r"^look / medium: (ink|clay)$")
            self.assertRegex(lines[1], r"^music / density: (sparse|full)$")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["axes.txt"])

    def test_help_runs_anywhere(self):
        result = subprocess.run([sys.executable, str(Path(tool.__file__)), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--count", result.stdout)
        self.assertIn("--axes", result.stdout)
        self.assertIn("never the data files", result.stdout)


if __name__ == "__main__":
    unittest.main()
