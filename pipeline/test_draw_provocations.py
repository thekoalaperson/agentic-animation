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

    def test_help_runs_anywhere(self):
        result = subprocess.run([sys.executable, str(Path(tool.__file__)), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--count", result.stdout)


if __name__ == "__main__":
    unittest.main()
