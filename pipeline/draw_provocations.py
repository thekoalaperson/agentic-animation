#!/usr/bin/env python3
"""Draw directorial provocations at random from the deck beside this tool.

Run this rather than reading the deck: a reader picks the most salient cards,
and operators on the same model pick the same ones. The draw uses the
operating system's random source, prints the cards and writes nothing. Reject,
combine or override what it draws; nothing records or validates a draw.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

DECK = Path(__file__).resolve().with_name("provocations.txt")


def read_deck(path: Path = DECK) -> list[str]:
    """Return the deck's cards in file order, ignoring blank and # lines."""
    cards = []
    for line in path.read_text(encoding="utf-8").splitlines():
        card = line.strip()
        if card and not card.startswith("#"):
            cards.append(card)
    if len(set(cards)) != len(cards):
        raise ValueError(f"deck repeats a card: {path}")
    return cards


def draw(cards: list[str], count: int = 2, rng: random.Random | None = None) -> list[str]:
    """Draw distinct cards without replacement."""
    if not 1 <= count <= len(cards):
        raise ValueError(f"count must be between 1 and {len(cards)}")
    return (rng or random.SystemRandom()).sample(cards, count)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2, help="cards to draw (default 2)")
    parser.add_argument("--deck", type=Path, default=DECK,
                        help="a deck of your own, one card per line")
    args = parser.parse_args(argv)
    try:
        cards = draw(read_deck(args.deck), args.count)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for card in cards:
        print(card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
