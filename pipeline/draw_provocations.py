#!/usr/bin/env python3
"""Draw directorial provocations at random from data files beside this tool.

By default it draws two story-neutral constraints from provocations.txt. With
--axes it instead assigns one value to every axis of look, camera and cutting,
performance, and music and sound, from provocation_axes.txt. Read only this
tool's output, never the data files: a reader picks the most salient entries,
and operators on the same model pick the same ones. The draw uses the operating
system's random source, prints its result and writes nothing. Keep, combine,
override or reject what it draws; nothing records or validates a draw.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

DECK = Path(__file__).resolve().with_name("provocations.txt")
AXES = Path(__file__).resolve().with_name("provocation_axes.txt")


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


def read_axes(path: Path = AXES) -> dict[str, list[str]]:
    """Return each "group: axis" heading's values in file order."""
    axes: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("[") and entry.endswith("]"):
            current = entry[1:-1].strip()
            if ":" not in current or current in axes:
                raise ValueError(f"axis heading must be a new [group: axis]: {entry} in {path}")
            axes[current] = []
        elif current is None:
            raise ValueError(f"value before any axis heading: {entry} in {path}")
        elif entry in axes[current]:
            raise ValueError(f"axis {current} repeats a value: {path}")
        else:
            axes[current].append(entry)
    if not axes:
        raise ValueError(f"no axes: {path}")
    for name, values in axes.items():
        if len(values) < 2:
            raise ValueError(f"axis {name} needs at least two values: {path}")
    return axes


def draw_axes(axes: dict[str, list[str]],
              rng: random.Random | None = None) -> list[tuple[str, str]]:
    """Pick one value per axis, in heading order."""
    rng = rng or random.SystemRandom()
    return [(name, rng.choice(values)) for name, values in axes.items()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=2, help="cards to draw (default 2)")
    parser.add_argument("--deck", type=Path, default=DECK,
                        help="a deck of your own, one card per line")
    parser.add_argument("--axes", action="store_true",
                        help="assign one value per look, camera and cutting, performance "
                             "and music and sound axis instead of drawing cards")
    parser.add_argument("--axes-file", type=Path, default=AXES,
                        help="axis values of your own, as [group: axis] headings "
                             "each followed by one value per line")
    args = parser.parse_args(argv)
    try:
        if args.axes:
            lines = [f"{name.replace(':', ' /', 1)}: {value}"
                     for name, value in draw_axes(read_axes(args.axes_file))]
        else:
            lines = draw(read_deck(args.deck), args.count)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
