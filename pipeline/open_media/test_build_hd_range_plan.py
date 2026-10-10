import unittest
from build_hd_range_plan import make_ranges


class Ranges(unittest.TestCase):
    def row(self, ident, start, count):
        return dict(id=ident, timelineStart=start/24, frames=count, memberSegmentIds=[ident])

    def test_long_continuous_take_is_split_without_losing_frames(self):
        rows, total = make_ranges([self.row('p1', 0, 175), self.row('circle', 175, 1045), self.row('close', 1220, 165)])
        self.assertEqual(total, 1385)
        self.assertEqual([r['frames'] for r in rows], [175, 600, 445, 165])
        self.assertEqual([r['startFrame'] for r in rows], [0, 175, 775, 1220])
        self.assertEqual(rows[2]['parentRunOffsetFrames'], 600)
        self.assertEqual(rows[1]['memberSegmentIds'], ['circle'])

    def test_bad_clocks_and_identity_rejected(self):
        for rows in ([], [self.row('p1', 1, 10)], [self.row('p1', 0, 0)],
                     [self.row('p1', 0, 10), self.row('p1', 10, 5)]):
            with self.assertRaises(ValueError): make_ranges(rows)


if __name__ == '__main__': unittest.main()
