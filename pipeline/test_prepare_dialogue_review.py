import unittest
from pipeline.prepare_dialogue_review import select_jobs


class SubsetReviewTest(unittest.TestCase):
    def setUp(self):
        self.plan = {"jobs": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}

    def test_full_default(self):
        self.assertEqual(select_jobs(self.plan, None, 3), self.plan["jobs"])

    def test_explicit_subset_keeps_source_order_and_original(self):
        self.assertEqual(select_jobs(self.plan, "c,a", 2), [{"id": "a"}, {"id": "c"}])
        self.assertEqual(len(self.plan["jobs"]), 3)

    def test_fail_closed(self):
        for ids, count in [("a,a", 2), ("a,z", 2), ("a,b", 3), (None, 2), ("", 0)]:
            with self.subTest(ids=ids, count=count), self.assertRaises(ValueError):
                select_jobs(self.plan, ids, count)


if __name__ == "__main__":
    unittest.main()
