"""评估指标测试（指标算错，整套评估就白做）。"""

import unittest

from eval.metrics import (
    aggregate,
    coverage_at_k,
    diff,
    first_hit_rank,
    recall_at_k,
    reciprocal_rank,
)


class MetricTest(unittest.TestCase):
    def test_first_hit_rank(self):
        self.assertEqual(first_hit_rank(["a", "b", "c"], ["b"]), 2)
        self.assertIsNone(first_hit_rank(["a", "b"], ["z"]))
        self.assertIsNone(first_hit_rank(["a"], []))

    def test_recall_at_k(self):
        ranked = ["a", "b", "c"]
        self.assertEqual(recall_at_k(ranked, ["b"], 1), 0)
        self.assertEqual(recall_at_k(ranked, ["b"], 2), 1)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(["a", "b"], ["b"]), 0.5)
        self.assertEqual(reciprocal_rank(["a"], ["z"]), 0.0)

    def test_coverage(self):
        self.assertEqual(coverage_at_k(["a", "b"], ["a", "b"], 2), 1.0)
        self.assertEqual(coverage_at_k(["a", "x"], ["a", "b"], 2), 0.5)
        self.assertEqual(coverage_at_k([], ["a"], 5), 0.0)


class AggregateTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "q1", "type": "single", "expected": ["a"],
             "ranked": ["a", "b"], "top_score": 0.03},
            {"id": "q2", "type": "paraphrase", "expected": ["c"],
             "ranked": ["d", "e", "c"], "top_score": 0.03},
            {"id": "q3", "type": "negative", "expected": [],
             "ranked": ["z"], "top_score": 0.01},
        ]

    def test_overall(self):
        report = aggregate(self.rows, ks=(1, 5), floor=0.02)
        self.assertEqual(report["overall"]["n"], 2)      # 负例不参与 recall
        self.assertEqual(report["overall"]["recall@1"], 0.5)
        self.assertEqual(report["overall"]["recall@5"], 1.0)
        # (1/1 + 1/3) / 2 = 0.6667
        self.assertEqual(report["overall"]["mrr"], 0.6667)
        self.assertEqual(report["overall"]["miss"], 0)

    def test_negative_abstain(self):
        report = aggregate(self.rows, ks=(1,), floor=0.02)
        self.assertEqual(report["negative"]["abstain_rate"], 1.0)

    def test_by_type(self):
        report = aggregate(self.rows, ks=(1,), floor=0.02)
        self.assertIn("single", report["by_type"])
        self.assertIn("paraphrase", report["by_type"])

    def test_empty(self):
        report = aggregate([], ks=(1,))
        self.assertEqual(report["overall"], {})

    def test_diff(self):
        baseline = {"overall": {"recall@5": 0.8, "mrr": 0.6}}
        current = {"overall": {"recall@5": 0.7, "mrr": 0.65}}
        out = dict((k, (b, a, d)) for k, b, a, d in diff(baseline, current))
        self.assertEqual(out["recall@5"], (0.8, 0.7, -0.1))
        self.assertEqual(out["mrr"], (0.6, 0.65, 0.05))


if __name__ == "__main__":
    unittest.main()
