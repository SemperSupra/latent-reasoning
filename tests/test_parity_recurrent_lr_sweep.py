#!/usr/bin/env python3
import unittest

from experiments.parity_recurrent_lr_sweep import choose_candidate


class RecurrentLRSweepTests(unittest.TestCase):
    def test_unqualified_candidates_are_excluded(self):
        candidates = [
            {"learning_rate": 0.001, "qualified": False, "mean_total_epochs": 100},
            {"learning_rate": 0.002, "qualified": True, "mean_total_epochs": 200},
        ]
        self.assertEqual(choose_candidate(candidates), 0.002)

    def test_selects_lowest_epoch_cost(self):
        candidates = [
            {"learning_rate": 0.0005, "qualified": True, "mean_total_epochs": 350},
            {"learning_rate": 0.001, "qualified": True, "mean_total_epochs": 220},
            {"learning_rate": 0.002, "qualified": True, "mean_total_epochs": 260},
        ]
        self.assertEqual(choose_candidate(candidates), 0.001)

    def test_tie_breaks_to_lower_rate(self):
        candidates = [
            {"learning_rate": 0.002, "qualified": True, "mean_total_epochs": 200},
            {"learning_rate": 0.001, "qualified": True, "mean_total_epochs": 200},
        ]
        self.assertEqual(choose_candidate(candidates), 0.001)

    def test_returns_none_without_joint_saturation(self):
        self.assertIsNone(
            choose_candidate([
                {"learning_rate": 0.001, "qualified": False, "mean_total_epochs": 100},
            ])
        )


if __name__ == "__main__":
    unittest.main()
