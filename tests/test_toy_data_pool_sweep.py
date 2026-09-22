#!/usr/bin/env python3
import unittest

import torch

from experiments import toy_data_pool_sweep as sweep


class DataPoolSweepTests(unittest.TestCase):
    def test_unique_examples_are_unique_and_correct(self):
        x, y = sweep.make_unique_examples(42, 512)
        self.assertEqual(len({tuple(row) for row in x.tolist()}), 512)

        for row, label in zip(x.tolist(), y.tolist()):
            state = row[0]
            for token in row[1:]:
                delta = sweep.ACTION_DELTAS[token - sweep.ACTION_TOKEN_BASE]
                state = (state + delta) % sweep.STATE_COUNT
            self.assertEqual(state, label)

    def test_unique_universe_limit(self):
        universe = sweep.STATE_COUNT * (
            len(sweep.ACTION_DELTAS) ** sweep.TRAIN_ACTION_STEPS
        )
        with self.assertRaises(ValueError):
            sweep.make_unique_examples(1, universe + 1)

    def test_forward_shape(self):
        x, _ = sweep.make_unique_examples(7, 4)
        model = sweep.DirectDataReasoner()
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4, sweep.STATE_COUNT))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
