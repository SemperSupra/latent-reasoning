#!/usr/bin/env python3
import itertools
import unittest

import torch

from experiments import register_state_saturation as exp


class RegisterStateSaturationTests(unittest.TestCase):
    def test_operation_semantics(self):
        self.assertEqual(exp.apply_op((2, 1), exp.OP_ADD_R0), (0, 1))
        self.assertEqual(exp.apply_op((2, 1), exp.OP_ADD_R1), (2, 2))
        self.assertEqual(exp.apply_op((2, 1), exp.OP_COPY_R0_R1), (1, 1))
        self.assertEqual(exp.apply_op((2, 1), exp.OP_COPY_R1_R0), (2, 2))
        self.assertEqual(exp.apply_op((2, 1), exp.OP_SWAP), (1, 2))

    def test_complete_universe_sizes(self):
        # For each length L: 3^2 initial states * 5^L op sequences * 2 queries.
        expected = {
            n: sum((exp.MODULUS**2) * (len(exp.OPS)**l) * 2 for l in range(1, n + 1))
            for n in (1, 2, 3)
        }
        for max_steps, size in expected.items():
            _, y, _ = exp.exhaustive_examples(max_steps)
            self.assertEqual(len(y), size)

    def test_targets_match_exact_execution(self):
        x, y, lengths = exp.exhaustive_examples(2)
        self.assertEqual(len(y), 540)
        # Exhaustive enumeration already derives labels through the same exact
        # transition function; spot-check explicit compositions separately.
        trace = exp.execute(
            (0, 2),
            (exp.OP_COPY_R0_R1, exp.OP_ADD_R1),
        )
        self.assertEqual(trace, [(0, 2), (2, 2), (2, 0)])
        row, target = exp.encode_example(
            (0, 2),
            (exp.OP_COPY_R0_R1, exp.OP_ADD_R1),
            1,
        )
        self.assertEqual(target, 0)
        self.assertEqual(row[-1], exp.CLS)

    def test_model_forward_shape(self):
        x, y, lengths = exp.exhaustive_examples(1)
        model = exp.RegisterStateTransformer(d_model=32, layers=1, heads=4)
        logits = model(x, lengths)
        self.assertEqual(tuple(logits.shape), (len(y), exp.MODULUS))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
