#!/usr/bin/env python3
import unittest

import torch

from experiments import toy_capacity_sweep as sweep


class CapacitySweepTests(unittest.TestCase):
    def test_width_changes_parameter_count_only_capacity_axis(self):
        m32 = sweep.DirectReasoner(32)
        m64 = sweep.DirectReasoner(64)
        self.assertGreater(
            sum(p.numel() for p in m64.parameters()),
            sum(p.numel() for p in m32.parameters()),
        )

    def test_forward_shapes(self):
        x, _ = sweep.make_examples(1, 4, action_steps=6)
        for width in (32, 64, 128):
            model = sweep.DirectReasoner(width)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (4, sweep.STATE_COUNT))
            self.assertTrue(torch.isfinite(logits).all().item())

    def test_invalid_width_rejected(self):
        with self.assertRaises(ValueError):
            sweep.DirectReasoner(30)


if __name__ == "__main__":
    unittest.main()
