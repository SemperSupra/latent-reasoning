#!/usr/bin/env python3
import unittest

import torch

from experiments import toy_depth_sweep as sweep


class DepthSweepTests(unittest.TestCase):
    def test_depth_changes_parameter_count(self):
        m1 = sweep.DirectDepthReasoner(1)
        m2 = sweep.DirectDepthReasoner(2)
        self.assertGreater(
            sum(p.numel() for p in m2.parameters()),
            sum(p.numel() for p in m1.parameters()),
        )

    def test_forward_shapes(self):
        x, _ = sweep.make_examples(1, 4, action_steps=6)
        for layers in (1, 2, 4):
            model = sweep.DirectDepthReasoner(layers)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (4, sweep.STATE_COUNT))
            self.assertTrue(torch.isfinite(logits).all().item())

    def test_invalid_depth_rejected(self):
        with self.assertRaises(ValueError):
            sweep.DirectDepthReasoner(0)


if __name__ == "__main__":
    unittest.main()
