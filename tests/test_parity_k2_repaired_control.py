#!/usr/bin/env python3
import unittest

from experiments.parity_k2_repaired_control import DIRECT_LR, RECURRENT_LR, lr_for


class RepairedParityComparisonTests(unittest.TestCase):
    def test_direct_and_recurrent_rates_are_explicit(self):
        self.assertEqual(DIRECT_LR,0.001)
        self.assertEqual(RECURRENT_LR,0.002)
        self.assertEqual(lr_for("direct"),0.001)
        self.assertEqual(lr_for("serial-control"),0.002)
        self.assertEqual(lr_for("latent"),0.002)

    def test_recurrent_comparators_share_learning_rate(self):
        self.assertEqual(lr_for("serial-control"),lr_for("latent"))


if __name__=="__main__":
    unittest.main()
