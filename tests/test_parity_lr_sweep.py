#!/usr/bin/env python3
import unittest

from experiments import parity_lr_sweep


class ParityLearningRateSweepTests(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(callable(parity_lr_sweep.main))


if __name__ == "__main__":
    unittest.main()
