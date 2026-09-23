#!/usr/bin/env python3
import unittest
from types import SimpleNamespace

from experiments.register_state_k_sweep import paired


class RegisterKSweepTests(unittest.TestCase):
    def test_paired_direction(self):
        rows=[]
        for seed,(serial,latent) in enumerate(((0.40,0.45),(0.50,0.52),(0.48,0.55))):
            rows.append(SimpleNamespace(
                treatment="serial-control",seed=seed,mean_ood_accuracy=serial
            ))
            rows.append(SimpleNamespace(
                treatment="latent",seed=seed,mean_ood_accuracy=latent
            ))
        result=paired(rows)
        self.assertEqual(result["paired_seed_count"],3)
        self.assertEqual(result["direction"],"all-positive")
        self.assertGreater(result["mean"],0)

    def test_missing_ood_is_excluded(self):
        rows=[
            SimpleNamespace(treatment="serial-control",seed=0,mean_ood_accuracy=None),
            SimpleNamespace(treatment="latent",seed=0,mean_ood_accuracy=0.5),
        ]
        result=paired(rows)
        self.assertEqual(result["paired_seed_count"],0)
        self.assertEqual(result["direction"],"unavailable")


if __name__=="__main__":
    unittest.main()
