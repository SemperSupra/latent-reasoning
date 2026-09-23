#!/usr/bin/env python3
import unittest

from tools.summarize_representation_comparison import classify, summarize


def receipt(deltas=(0.1, -0.05, 0.02), *, valid=True):
    results=[]
    for seed, delta in enumerate(deltas):
        results.extend([
            {
                "treatment":"direct","seed":seed,
                "mean_ood_accuracy":0.50,
            },
            {
                "treatment":"serial-control","seed":seed,
                "mean_ood_accuracy":0.45,
            },
            {
                "treatment":"latent","seed":seed,
                "mean_ood_accuracy":0.45+delta,
            },
        ])
    return {
        "evidence_class":"fixed-k-reasoning-representation-comparison",
        "family":"fixture",
        "settings":{"recurrent_steps":2},
        "comparison_valid":valid,
        "aggregate":{
            "direct":{
                "all_seeds_id_saturated":True,
                "forward_calls_per_batch":1,
            },
            "serial-control":{
                "all_seeds_id_saturated":True,
                "forward_calls_per_batch":3,
            },
            "latent":{
                "all_seeds_id_saturated":True,
                "forward_calls_per_batch":3,
            },
        },
        "results":results,
    }


class SummaryTests(unittest.TestCase):
    def test_direction(self):
        self.assertEqual(classify([1,2,3]),"all-positive")
        self.assertEqual(classify([-1,-2]),"all-negative")
        self.assertEqual(classify([1,-1]),"mixed")

    def test_paired_seed_summary(self):
        result=summarize(receipt((0.1,0.2,0.3)))
        comp=result["paired"]["latent_minus_serial-control"]
        self.assertEqual(comp["paired_seed_count"],3)
        self.assertEqual(comp["direction"],"all-positive")
        self.assertAlmostEqual(comp["mean"],0.2)
        self.assertEqual(result["inference_boundary"],"descriptive-only")

    def test_compute_mismatch_invalidates(self):
        doc=receipt((0.1,0.2,0.3))
        doc["aggregate"]["latent"]["forward_calls_per_batch"]=4
        result=summarize(doc)
        self.assertFalse(result["comparison_valid"])
        self.assertFalse(result["compute_match_valid"])

    def test_unsaturated_treatment_invalidates(self):
        doc=receipt((0.1,0.2,0.3))
        doc["aggregate"]["latent"]["all_seeds_id_saturated"]=False
        result=summarize(doc)
        self.assertFalse(result["comparison_valid"])


if __name__=="__main__":
    unittest.main()
