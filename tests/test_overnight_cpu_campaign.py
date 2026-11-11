import unittest
import copy
from datetime import date, timedelta

from tools.overnight_cpu_campaign import (
    START, COHORTS, DEPTHS, planned, verify, digest, SCHEMA
)

class OvernightCampaignTests(unittest.TestCase):
    def test_bounded_and_nonrepeating(self):
        jobs = [planned(START + timedelta(days=i)) for i in range(24)]
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({r["id"] for r in jobs}), 24)
        self.assertIsNone(planned(START - timedelta(days=1)))
        self.assertIsNone(planned(START + timedelta(days=24)))
        self.assertEqual(COHORTS * len(DEPTHS), 24)

    def test_cohort_is_paired_across_depth(self):
        first = [planned(START + timedelta(days=i)) for i in range(3)]
        self.assertEqual([r["arguments"]["ks"] for r in first], ["1", "2", "4"])
        self.assertEqual({r["arguments"]["seeds"] for r in first}, {"8,9"})
        second = planned(START + timedelta(days=3))
        self.assertEqual(second["arguments"]["seeds"], "10,11")

    def test_cpu_only_fixed_method(self):
        for offset in range(24):
            p = planned(START + timedelta(days=offset))
            self.assertEqual(p["module"], "experiments.register_state_k_sweep")
            self.assertEqual(p["resource_class"], "cpu")
            self.assertLessEqual(p["timeout_seconds"], 1080)
            self.assertEqual(p["arguments"]["stages"], "1,2")
            self.assertNotIn("script", p["arguments"])
            self.assertNotIn("provider", p["arguments"])

    def fixture(self, saturate=True):
        p = planned(START)
        seeds = [8,9]
        records=[]
        for t in ["serial-control","latent"]:
            for s in seeds:
                records.append({
                    "treatment": t,"seed": s,"recurrent_steps": 1,
                    "forward_calls_per_batch": 2,
                    "id_saturated": saturate,"final_id_accuracy": 1.0 if saturate else .75,
                    "mean_ood_accuracy": (.53 if t == "latent" else .5) if saturate else None})
        science={"schema_version":1, "evidence_class":"recurrent-depth-sweep",
                 "family":"register-state-mini-v1",
                 "treatments":["serial-control","latent"],
                 "settings":{"ks":[1],"seeds":seeds,"stages":[1,2]},
                 "sweep":[{"k":1,"comparison_valid":saturate,
                           "results":records,"paired_latent_minus_serial":{
                               "mean":.03 if saturate else None,
                               "paired_seed_count":2 if saturate else 0,
                               "by_seed":{"8":.03,"9":.03} if saturate else {}}}],
                 "valid_ks":[1] if saturate else []}
        meta={"replay_spec_id":p["id"],"replay_spec_sha256":digest(p),
              "scientific_receipt_sha256":digest(science), "resource_class":"cpu",
              "source_revision":"0123456789abcdef"}
        return p, science, meta

    def test_valid_but_descriptive_only(self):
        p, s, m=self.fixture()
        report=verify(p,s,m)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["method_status"], "VALID_RECEIPT")
        self.assertEqual(report["claim_status"], "DESCRIPTIVE_PAIRED_ONLY")

    def test_underfit_is_a_valid_negative(self):
        p,s,m=self.fixture(False)
        report=verify(p,s,m)
        self.assertEqual(report["claim_status"], "HOLD_ID_UNSATURATED")
        self.assertIsNone(report["paired_ood_mean"])

    def test_ood_leak_rejected(self):
        p,s,m=self.fixture(False)
        s["sweep"][0]["results"][0]["mean_ood_accuracy"]=.8
        m["scientific_receipt_sha256"]=digest(s)
        with self.assertRaisesRegex(ValueError,"OOD leaked"):
            verify(p,s,m)

    def test_wrong_depth_rejected(self):
        p,s,m=self.fixture()
        s["sweep"][0]["results"][0]["forward_calls_per_batch"]=1
        m["scientific_receipt_sha256"]=digest(s)
        with self.assertRaisesRegex(ValueError,"forward-call"):
            verify(p,s,m)

    def test_tampered_receipt_rejected(self):
        p,s,m=self.fixture()
        s["treatments"]=["latent"]
        with self.assertRaisesRegex(ValueError,"provenance"):
            verify(p,s,m)

    def test_forged_paired_ood_mean_rejected(self):
        p,s,m=self.fixture()
        s["sweep"][0]["paired_latent_minus_serial"]["mean"]=0.9
        m["scientific_receipt_sha256"]=digest(s)
        with self.assertRaisesRegex(ValueError,"independent paired OOD mean"):
            verify(p,s,m)

    def test_forged_source_spec_rejected(self):
        p,s,m=self.fixture()
        p["arguments"]["max_epochs_per_stage"]=200
        m["replay_spec_sha256"]=digest(p)
        with self.assertRaisesRegex(ValueError,"unregistered"):
            verify(p,s,m)

    def test_inconsistent_gate_rejected(self):
        p,s,m=self.fixture()
        s["sweep"][0]["comparison_valid"]=False
        m["scientific_receipt_sha256"]=digest(s)
        with self.assertRaisesRegex(ValueError,"validity gate"):
            verify(p,s,m)

if __name__ == "__main__":
    unittest.main()
