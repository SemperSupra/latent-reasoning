"""Offline no-network contract tests for the Kaggle direct path."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kaggle_latent_direct",
    ROOT / "tools" / "kaggle_latent_direct.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class TestKaggleLatentDirect(unittest.TestCase):
    def test_both_workloads_are_fixed_allowlist(self):
        self.assertEqual(set(module.WORKLOADS), {"gpu-cuda-v1", "tpu-bridge-v1"})
        for workload in module.WORKLOADS:
            p = module.plan("plan", workload)
            self.assertEqual(len(p["source_sha256"]), 64)
            self.assertTrue(Path(p["source_path"]).is_file())
            self.assertEqual(p["native_version"], 0)

    def test_prohibit_unknown_workload(self):
        with self.assertRaises(module.Denied):
            module.plan("submit", "arbitrary-script")

    def test_observe_requires_exact_version(self):
        with self.assertRaises(module.Denied):
            module.plan("observe", "tpu-bridge-v1")
        with self.assertRaises(module.Denied):
            module.plan("submit", "gpu-cuda-v1", 2)
        self.assertEqual(module.plan("observe", "tpu-bridge-v1", 1)["native_version"], 1)

    def test_provider_metadata_private_offline(self):
        for workload in module.WORKLOADS:
            p = module.plan("submit", workload)
            m = module.metadata("qualifieduser/" + p["slug"], p)
            self.assertIs(m["is_private"], True)
            self.assertIs(m["enable_internet"], False)
            self.assertEqual(m["dataset_sources"], [])
            self.assertEqual(m["model_sources"], [])
            self.assertEqual(m["machine_shape"], p["accelerator"])
            self.assertNotEqual(m["enable_gpu"], m["enable_tpu"])

    def test_injected_provider_ref_refused(self):
        p = module.plan("submit", "gpu-cuda-v1")
        with self.assertRaises(module.Denied):
            module.metadata("user/../../other", p)

    def test_gpu_oracle(self):
        expected = module.plan("plan", "gpu-cuda-v1")["source_sha256"]
        result = {"schema": module.RESULT_SCHEMA, "workload": "gpu-cuda-v1",
                  "status": "PASS", "source_sha256": expected, "metrics": {
                    "gpu_count": 2, "requested_accelerator": "NvidiaTeslaT4",
                    "device_models": ["Tesla T4", "Tesla T4"],
                    "device_arithmetic_checksums": [8386560.0, 8386560.0]}}
        self.assertEqual(module.verify_science(result, "gpu-cuda-v1", expected)["gpu_count"], 2)
        result["metrics"]["device_arithmetic_checksums"][1] = 0.0
        with self.assertRaises(module.Denied):
            module.verify_science(result, "gpu-cuda-v1", expected)

    def test_tpu_oracle_and_shuffled_control(self):
        expected = module.plan("plan", "tpu-bridge-v1")["source_sha256"]
        result = {"schema": module.RESULT_SCHEMA, "workload": "tpu-bridge-v1",
                  "status": "PASS", "source_sha256": expected, "metrics": {
                    "requested_accelerator": "TpuV5E8", "tpu_count": 8,
                    "jax_backend_tpu": True, "tpu_arithmetic": True,
                    "tpu_arithmetic_checksum": 148.0, "data_parallel_devices": 8,
                    "scope": "synthetic-method-sanity-only",
                    "heldout_cosine": 0.95, "shuffled_control_cosine": 0.10}}
        self.assertEqual(module.verify_science(result, "tpu-bridge-v1", expected)["tpu_count"], 8)
        result["metrics"]["shuffled_control_cosine"] = 0.85
        with self.assertRaises(module.Denied):
            module.verify_science(result, "tpu-bridge-v1", expected)

    def test_bad_scientific_receipt(self):
        with self.assertRaises(module.Denied):
            module.verify_science({"schema": "other"}, "gpu-cuda-v1", "0" * 64)

    def test_quota_unknown_and_insufficient_fail_closed(self):
        from datetime import timedelta
        from types import SimpleNamespace
        class Fake:
            def quota_view(self):
                return SimpleNamespace(gpu_quota=SimpleNamespace(
                    time_used=timedelta(seconds=590),
                    total_time_allowed=timedelta(seconds=600)),
                    tpu_quota=None)
        with self.assertRaises(module.Denied):
            module.quota(Fake(), module.plan("submit", "gpu-cuda-v1"))
        with self.assertRaises(module.Denied):
            module.quota(Fake(), module.plan("submit", "tpu-bridge-v1"))

    def test_existing_target_blocks_duplicate_submission(self):
        from types import SimpleNamespace
        p = module.plan("submit", "gpu-cuda-v1")
        target = "user/" + p["slug"]
        class Fake:
            def kernels_list_with_response(self, **kwargs):
                return SimpleNamespace(kernels=[SimpleNamespace(ref=target)],
                                       next_page_token=None)
        with self.assertRaisesRegex(module.Denied, "TARGET_EXISTS"):
            module.ensure_not_existing_or_active(Fake(), target)

    def test_other_active_experiment_blocks_submission(self):
        from types import SimpleNamespace
        class Fake:
            def kernels_list_with_response(self, **kwargs):
                return SimpleNamespace(
                    kernels=[SimpleNamespace(ref="user/ad-rdte-active-other")],
                    next_page_token=None)
            def kernels_status(self, ref):
                return SimpleNamespace(status="RUNNING")
        with self.assertRaisesRegex(module.Denied, "ACTIVE"):
            module.ensure_not_existing_or_active(
                Fake(), "user/latent-c001-gpu-cuda-v1")

    def test_credentialless_submit_is_blocked_before_importing_kaggle(self):
        import json
        import os
        import sys
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["kaggle_latent_direct.py", "--action", "submit",
                    "--workload", "gpu-cuda-v1", "--out-dir", tmp]
            with patch.object(sys, "argv", argv), patch.dict(
                    os.environ, {"KAGGLE_API_TOKEN": ""}):
                self.assertEqual(module.main(), 2)
            with open(Path(tmp) / "control-receipt.json", encoding="utf-8") as stream:
                receipt = json.load(stream)
            self.assertEqual(receipt["state"], "BLOCKED")
            self.assertEqual(receipt["failure_class"], "NO_KAGGLE_ENV_TOKEN")

    def test_pr_workflow_contains_no_credentialed_execution(self):
        workflow = (ROOT / ".github/workflows/latent-kaggle-direct.yml"
                    ).read_text(encoding="utf-8")
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("github.event_name == 'workflow_dispatch'", workflow)
        self.assertIn("environment: kaggle-research", workflow)
        self.assertIn("if: github.event_name == 'pull_request'", workflow)


    def test_source_hash_is_execution_bound_and_not_optional(self):
        for workload in module.WORKLOADS:
            plan = module.plan("plan", workload)
            f = Path(plan["source_path"])
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(), plan["source_sha256"])
            text = f.read_text()
            self.assertIn('hashlib.sha256(Path(__file__).read_bytes()).hexdigest()', text)

        expected = module.plan("plan", "gpu-cuda-v1")["source_sha256"]
        metrics = {"gpu_count": 2, "requested_accelerator": "NvidiaTeslaT4",
                   "device_models": ["Tesla T4", "Tesla T4"],
                   "device_arithmetic_checksums": [8386560.0, 8386560.0]}
        valid = {"schema": module.RESULT_SCHEMA, "workload": "gpu-cuda-v1",
                 "status": "PASS", "source_sha256": expected, "metrics": metrics}
        self.assertEqual(module.verify_science(valid, "gpu-cuda-v1", expected)["gpu_count"], 2)
        for invalid in (None, "", "0" * 64, 43, "F" * 64):
            changed = dict(valid)
            if invalid is None:
                changed.pop("source_sha256")
            else:
                changed["source_sha256"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(module.Denied, "SOURCE"):
                module.verify_science(changed, "gpu-cuda-v1", expected)
        with self.assertRaisesRegex(module.Denied, "SOURCE"):
            module.verify_science(valid, "gpu-cuda-v1", "x")

    def test_tpu_nonfinite_out_of_range_and_bool_scores_rejected(self):
        import math
        expected = module.plan("plan", "tpu-bridge-v1")["source_sha256"]
        base = {"requested_accelerator": "TpuV5E8", "tpu_count": 8,
                "jax_backend_tpu": True, "tpu_arithmetic": True,
                "tpu_arithmetic_checksum": 148.0, "data_parallel_devices": 8,
                "scope": "synthetic-method-sanity-only",
                "heldout_cosine": .95, "shuffled_control_cosine": .10}
        for score in (float("inf"), float("nan"), -float("inf"), 1.5, True):
            with self.subTest(score=score):
                metrics = dict(base, heldout_cosine=score)
                record = {"schema": module.RESULT_SCHEMA, "workload": "tpu-bridge-v1",
                          "status": "PASS", "source_sha256": expected, "metrics": metrics}
                with self.assertRaisesRegex(module.Denied, "BRIDGE"):
                    module.verify_science(record, "tpu-bridge-v1", expected)

if __name__ == "__main__":
    unittest.main()
