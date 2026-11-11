"""Offline no-network contract tests for the Kaggle direct path."""
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
        result = {"schema": module.RESULT_SCHEMA, "workload": "gpu-cuda-v1",
                  "status": "PASS", "metrics": {
                    "gpu_count": 2, "requested_accelerator": "NvidiaTeslaT4",
                    "device_models": ["Tesla T4", "Tesla T4"],
                    "device_arithmetic_checksums": [8386560.0, 8386560.0]}}
        self.assertEqual(module.verify_science(result, "gpu-cuda-v1")["gpu_count"], 2)
        result["metrics"]["device_arithmetic_checksums"][1] = 0.0
        with self.assertRaises(module.Denied):
            module.verify_science(result, "gpu-cuda-v1")

    def test_tpu_oracle_and_shuffled_control(self):
        result = {"schema": module.RESULT_SCHEMA, "workload": "tpu-bridge-v1",
                  "status": "PASS", "metrics": {
                    "requested_accelerator": "TpuV5E8", "tpu_count": 8,
                    "jax_backend_tpu": True, "tpu_arithmetic": True,
                    "tpu_arithmetic_checksum": 148.0,
                    "scope": "synthetic-method-sanity-only",
                    "heldout_cosine": 0.95, "shuffled_control_cosine": 0.10}}
        self.assertEqual(module.verify_science(result, "tpu-bridge-v1")["tpu_count"], 8)
        result["metrics"]["shuffled_control_cosine"] = 0.85
        with self.assertRaises(module.Denied):
            module.verify_science(result, "tpu-bridge-v1")

    def test_bad_scientific_receipt(self):
        with self.assertRaises(module.Denied):
            module.verify_science({"schema": "other"}, "gpu-cuda-v1")

if __name__ == "__main__":
    unittest.main()
