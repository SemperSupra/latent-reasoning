#!/usr/bin/env python3
import json
import unittest
from pathlib import Path

import yaml


EXPECTED_BASE="sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
EXPECTED_TRAIN="0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c"


class Stage0CapsuleTests(unittest.TestCase):
    def test_environment_is_gpu_and_digest_pinned(self):
        env=json.loads(Path("environments/coconut-stage0-pytorch251-cuda124.json").read_text())
        self.assertEqual(env["resource_class"],"gpu")
        self.assertEqual(env["base_image_digest"],EXPECTED_BASE)
        self.assertEqual(env["requirements_file"],"requirements-coconut-stage0.txt")

    def test_plan_and_config_preserve_target_batch(self):
        plan=json.loads(Path("plans/coconut-gsm-cot-stage0-2gpu.json").read_text())
        config=yaml.safe_load(Path("configs/coconut-gsm-cot-stage0-2gpu.yaml").read_text())
        target=plan["target_execution"]
        self.assertEqual(
            target["world_size"]*config["batch_size_training"]*config["gradient_accumulation_steps"],
            128,
        )
        self.assertEqual(target["world_size"],2)
        self.assertEqual(plan["immutable_inputs"]["gsm_train_sha256"],EXPECTED_TRAIN)

    def test_runtime_has_inspection_and_gpu_gate(self):
        text=Path("tools/run_coconut_stage0.py").read_text()
        self.assertIn("--inspect-only",text)
        self.assertIn("torch.cuda.device_count() < 2",text)
        self.assertIn('"torchrun"',text)
        self.assertIn('"--nproc_per_node=2"',text)

    def test_dockerfile_uses_portable_lfs_hydrator(self):
        text=Path("containers/coconut-stage0.Dockerfile").read_text()
        self.assertIn("hydrate_pinned_lfs.py",text)
        self.assertIn(EXPECTED_BASE,text)
        self.assertIn("HF_HUB_OFFLINE=1",text)
        self.assertIn("WANDB_MODE=offline",text)


if __name__=="__main__":
    unittest.main()
