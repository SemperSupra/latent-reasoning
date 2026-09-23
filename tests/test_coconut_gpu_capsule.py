#!/usr/bin/env python3
import json
import unittest
from pathlib import Path


EXPECTED_COCONUT = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"
EXPECTED_GPT2 = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EXPECTED_BASE_DIGEST = "sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"


class CoconutGpuCapsuleContractTests(unittest.TestCase):
    def test_replay_is_gpu_scoped_and_pinned(self):
        replay=json.loads(Path("replays/coconut-reference-gpu-smoke.json").read_text())
        self.assertEqual(replay["resource_class"],"gpu")
        self.assertEqual(replay["module"],"experiments.coconut_gpu_smoke")
        self.assertEqual(replay["arguments"]["latent_slots"],2)

    def test_environment_pins_cuda_base_digest(self):
        env=json.loads(Path("environments/pytorch251-cuda124-cudnn9-gpu.json").read_text())
        self.assertEqual(env["base_image_digest"],EXPECTED_BASE_DIGEST)
        self.assertEqual(env["resource_class"],"gpu")

    def test_upstream_manifest_pins_both_sources(self):
        upstream=json.loads(Path("upstreams/coconut.json").read_text())
        self.assertEqual(upstream["exact_revision"],EXPECTED_COCONUT)
        self.assertEqual(upstream["reference_backbone_revision"],EXPECTED_GPT2)


if __name__=="__main__":
    unittest.main()
