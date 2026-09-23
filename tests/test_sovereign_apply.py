#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from tools.apply_coconut_gpu_smoke import build_argv, replay_argv, verify_outputs


class SovereignApplyTests(unittest.TestCase):
    def test_build_is_argv_not_shell(self):
        argv=build_argv(
            image_tag="capsule:test",
            dockerfile=Path("/repo/Dockerfile"),
            source_revision="a"*40,
        )
        self.assertEqual(argv[0:2],["docker","build"])
        self.assertIn("SOURCE_REVISION=" + "a"*40,argv)
        self.assertNotIn("sh",argv)

    def test_replay_requires_all_gpus_and_mount(self):
        argv=replay_argv(
            image_tag="capsule:test",
            replay_spec=Path.cwd()/"replays"/"coconut-reference-gpu-smoke.json",
            output_dir=Path("/tmp/out"),
        )
        self.assertEqual(argv[0:3],["docker","run","--rm"])
        self.assertIn("--gpus",argv)
        self.assertIn("all",argv)
        self.assertIn("/tmp/out:/out",argv)

    def test_verify_accepts_matching_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            scientific={
                "evidence_class":"pinned-upstream-reference-backbone-gpu-smoke",
                "gradient_through_recurrence":True,
                "optimizer_step_changed_parameter":True,
                "device":{"name":"GPU","total_memory_bytes":24576},
                "peak_cuda_allocated_bytes":100,
                "peak_cuda_reserved_bytes":200,
                "elapsed_seconds":1.5,
            }
            metadata={
                "replay_spec_id":"gpu-smoke",
                "source_revision":"b"*40,
                "scientific_receipt_sha256":"abc",
                "execution":{"container_contract":"coconut-gpu-smoke-v1"},
            }
            (root/"scientific-receipt.json").write_text(json.dumps(scientific))
            (root/"replay-metadata.json").write_text(json.dumps(metadata))
            result=verify_outputs(
                output_dir=root,
                replay_spec={"id":"gpu-smoke"},
                source_revision="b"*40,
            )
            self.assertEqual(result["status"],"verified")
            self.assertEqual(result["scientific_receipt_sha256"],"abc")


if __name__=="__main__":
    unittest.main()
