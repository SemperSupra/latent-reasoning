#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from tools.apply_coconut_stage0 import build_argv, run_argv, verify_outputs


class Stage0ApplyTests(unittest.TestCase):
    def test_build_is_digest_source_revision_bound(self):
        argv=build_argv(
            image_tag="stage0:test",
            dockerfile=Path("/repo/containers/coconut-stage0.Dockerfile"),
            source_revision="a"*40,
        )
        self.assertEqual(argv[:2],["docker","build"])
        self.assertIn("SOURCE_REVISION="+"a"*40,argv)
        self.assertIn("stage0:test",argv)

    def test_run_uses_gpu_without_shell(self):
        with tempfile.TemporaryDirectory() as td:
            argv=run_argv(image_tag="stage0:test",output_dir=Path(td))
        self.assertEqual(argv[:3],["docker","run","--rm"])
        self.assertIn("--gpus",argv)
        self.assertIn("all",argv)
        self.assertIn("--shm-size",argv)
        self.assertNotIn("sh",argv)
        self.assertNotIn("bash",argv)

    def test_verify_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            rev="b"*40
            (root/"stage0-capsule-inspection.json").write_text(json.dumps({
                "evidence_class":"coconut-stage0-capsule-inspection",
                "container_contract":"coconut-stage0-v1",
                "repository_source_revision":rev,
            }))
            (root/"stage0-execution.json").write_text(json.dumps({
                "evidence_class":"coconut-stage0-training-execution",
                "container_contract":"coconut-stage0-v1",
                "repository_source_revision":rev,
                "exit_code":0,
                "cuda_device_count":2,
                "elapsed_seconds":1.0,
                "checkpoints":[],
            }))
            with self.assertRaises(RuntimeError):
                verify_outputs(output_dir=root,source_revision=rev)

    def test_verify_accepts_hashed_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            rev="c"*40
            (root/"stage0-capsule-inspection.json").write_text(json.dumps({
                "evidence_class":"coconut-stage0-capsule-inspection",
                "container_contract":"coconut-stage0-v1",
                "repository_source_revision":rev,
            }))
            (root/"stage0-execution.json").write_text(json.dumps({
                "evidence_class":"coconut-stage0-training-execution",
                "container_contract":"coconut-stage0-v1",
                "repository_source_revision":rev,
                "exit_code":0,
                "cuda_device_count":2,
                "elapsed_seconds":1.0,
                "checkpoints":[{
                    "name":"checkpoint_1",
                    "size_bytes":123,
                    "sha256":"d"*64,
                }],
            }))
            result=verify_outputs(output_dir=root,source_revision=rev)
            self.assertEqual(result["status"],"verified")
            self.assertEqual(result["checkpoint_count"],1)


if __name__=="__main__":
    unittest.main()
