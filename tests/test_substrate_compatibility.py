#!/usr/bin/env python3
import unittest

from tools.check_substrate_compatibility import evaluate


PROFILE={
    "id":"gpu-smoke",
    "resource_class":"gpu",
    "requirements":{
        "min_logical_cpus":4,
        "min_memory_bytes":16,
        "min_disk_free_bytes":30,
        "docker_required":True,
        "nvidia_required":True,
        "min_gpu_count":1,
        "min_gpu_memory_mib":20000,
        "nvidia_container_runtime_required":True,
    },
}


def observation(*,gpu=True,runtime=True):
    return {
        "resources":{
            "logical_cpus":8,
            "memory_total_bytes":64,
            "disk_free_bytes":100,
        },
        "docker":{"available":True,"runtimes":["runc","nvidia"] if runtime else ["runc"]},
        "nvidia":{
            "nvidia_smi_available":gpu,
            "container_runtime_declared":runtime,
            "gpus":[{"name":"GPU","memory_total_mib":24576,"driver_version":"x"}] if gpu else [],
        },
    }


class CompatibilityTests(unittest.TestCase):
    def test_compatible_gpu_host(self):
        result=evaluate(observation(),PROFILE)
        self.assertTrue(result["compatible"])
        self.assertEqual(result["decision"],"eligible-for-bounded-apply")

    def test_cpu_only_host_fails_closed(self):
        result=evaluate(observation(gpu=False,runtime=False),PROFILE)
        self.assertFalse(result["compatible"])
        self.assertIn("nvidia_device_visibility_unavailable",result["blocking_reasons"])
        self.assertIn("gpu_count_below_minimum",result["blocking_reasons"])
        self.assertIn("nvidia_container_runtime_not_declared",result["blocking_reasons"])

    def test_insufficient_gpu_memory_blocks(self):
        obs=observation()
        obs["nvidia"]["gpus"][0]["memory_total_mib"]=16000
        result=evaluate(obs,PROFILE)
        self.assertFalse(result["compatible"])
        self.assertIn("qualifying_gpu_memory_below_minimum",result["blocking_reasons"])


if __name__=="__main__":
    unittest.main()
