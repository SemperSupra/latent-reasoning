#!/usr/bin/env python3
import unittest
from unittest.mock import patch

from tools import observe_substrate as obs


class SubstrateObserverTests(unittest.TestCase):
    def test_nvidia_parser(self):
        docker={"runtimes":["io.containerd.runc.v2","nvidia"]}
        with patch.object(obs,"run",return_value="RTX A5000, 24564, 550.142\nRTX A5000, 24564, 550.142"):
            result=obs.nvidia_observation(docker)
        self.assertTrue(result["nvidia_smi_available"])
        self.assertTrue(result["container_runtime_declared"])
        self.assertEqual(len(result["gpus"]),2)
        self.assertEqual(result["gpus"][0]["memory_total_mib"],24564)

    def test_no_nvidia_is_valid_observation(self):
        with patch.object(obs,"run",return_value=None):
            result=obs.nvidia_observation({"runtimes":["runc"]})
        self.assertFalse(result["nvidia_smi_available"])
        self.assertEqual(result["gpus"],[])
        self.assertFalse(result["container_runtime_declared"])


if __name__=="__main__":
    unittest.main()
