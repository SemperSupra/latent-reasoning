#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from tools.run_replay import argv_for, load_spec


class ReplayRunnerTests(unittest.TestCase):
    def test_builds_module_argv_without_shell(self):
        spec={
            "schema_version":1,
            "id":"fixture",
            "resource_class":"cpu",
            "environment_ref":"env.json",
            "module":"experiments.example",
            "arguments":{"seeds":"0,1","epochs":5},
            "timeout_seconds":60,
        }
        argv=argv_for(spec)
        self.assertEqual(argv[1:3],["-m","experiments.example"])
        self.assertIn("--seeds",argv)
        self.assertIn("0,1",argv)
        self.assertIn("--epochs",argv)
        self.assertIn("5",argv)

    def test_rejects_non_experiment_module(self):
        doc={
            "schema_version":1,
            "id":"fixture",
            "resource_class":"cpu",
            "environment_ref":"env.json",
            "module":"os",
            "arguments":{},
            "timeout_seconds":60,
        }
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"spec.json"
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                load_spec(path)

    def test_rejects_boolean_argument(self):
        doc={
            "schema_version":1,
            "id":"fixture",
            "resource_class":"cpu",
            "environment_ref":"env.json",
            "module":"experiments.example",
            "arguments":{"unsafe":True},
            "timeout_seconds":60,
        }
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"spec.json"
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                load_spec(path)


if __name__=="__main__":
    unittest.main()
