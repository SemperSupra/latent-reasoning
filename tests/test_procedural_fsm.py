#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_fsm_tasks.py"
spec = importlib.util.spec_from_file_location("generate_fsm_tasks", MODULE_PATH)
fsm = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fsm)


class ProceduralFSMTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        self.assertEqual(fsm.generate_task(17), fsm.generate_task(17))

    def test_distinct_seeds_change_instances(self):
        self.assertNotEqual(fsm.generate_task(17), fsm.generate_task(18))

    def test_trace_matches_transition_table(self):
        task = fsm.generate_task(901, state_count=9, steps=12)
        rebuilt = fsm.apply_actions(
            task["start_state"],
            task["action_sequence"],
            task["transitions"],
        )
        self.assertEqual(rebuilt, task["trace"])
        self.assertEqual(rebuilt[-1], task["final_state"])
        self.assertEqual(len(rebuilt), 13)

    def test_all_referenced_states_are_in_declared_state_set(self):
        task = fsm.generate_task(31415, state_count=11, steps=20)
        states = set(task["states"])
        self.assertIn(task["start_state"], states)
        self.assertIn(task["final_state"], states)
        self.assertTrue(set(task["trace"]) <= states)
        for source, row in task["transitions"].items():
            self.assertIn(source, states)
            self.assertEqual(set(row), set(fsm.ACTIONS))
            self.assertTrue(set(row.values()) <= states)


if __name__ == "__main__":
    unittest.main()
