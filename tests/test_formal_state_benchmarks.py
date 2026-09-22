#!/usr/bin/env python3
import unittest

from benchmarks.formal_state import (
    RegisterOp,
    apply_register_op,
    generate_parity,
    generate_register_state,
)


class ParityTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_parity(17, 12), generate_parity(17, 12))

    def test_trace_is_exact_prefix_parity(self):
        task = generate_parity(101, 21)
        state = 0
        expected = [state]
        for bit in task["input"]["bits"]:
            state ^= bit
            expected.append(state)
        self.assertEqual(task["trace"], expected)
        self.assertEqual(task["target"], state)

    def test_length_is_controlled(self):
        for length in (1, 4, 8, 16, 32):
            task = generate_parity(7, length)
            self.assertEqual(len(task["input"]["bits"]), length)
            self.assertEqual(len(task["trace"]), length + 1)


class RegisterStateTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(
            generate_register_state(23, 8),
            generate_register_state(23, 8),
        )

    def test_operations_replay_exact_trace(self):
        task = generate_register_state(901, 20, registers=4, modulus=7)
        state = list(task["input"]["initial"])
        rebuilt = [list(state)]

        for raw in task["input"]["operations"]:
            state = apply_register_op(
                state,
                RegisterOp(raw["kind"], raw["a"], raw["b"]),
                modulus=7,
            )
            rebuilt.append(list(state))

        self.assertEqual(rebuilt, task["trace"])
        self.assertEqual(state, task["final_state"])
        q = task["input"]["query_register"]
        self.assertEqual(task["target"], state[q])

    def test_all_states_stay_in_modulus(self):
        task = generate_register_state(31415, 50, registers=5, modulus=11)
        for state in task["trace"]:
            self.assertEqual(len(state), 5)
            self.assertTrue(all(0 <= value < 11 for value in state))

    def test_add_wraps_modulo(self):
        self.assertEqual(
            apply_register_op([0, 1], RegisterOp("add", 0, 6), modulus=7),
            [6, 1],
        )


if __name__ == "__main__":
    unittest.main()
