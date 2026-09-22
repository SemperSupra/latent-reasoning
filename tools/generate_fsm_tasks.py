#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


ACTIONS = ("a", "b", "c")


def build_transition_table(rng: random.Random, state_count: int) -> Dict[str, Dict[str, str]]:
    states = [f"S{i}" for i in range(state_count)]
    return {
        state: {action: rng.choice(states) for action in ACTIONS}
        for state in states
    }


def apply_actions(start: str, actions: List[str], transitions: Dict[str, Dict[str, str]]):
    trace = [start]
    current = start
    for action in actions:
        current = transitions[current][action]
        trace.append(current)
    return trace


def generate_task(seed: int, state_count: int = 7, steps: int = 8):
    rng = random.Random(seed)
    transitions = build_transition_table(rng, state_count)
    start = f"S{rng.randrange(state_count)}"
    actions = [rng.choice(ACTIONS) for _ in range(steps)]
    trace = apply_actions(start, actions, transitions)

    return {
        "schema_version": 1,
        "generator": "deterministic-fsm-v0",
        "seed": seed,
        "states": [f"S{i}" for i in range(state_count)],
        "actions": list(ACTIONS),
        "transitions": transitions,
        "start_state": start,
        "action_sequence": actions,
        "trace": trace,
        "final_state": trace[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--states", type=int, default=7)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for offset in range(args.count):
            task = generate_task(args.seed + offset, args.states, args.steps)
            handle.write(json.dumps(task, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
