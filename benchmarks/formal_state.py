#!/usr/bin/env python3
"""Independent formal-state benchmark generators.

The tasks are specified mathematically and implemented from scratch in this
repository. No third-party benchmark source code or data is copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def canonical_digest(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def generate_parity(seed: int, length: int) -> dict:
    if length < 1:
        raise ValueError("length must be >= 1")
    rng = random.Random(seed)
    bits = [rng.randrange(2) for _ in range(length)]
    state = 0
    trace = [state]
    for bit in bits:
        state ^= bit
        trace.append(state)

    task = {
        "schema_version": 1,
        "family": "parity",
        "seed": seed,
        "difficulty": {"length": length},
        "input": {"bits": bits},
        "initial_state": 0,
        "trace": trace,
        "target": state,
        "oracle": "xor-prefix-state",
    }
    task["instance_digest"] = canonical_digest(task)
    return task


@dataclass(frozen=True)
class RegisterOp:
    kind: str
    a: int
    b: int


def _random_register_op(rng: random.Random, registers: int, modulus: int) -> RegisterOp:
    kind = rng.choice(("add", "copy", "swap"))
    if kind == "add":
        return RegisterOp(kind, rng.randrange(registers), rng.choice((-2, -1, 1, 2)) % modulus)
    a = rng.randrange(registers)
    b = rng.randrange(registers - 1)
    if b >= a:
        b += 1
    return RegisterOp(kind, a, b)


def apply_register_op(state: list[int], op: RegisterOp, modulus: int) -> list[int]:
    out = list(state)
    if op.kind == "add":
        out[op.a] = (out[op.a] + op.b) % modulus
    elif op.kind == "copy":
        out[op.a] = out[op.b]
    elif op.kind == "swap":
        out[op.a], out[op.b] = out[op.b], out[op.a]
    else:
        raise ValueError(f"unknown register op {op.kind}")
    return out


def generate_register_state(
    seed: int,
    steps: int,
    *,
    registers: int = 4,
    modulus: int = 7,
) -> dict:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if registers < 2:
        raise ValueError("registers must be >= 2")
    if modulus < 3:
        raise ValueError("modulus must be >= 3")

    rng = random.Random(seed)
    initial = [rng.randrange(modulus) for _ in range(registers)]
    state = list(initial)
    trace = [list(state)]
    ops: list[RegisterOp] = []

    for _ in range(steps):
        op = _random_register_op(rng, registers, modulus)
        ops.append(op)
        state = apply_register_op(state, op, modulus)
        trace.append(list(state))

    query_register = rng.randrange(registers)
    task = {
        "schema_version": 1,
        "family": "register-state",
        "seed": seed,
        "difficulty": {
            "steps": steps,
            "registers": registers,
            "modulus": modulus,
        },
        "input": {
            "initial": initial,
            "operations": [
                {"kind": op.kind, "a": op.a, "b": op.b}
                for op in ops
            ],
            "query_register": query_register,
        },
        "trace": trace,
        "target": state[query_register],
        "final_state": state,
        "oracle": "exact-register-transition",
    }
    task["instance_digest"] = canonical_digest(task)
    return task


def generate_family(
    family: str,
    *,
    seed_start: int,
    count: int,
    difficulty: int,
) -> Iterable[dict]:
    for offset in range(count):
        seed = seed_start + offset
        if family == "parity":
            yield generate_parity(seed, difficulty)
        elif family == "register-state":
            yield generate_register_state(seed, difficulty)
        else:
            raise ValueError(f"unknown family {family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("parity", "register-state"), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--difficulty",
        type=int,
        required=True,
        help="parity length or register-state operation count",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task in generate_family(
            args.family,
            seed_start=args.seed_start,
            count=args.count,
            difficulty=args.difficulty,
        ):
            handle.write(json.dumps(task, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
