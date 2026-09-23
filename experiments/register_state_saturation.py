#!/usr/bin/env python3
"""Direct-only saturation experiment for an enumerable register-state regime.

The regime is intentionally small enough to enumerate the complete ID universe:
2 registers, modulus 3, five deterministic operation tokens, two query choices,
and cumulative operation lengths <=1, <=2, <=3.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PAD = 0
INIT_R0_BASE = 1       # 1..3
INIT_R1_BASE = 4       # 4..6
OP_ADD_R0 = 7
OP_ADD_R1 = 8
OP_COPY_R0_R1 = 9
OP_COPY_R1_R0 = 10
OP_SWAP = 11
QUERY_R0 = 12
QUERY_R1 = 13
CLS = 14
VOCAB_SIZE = 15
MODULUS = 3

OPS = (
    OP_ADD_R0,
    OP_ADD_R1,
    OP_COPY_R0_R1,
    OP_COPY_R1_R0,
    OP_SWAP,
)


def apply_op(state: tuple[int, int], op: int) -> tuple[int, int]:
    r0, r1 = state
    if op == OP_ADD_R0:
        return ((r0 + 1) % MODULUS, r1)
    if op == OP_ADD_R1:
        return (r0, (r1 + 1) % MODULUS)
    if op == OP_COPY_R0_R1:
        return (r1, r1)
    if op == OP_COPY_R1_R0:
        return (r0, r0)
    if op == OP_SWAP:
        return (r1, r0)
    raise ValueError(f"unknown op token {op}")


def execute(initial: tuple[int, int], operations: tuple[int, ...]) -> list[tuple[int, int]]:
    trace = [initial]
    state = initial
    for op in operations:
        state = apply_op(state, op)
        trace.append(state)
    return trace


def encode_example(
    initial: tuple[int, int],
    operations: tuple[int, ...],
    query: int,
) -> tuple[list[int], int]:
    trace = execute(initial, operations)
    final = trace[-1]
    if query == 0:
        query_token = QUERY_R0
        target = final[0]
    elif query == 1:
        query_token = QUERY_R1
        target = final[1]
    else:
        raise ValueError("query must be 0 or 1")

    row = [
        INIT_R0_BASE + initial[0],
        INIT_R1_BASE + initial[1],
        *operations,
        query_token,
        CLS,
    ]
    return row, target


def exhaustive_examples(max_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    rows: list[list[int]] = []
    targets: list[int] = []
    lengths: list[int] = []

    for steps in range(1, max_steps + 1):
        for initial in itertools.product(range(MODULUS), repeat=2):
            for operations in itertools.product(OPS, repeat=steps):
                for query in (0, 1):
                    row, target = encode_example(initial, operations, query)
                    rows.append(row)
                    targets.append(target)
                    lengths.append(len(row))

    width = max(len(row) for row in rows)
    padded = [row + [PAD] * (width - len(row)) for row in rows]
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def sampled_ood_examples(
    seed: int,
    *,
    min_steps: int,
    max_steps: int,
    per_steps: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    out = {}

    for steps in range(min_steps, max_steps + 1):
        seen = set()
        rows: list[list[int]] = []
        targets: list[int] = []
        lengths: list[int] = []
        universe = (MODULUS**2) * (len(OPS) ** steps) * 2
        target_count = min(per_steps, universe)

        while len(rows) < target_count:
            initial = (rng.randrange(MODULUS), rng.randrange(MODULUS))
            operations = tuple(rng.choice(OPS) for _ in range(steps))
            query = rng.randrange(2)
            key = (initial, operations, query)
            if key in seen:
                continue
            seen.add(key)
            row, target = encode_example(initial, operations, query)
            rows.append(row)
            targets.append(target)
            lengths.append(len(row))

        out[steps] = (
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        )

    return out


class RegisterStateTransformer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_positions: int = 24,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_positions, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.classifier = nn.Linear(d_model, MODULUS)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=input_ids.eq(PAD))
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        cls_index = lengths.to(input_ids.device) - 1
        return self.classifier(hidden[batch, cls_index])


@torch.no_grad()
def accuracy(
    model: RegisterStateTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    loader = DataLoader(
        TensorDataset(x, y, lengths),
        batch_size=batch_size,
        shuffle=False,
    )
    correct = 0
    total = 0
    for bx, by, bl in loader:
        pred = model(bx, bl).argmax(dim=-1)
        correct += int((pred == by).sum())
        total += by.numel()
    return correct / total


@dataclass
class StageResult:
    max_steps: int
    universe_size: int
    saturated: bool
    epochs_used: int
    final_accuracy: float
    stable_checks: int
    wall_seconds: float


@dataclass
class SeedResult:
    seed: int
    stages: list[StageResult]
    final_id_accuracy: float
    id_saturated: bool
    ood_accuracy_by_steps: dict[str, float] | None
    mean_ood_accuracy: float | None
    total_wall_seconds: float
    parameter_count: int


def train_seed(
    seed: int,
    *,
    stages: list[int],
    max_epochs_per_stage: int,
    check_interval: int,
    stable_checks_required: int,
    batch_size: int,
    learning_rate: float,
    d_model: int,
    layers: int,
    heads: int,
    ood_min_steps: int,
    ood_max_steps: int,
    ood_per_steps: int,
) -> SeedResult:
    torch.manual_seed(seed)
    random.seed(seed)

    model = RegisterStateTransformer(
        d_model=d_model,
        layers=layers,
        heads=heads,
        max_positions=max(24, ood_max_steps + 8),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    stages_out = []
    total_started = time.perf_counter()

    for max_steps in stages:
        x, y, lengths = exhaustive_examples(max_steps)
        loader = DataLoader(
            TensorDataset(x, y, lengths),
            batch_size=min(batch_size, len(y)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + max_steps * 1000),
        )
        stable_checks = 0
        saturated = False
        final_accuracy = 0.0
        stage_started = time.perf_counter()

        for epoch in range(1, max_epochs_per_stage + 1):
            model.train()
            for bx, by, bl in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(bx, bl)
                loss = loss_fn(logits, by)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite register-state loss")
                loss.backward()
                optimizer.step()

            if epoch % check_interval == 0 or epoch == max_epochs_per_stage:
                final_accuracy = accuracy(model, x, y, lengths, batch_size)
                stable_checks = stable_checks + 1 if final_accuracy == 1.0 else 0
                if stable_checks >= stable_checks_required:
                    saturated = True
                    stages_out.append(
                        StageResult(
                            max_steps=max_steps,
                            universe_size=len(y),
                            saturated=True,
                            epochs_used=epoch,
                            final_accuracy=final_accuracy,
                            stable_checks=stable_checks,
                            wall_seconds=time.perf_counter() - stage_started,
                        )
                    )
                    break

        if not saturated:
            stages_out.append(
                StageResult(
                    max_steps=max_steps,
                    universe_size=len(y),
                    saturated=False,
                    epochs_used=max_epochs_per_stage,
                    final_accuracy=final_accuracy,
                    stable_checks=stable_checks,
                    wall_seconds=time.perf_counter() - stage_started,
                )
            )
            break

    final_x, final_y, final_lengths = exhaustive_examples(stages[-1])
    final_id_accuracy = accuracy(model, final_x, final_y, final_lengths, batch_size)
    id_saturated = (
        len(stages_out) == len(stages)
        and all(stage.saturated for stage in stages_out)
        and final_id_accuracy == 1.0
    )

    ood_by_steps = None
    mean_ood = None
    if id_saturated:
        sampled = sampled_ood_examples(
            seed + 700000,
            min_steps=ood_min_steps,
            max_steps=ood_max_steps,
            per_steps=ood_per_steps,
        )
        ood_by_steps = {}
        for steps, (x, y, lengths) in sampled.items():
            ood_by_steps[str(steps)] = accuracy(model, x, y, lengths, batch_size)
        mean_ood = statistics.mean(ood_by_steps.values())

    return SeedResult(
        seed=seed,
        stages=stages_out,
        final_id_accuracy=final_id_accuracy,
        id_saturated=id_saturated,
        ood_accuracy_by_steps=ood_by_steps,
        mean_ood_accuracy=mean_ood,
        total_wall_seconds=time.perf_counter() - total_started,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--stages", default="1,2,3")
    p.add_argument("--max-epochs-per-stage", type=int, default=500)
    p.add_argument("--check-interval", type=int, default=5)
    p.add_argument("--stable-checks-required", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ood-min-steps", type=int, default=4)
    p.add_argument("--ood-max-steps", type=int, default=6)
    p.add_argument("--ood-per-steps", type=int, default=512)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    stages = [int(x) for x in args.stages.split(",") if x.strip()]
    if sorted(stages) != stages or not stages:
        raise ValueError("stages must be non-empty and ascending")
    if args.ood_min_steps <= stages[-1]:
        raise ValueError("OOD minimum steps must exceed final ID stage")

    results = [
        train_seed(
            seed,
            stages=stages,
            max_epochs_per_stage=args.max_epochs_per_stage,
            check_interval=args.check_interval,
            stable_checks_required=args.stable_checks_required,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            ood_min_steps=args.ood_min_steps,
            ood_max_steps=args.ood_max_steps,
            ood_per_steps=args.ood_per_steps,
        )
        for seed in seeds
    ]

    payload = {
        "schema_version": 1,
        "evidence_class": "benchmark-regime-saturation",
        "family": "register-state-mini-v1",
        "treatment": "direct",
        "settings": {
            "registers": 2,
            "modulus": MODULUS,
            "operation_tokens": list(OPS),
            "seeds": seeds,
            "stages": stages,
            "max_epochs_per_stage": args.max_epochs_per_stage,
            "learning_rate": args.learning_rate,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "ood_min_steps": args.ood_min_steps,
            "ood_max_steps": args.ood_max_steps,
            "ood_per_steps": args.ood_per_steps,
        },
        "results": [
            {
                **asdict(result),
                "stages": [asdict(stage) for stage in result.stages],
            }
            for result in results
        ],
        "all_seeds_id_saturated": all(r.id_saturated for r in results),
        "decision": (
            "register-state-regime-admitted-for-ood-comparison"
            if all(r.id_saturated for r in results)
            else "register-state-regime-not-yet-admitted"
        ),
        "claim_boundary": (
            "This is a complete-universe mini-regime for qualifying register-state "
            "learnability before any latent-vs-control comparison. OOD is emitted "
            "only for seeds that first saturate every ID stage."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
