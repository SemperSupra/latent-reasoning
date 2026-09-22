#!/usr/bin/env python3
"""WP5e direct-baseline data-pool-size sweep.

Hold architecture, optimizer, batch size, and optimizer-update count fixed while
varying only the number of unique training examples available.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.toy_modular_walk import (
    ACTION_DELTAS,
    ACTION_TOKEN_BASE,
    STATE_COUNT,
    VOCAB_SIZE,
    make_examples,
)


FIXED_WIDTH = 64
FIXED_LAYERS = 4
FIXED_HEADS = 4
FIXED_FFN_MULTIPLIER = 2
TRAIN_ACTION_STEPS = 6
OOD_ACTION_STEPS = 10


def make_unique_examples(
    seed: int,
    count: int,
    action_steps: int = TRAIN_ACTION_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample unique state/action sequences from the finite procedural universe."""
    universe = STATE_COUNT * (len(ACTION_DELTAS) ** action_steps)
    if count > universe:
        raise ValueError(f"count {count} exceeds unique universe {universe}")

    rng = random.Random(seed)
    chosen = rng.sample(range(universe), count)

    rows: list[list[int]] = []
    labels: list[int] = []
    base = len(ACTION_DELTAS)

    for code in chosen:
        start = code % STATE_COUNT
        action_code = code // STATE_COUNT
        action_indices = []
        for _ in range(action_steps):
            action_indices.append(action_code % base)
            action_code //= base

        state = start
        row = [start]
        for action_index in action_indices:
            row.append(ACTION_TOKEN_BASE + action_index)
            state = (state + ACTION_DELTAS[action_index]) % STATE_COUNT

        rows.append(row)
        labels.append(state)

    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DirectDataReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, FIXED_WIDTH)
        self.position_embedding = nn.Embedding(32, FIXED_WIDTH)
        layer = nn.TransformerEncoderLayer(
            d_model=FIXED_WIDTH,
            nhead=FIXED_HEADS,
            dim_feedforward=FIXED_WIDTH * FIXED_FFN_MULTIPLIER,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=FIXED_LAYERS)
        self.classifier = nn.Linear(FIXED_WIDTH, STATE_COUNT)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.encoder(embeddings)
        return self.classifier(hidden[:, -1, :])


@dataclass
class Run:
    train_count: int
    seed: int
    updates: int
    examples_seen: int
    train_accuracy: float
    validation_accuracy: float
    ood_accuracy: float
    wall_seconds: float
    parameter_count: int


@torch.no_grad()
def accuracy(model: DirectDataReasoner, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        pred = model(x).argmax(dim=-1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / total


def train_one(
    train_count: int,
    seed: int,
    *,
    updates: int,
    eval_count: int,
    batch_size: int,
) -> Run:
    torch.manual_seed(seed)
    random.seed(seed)

    train_x, train_y = make_unique_examples(seed + 1000, train_count)
    val_x, val_y = make_examples(seed + 2000, eval_count, action_steps=TRAIN_ACTION_STEPS)
    ood_x, ood_y = make_examples(seed + 3000, eval_count, action_steps=OOD_ACTION_STEPS)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
    )
    train_eval = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    ood_loader = DataLoader(TensorDataset(ood_x, ood_y), batch_size=batch_size)

    model = DirectDataReasoner()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    started = time.perf_counter()
    iterator = iter(train_loader)

    for _ in range(updates):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not math.isfinite(float(loss)):
            raise RuntimeError("non-finite loss")
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - started

    return Run(
        train_count=train_count,
        seed=seed,
        updates=updates,
        examples_seen=updates * batch_size,
        train_accuracy=accuracy(model, train_eval),
        validation_accuracy=accuracy(model, val_loader),
        ood_accuracy=accuracy(model, ood_loader),
        wall_seconds=elapsed,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "pstdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-counts", default="768,3072,12288")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--updates", type=int, default=240)
    p.add_argument("--eval-count", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    train_counts = [int(x) for x in args.train_counts.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    runs: list[Run] = []
    for train_count in train_counts:
        for seed in seeds:
            runs.append(
                train_one(
                    train_count,
                    seed,
                    updates=args.updates,
                    eval_count=args.eval_count,
                    batch_size=args.batch_size,
                )
            )

    aggregate = {}
    for train_count in train_counts:
        rows = [r for r in runs if r.train_count == train_count]
        aggregate[str(train_count)] = {
            "train_accuracy": stats([r.train_accuracy for r in rows]),
            "validation_accuracy": stats([r.validation_accuracy for r in rows]),
            "ood_accuracy": stats([r.ood_accuracy for r in rows]),
            "wall_seconds": stats([r.wall_seconds for r in rows]),
            "parameter_count": rows[0].parameter_count,
            "examples_seen_per_run": rows[0].examples_seen,
        }

    qualified = [
        count for count in train_counts
        if aggregate[str(count)]["train_accuracy"]["mean"] >= 0.90
        and aggregate[str(count)]["validation_accuracy"]["mean"] >= 0.75
    ]

    baseline_val = aggregate[str(train_counts[0])]["validation_accuracy"]["mean"]
    val_gain = {
        str(count): aggregate[str(count)]["validation_accuracy"]["mean"] - baseline_val
        for count in train_counts
    }

    payload = {
        "schema_version": 1,
        "evidence_class": "toy-baseline-learnability",
        "controlled_variable": "unique_training_pool_size",
        "fixed": {
            "width": FIXED_WIDTH,
            "layers": FIXED_LAYERS,
            "heads": FIXED_HEADS,
            "feedforward_multiplier": FIXED_FFN_MULTIPLIER,
            "train_action_steps": TRAIN_ACTION_STEPS,
            "ood_action_steps": OOD_ACTION_STEPS,
            "optimizer": "AdamW(lr=3e-3, weight_decay=0.01)",
            "optimizer_updates": args.updates,
            "batch_size": args.batch_size,
            "examples_seen_per_run": args.updates * args.batch_size,
            "eval_count": args.eval_count,
        },
        "train_counts": train_counts,
        "seeds": seeds,
        "runs": [asdict(r) for r in runs],
        "aggregate": aggregate,
        "validation_gain_vs_smallest_pool": val_gain,
        "qualified_train_counts": qualified,
        "decision": (
            "freeze-smallest-qualified-data-pool"
            if qualified
            else "data-pool-alone-did-not-establish-learnability"
        ),
        "claim_boundary": (
            "This sweep qualifies a toy direct baseline only; optimizer-update count "
            "is fixed so data diversity is not confounded with extra training compute."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
