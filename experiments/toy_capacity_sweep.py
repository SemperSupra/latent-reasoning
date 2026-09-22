#!/usr/bin/env python3
"""WP5c direct-baseline capacity sweep.

Hold task, optimizer, data, depth, and training schedule fixed while varying
only transformer width. This establishes whether the toy task is learnable
before any further latent-vs-control comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.toy_modular_walk import make_examples, STATE_COUNT, VOCAB_SIZE


class DirectReasoner(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.position_embedding = nn.Embedding(32, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Linear(d_model, STATE_COUNT)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.encoder(embeddings)
        return self.classifier(hidden[:, -1, :])


@dataclass
class Run:
    width: int
    seed: int
    train_accuracy: float
    validation_accuracy: float
    ood_accuracy: float
    wall_seconds: float
    parameter_count: int


@torch.no_grad()
def accuracy(model: DirectReasoner, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        pred = model(x).argmax(dim=-1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / total


def train_one(
    width: int,
    seed: int,
    *,
    epochs: int,
    train_count: int,
    eval_count: int,
    batch_size: int,
) -> Run:
    torch.manual_seed(seed)
    random.seed(seed)

    train_x, train_y = make_examples(seed + 1000, train_count, action_steps=6)
    val_x, val_y = make_examples(seed + 2000, eval_count, action_steps=6)
    ood_x, ood_y = make_examples(seed + 3000, eval_count, action_steps=10)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_eval = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    ood_loader = DataLoader(TensorDataset(ood_x, ood_y), batch_size=batch_size)

    model = DirectReasoner(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if not math.isfinite(float(loss)):
                raise RuntimeError("non-finite loss")
            loss.backward()
            optimizer.step()

    elapsed = time.perf_counter() - started
    return Run(
        width=width,
        seed=seed,
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
    p.add_argument("--widths", default="32,64,128")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--train-count", type=int, default=768)
    p.add_argument("--eval-count", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    runs: list[Run] = []
    for width in widths:
        for seed in seeds:
            runs.append(
                train_one(
                    width,
                    seed,
                    epochs=args.epochs,
                    train_count=args.train_count,
                    eval_count=args.eval_count,
                    batch_size=args.batch_size,
                )
            )

    aggregate = {}
    for width in widths:
        rows = [r for r in runs if r.width == width]
        aggregate[str(width)] = {
            "train_accuracy": stats([r.train_accuracy for r in rows]),
            "validation_accuracy": stats([r.validation_accuracy for r in rows]),
            "ood_accuracy": stats([r.ood_accuracy for r in rows]),
            "wall_seconds": stats([r.wall_seconds for r in rows]),
            "parameter_count": rows[0].parameter_count,
        }

    qualified = [
        width for width in widths
        if aggregate[str(width)]["train_accuracy"]["mean"] >= 0.90
        and aggregate[str(width)]["validation_accuracy"]["mean"] >= 0.75
    ]

    payload = {
        "schema_version": 1,
        "evidence_class": "toy-baseline-learnability",
        "controlled_variable": "transformer_width",
        "fixed": {
            "layers": 1,
            "heads": 4,
            "feedforward_multiplier": 2,
            "train_action_steps": 6,
            "ood_action_steps": 10,
            "optimizer": "AdamW(lr=3e-3, weight_decay=0.01)",
            "epochs": args.epochs,
            "train_count": args.train_count,
            "eval_count": args.eval_count,
            "batch_size": args.batch_size,
        },
        "widths": widths,
        "seeds": seeds,
        "runs": [asdict(r) for r in runs],
        "aggregate": aggregate,
        "qualified_widths": qualified,
        "decision": (
            "freeze-smallest-qualified-width"
            if qualified
            else "width-alone-did-not-establish-learnability"
        ),
        "claim_boundary": (
            "This sweep qualifies a toy direct baseline only; it does not compare "
            "latent reasoning mechanisms or establish pretrained-model capability."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
