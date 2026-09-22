#!/usr/bin/env python3
"""Direct-only parity saturation experiment.

The experiment uses the independently implemented formal-state parity task.
Training proceeds through an exhaustive curriculum over all bit strings up to
length 2, then 4, then 8. OOD evaluation is performed only after repeated 100%
accuracy on the complete ID universe through the final stage.

This is benchmark-regime qualification, not a latent-vs-control comparison.
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
BIT0 = 1
BIT1 = 2
CLS = 3
VOCAB_SIZE = 4


def parity(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value ^= bit
    return value


def exhaustive_examples(max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    labels: list[int] = []
    lengths: list[int] = []

    for length in range(1, max_length + 1):
        for bits in itertools.product((0, 1), repeat=length):
            row = [BIT0 + bit for bit in bits] + [CLS]
            rows.append(row)
            labels.append(parity(bits))
            lengths.append(len(row))

    width = max(len(row) for row in rows)
    padded = [row + [PAD] * (width - len(row)) for row in rows]
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def sampled_ood_examples(
    seed: int,
    *,
    min_length: int,
    max_length: int,
    per_length: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    out = {}

    for length in range(min_length, max_length + 1):
        rows: list[list[int]] = []
        labels: list[int] = []
        lengths: list[int] = []
        seen: set[tuple[int, ...]] = set()

        universe = 2**length
        target_count = min(per_length, universe)
        while len(rows) < target_count:
            bits = tuple(rng.randrange(2) for _ in range(length))
            if bits in seen:
                continue
            seen.add(bits)
            rows.append([BIT0 + bit for bit in bits] + [CLS])
            labels.append(parity(bits))
            lengths.append(length + 1)

        out[length] = (
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        )

    return out


class ParityTransformer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_positions: int = 40,
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
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = input_ids.eq(PAD)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        cls_index = lengths.to(input_ids.device) - 1
        cls_hidden = hidden[batch, cls_index]
        return self.classifier(cls_hidden)


@torch.no_grad()
def accuracy(
    model: ParityTransformer,
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
    max_length: int
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
    ood_accuracy_by_length: dict[str, float] | None
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
    ood_min_length: int,
    ood_max_length: int,
    ood_per_length: int,
) -> SeedResult:
    torch.manual_seed(seed)
    random.seed(seed)

    model = ParityTransformer(
        d_model=d_model,
        layers=layers,
        heads=heads,
        max_positions=max(40, ood_max_length + 4),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    loss_fn = nn.CrossEntropyLoss()

    stage_results: list[StageResult] = []
    total_started = time.perf_counter()

    for max_length in stages:
        x, y, lengths = exhaustive_examples(max_length)
        loader = DataLoader(
            TensorDataset(x, y, lengths),
            batch_size=min(batch_size, len(y)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + max_length * 1000),
        )

        stable_checks = 0
        saturated = False
        final_accuracy = 0.0
        stage_started = time.perf_counter()
        epochs_used = 0

        for epoch in range(1, max_epochs_per_stage + 1):
            model.train()
            for bx, by, bl in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(bx, bl)
                loss = loss_fn(logits, by)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite parity training loss")
                loss.backward()
                optimizer.step()

            epochs_used = epoch
            if epoch % check_interval == 0 or epoch == max_epochs_per_stage:
                final_accuracy = accuracy(model, x, y, lengths, batch_size)
                if final_accuracy == 1.0:
                    stable_checks += 1
                else:
                    stable_checks = 0

                if stable_checks >= stable_checks_required:
                    saturated = True
                    break

        stage_results.append(
            StageResult(
                max_length=max_length,
                universe_size=len(y),
                saturated=saturated,
                epochs_used=epochs_used,
                final_accuracy=final_accuracy,
                stable_checks=stable_checks,
                wall_seconds=time.perf_counter() - stage_started,
            )
        )

        if not saturated:
            break

    final_stage = stages[-1]
    final_x, final_y, final_lengths = exhaustive_examples(final_stage)
    final_id_accuracy = accuracy(
        model,
        final_x,
        final_y,
        final_lengths,
        batch_size,
    )
    id_saturated = (
        len(stage_results) == len(stages)
        and all(stage.saturated for stage in stage_results)
        and final_id_accuracy == 1.0
    )

    ood_by_length = None
    mean_ood = None

    if id_saturated:
        sampled = sampled_ood_examples(
            seed + 900000,
            min_length=ood_min_length,
            max_length=ood_max_length,
            per_length=ood_per_length,
        )
        ood_by_length = {}
        for length, (x, y, lengths) in sampled.items():
            ood_by_length[str(length)] = accuracy(
                model,
                x,
                y,
                lengths,
                batch_size,
            )
        mean_ood = statistics.mean(ood_by_length.values())

    return SeedResult(
        seed=seed,
        stages=stage_results,
        final_id_accuracy=final_id_accuracy,
        id_saturated=id_saturated,
        ood_accuracy_by_length=ood_by_length,
        mean_ood_accuracy=mean_ood,
        total_wall_seconds=time.perf_counter() - total_started,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--stages", default="2,4,8")
    parser.add_argument("--max-epochs-per-stage", type=int, default=300)
    parser.add_argument("--check-interval", type=int, default=5)
    parser.add_argument("--stable-checks-required", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ood-min-length", type=int, default=9)
    parser.add_argument("--ood-max-length", type=int, default=16)
    parser.add_argument("--ood-per-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    stages = [int(x) for x in args.stages.split(",") if x.strip()]
    if sorted(stages) != stages or not stages:
        raise ValueError("stages must be non-empty and ascending")
    if args.ood_min_length <= stages[-1]:
        raise ValueError("OOD minimum length must be greater than final ID stage")

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
            ood_min_length=args.ood_min_length,
            ood_max_length=args.ood_max_length,
            ood_per_length=args.ood_per_length,
        )
        for seed in seeds
    ]

    all_saturated = all(result.id_saturated for result in results)
    payload = {
        "schema_version": 1,
        "evidence_class": "benchmark-regime-saturation",
        "family": "parity",
        "treatment": "direct",
        "settings": {
            "seeds": seeds,
            "stages": stages,
            "max_epochs_per_stage": args.max_epochs_per_stage,
            "check_interval": args.check_interval,
            "stable_checks_required": args.stable_checks_required,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "ood_min_length": args.ood_min_length,
            "ood_max_length": args.ood_max_length,
            "ood_per_length": args.ood_per_length,
        },
        "results": [
            {
                **asdict(result),
                "stages": [asdict(stage) for stage in result.stages],
            }
            for result in results
        ],
        "all_seeds_id_saturated": all_saturated,
        "decision": (
            "parity-regime-admitted-for-ood-comparison"
            if all_saturated
            else "parity-regime-not-yet-admitted"
        ),
        "claim_boundary": (
            "OOD metrics are emitted only for seeds that repeatedly reached 100% "
            "accuracy on the complete ID universe. This run does not compare latent "
            "reasoning treatments."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
