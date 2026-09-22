#!/usr/bin/env python3
"""CPU-only toy learning experiment for latent-reasoning mechanics.

This experiment is deliberately small and synthetic. It tests whether a latent
recurrence treatment can be trained end-to-end on an exact-state sequential
task under the same number of transformer forward calls as a serial-control
treatment. Results are mechanism/toy-system evidence only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STATE_COUNT = 8
ACTION_DELTAS = (-3, -1, 1, 3)
ACTION_TOKEN_BASE = STATE_COUNT
VOCAB_SIZE = STATE_COUNT + len(ACTION_DELTAS)


def make_examples(seed: int, count: int, action_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    rows: list[list[int]] = []
    labels: list[int] = []

    for _ in range(count):
        state = rng.randrange(STATE_COUNT)
        row = [state]
        for _ in range(action_steps):
            action_index = rng.randrange(len(ACTION_DELTAS))
            row.append(ACTION_TOKEN_BASE + action_index)
            state = (state + ACTION_DELTAS[action_index]) % STATE_COUNT
        rows.append(row)
        labels.append(state)

    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class TinyReasoner(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 4, dim_feedforward: int = 64):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.position_embedding = nn.Embedding(32, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.pause_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pause_embedding, std=0.02)
        self.classifier = nn.Linear(d_model, STATE_COUNT)

    def _encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            embeddings.shape[1], device=embeddings.device, dtype=torch.long
        ).unsqueeze(0)
        return self.encoder(embeddings + self.position_embedding(positions))

    def forward(
        self,
        input_ids: torch.Tensor,
        treatment: str,
        latent_steps: int,
    ) -> tuple[torch.Tensor, int]:
        embeddings = self.token_embedding(input_ids)

        if treatment == "direct":
            hidden = self._encode(embeddings)
            return self.classifier(hidden[:, -1, :]), 1

        if treatment not in {"serial-control", "latent"}:
            raise ValueError(f"unknown treatment: {treatment}")

        forward_calls = 0
        current = embeddings
        for _ in range(latent_steps):
            hidden = self._encode(current)
            forward_calls += 1
            if treatment == "latent":
                slot = hidden[:, -1:, :]
            else:
                slot = self.pause_embedding.expand(input_ids.shape[0], -1, -1)
            current = torch.cat((current, slot), dim=1)

        hidden = self._encode(current)
        forward_calls += 1
        return self.classifier(hidden[:, -1, :]), forward_calls


@dataclass
class TreatmentResult:
    treatment: str
    seed: int
    latent_steps: int
    forward_calls: int
    train_accuracy: float
    validation_accuracy: float
    ood_accuracy: float
    wall_seconds: float
    parameter_count: int
    learning_curve: list[dict[str, float]]


@torch.no_grad()
def accuracy(
    model: TinyReasoner,
    loader: DataLoader,
    treatment: str,
    latent_steps: int,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for input_ids, target in loader:
        logits, _ = model(input_ids, treatment, latent_steps)
        correct += int((logits.argmax(dim=-1) == target).sum())
        total += target.numel()
    return correct / total


def snapshot(
    epoch: int,
    model: TinyReasoner,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ood_loader: DataLoader,
    treatment: str,
    latent_steps: int,
) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "train_accuracy": accuracy(model, train_loader, treatment, latent_steps),
        "validation_accuracy": accuracy(model, val_loader, treatment, latent_steps),
        "ood_accuracy": accuracy(model, ood_loader, treatment, latent_steps),
    }


def train_one(
    treatment: str,
    seed: int,
    *,
    latent_steps: int,
    epochs: int,
    train_count: int,
    eval_count: int,
    batch_size: int,
    report_epochs: set[int],
) -> TreatmentResult:
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
    train_eval_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    ood_loader = DataLoader(TensorDataset(ood_x, ood_y), batch_size=batch_size)

    model = TinyReasoner()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    started = time.perf_counter()
    observed_calls = None
    curve: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for input_ids, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, forward_calls = model(input_ids, treatment, latent_steps)
            observed_calls = forward_calls
            loss = loss_fn(logits, target)
            if not math.isfinite(float(loss)):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()

        if epoch in report_epochs:
            curve.append(
                snapshot(
                    epoch,
                    model,
                    train_eval_loader,
                    val_loader,
                    ood_loader,
                    treatment,
                    latent_steps,
                )
            )

    elapsed = time.perf_counter() - started
    assert observed_calls is not None

    if curve and int(curve[-1]["epoch"]) == epochs:
        final = curve[-1]
    else:
        final = snapshot(
            epochs,
            model,
            train_eval_loader,
            val_loader,
            ood_loader,
            treatment,
            latent_steps,
        )
        curve.append(final)

    return TreatmentResult(
        treatment=treatment,
        seed=seed,
        latent_steps=latent_steps,
        forward_calls=observed_calls,
        train_accuracy=final["train_accuracy"],
        validation_accuracy=final["validation_accuracy"],
        ood_accuracy=final["ood_accuracy"],
        wall_seconds=elapsed,
        parameter_count=sum(p.numel() for p in model.parameters()),
        learning_curve=curve,
    )


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "pstdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate(results: Iterable[TreatmentResult]) -> dict[str, dict[str, object]]:
    groups: dict[str, list[TreatmentResult]] = {}
    for result in results:
        groups.setdefault(result.treatment, []).append(result)

    summary: dict[str, dict[str, object]] = {}
    for treatment, rows in groups.items():
        summary[treatment] = {
            "train_accuracy": summarize([r.train_accuracy for r in rows]),
            "validation_accuracy": summarize([r.validation_accuracy for r in rows]),
            "ood_accuracy": summarize([r.ood_accuracy for r in rows]),
            "wall_seconds": summarize([r.wall_seconds for r in rows]),
            "forward_calls": rows[0].forward_calls,
        }
    return summary


def paired_analysis(results: list[TreatmentResult]) -> dict[str, object]:
    by_treatment = {
        treatment: {r.seed: r for r in results if r.treatment == treatment}
        for treatment in ("direct", "serial-control", "latent")
    }
    seeds = sorted(set.intersection(*(set(rows) for rows in by_treatment.values())))

    comparisons: dict[str, object] = {}
    for reference in ("direct", "serial-control"):
        for metric in ("validation_accuracy", "ood_accuracy"):
            deltas = [
                getattr(by_treatment["latent"][seed], metric)
                - getattr(by_treatment[reference][seed], metric)
                for seed in seeds
            ]
            comparisons[f"latent_minus_{reference}_{metric}"] = {
                "by_seed": {str(seed): delta for seed, delta in zip(seeds, deltas)},
                "mean": statistics.mean(deltas),
                "pstdev": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
                "all_positive": all(delta > 0 for delta in deltas),
            }

    return {"paired_seeds": seeds, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--report-epochs", default="1,2,4,8")
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-steps", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    requested_reports = {
        int(item) for item in args.report_epochs.split(",") if item.strip()
    }
    report_epochs = {epoch for epoch in requested_reports if 1 <= epoch <= args.epochs}
    report_epochs.add(args.epochs)

    treatments = ("direct", "serial-control", "latent")
    results: list[TreatmentResult] = []

    for treatment in treatments:
        for seed in seeds:
            results.append(
                train_one(
                    treatment,
                    seed,
                    latent_steps=args.latent_steps,
                    epochs=args.epochs,
                    train_count=args.train_count,
                    eval_count=args.eval_count,
                    batch_size=args.batch_size,
                    report_epochs=report_epochs,
                )
            )

    aggregated = aggregate(results)
    best_mean_train = max(
        float(aggregated[treatment]["train_accuracy"]["mean"])
        for treatment in treatments
    )
    if best_mean_train >= 0.90:
        learnability = "well-learned"
    elif best_mean_train >= 0.60:
        learnability = "partially-learned"
    else:
        learnability = "underfit"

    payload = {
        "schema_version": 2,
        "evidence_class": "toy-system-learning",
        "task": {
            "name": "modular-walk-v0",
            "state_count": STATE_COUNT,
            "action_deltas": ACTION_DELTAS,
            "train_action_steps": 6,
            "ood_action_steps": 10,
        },
        "settings": {
            "seeds": seeds,
            "epochs": args.epochs,
            "report_epochs": sorted(report_epochs),
            "train_count": args.train_count,
            "eval_count": args.eval_count,
            "batch_size": args.batch_size,
            "latent_steps": args.latent_steps,
        },
        "results": [asdict(result) for result in results],
        "aggregate": aggregated,
        "paired_analysis": paired_analysis(results),
        "learnability_status": learnability,
        "claim_boundary": (
            "This experiment is toy-system mechanism evidence. It does not establish "
            "that latent reasoning improves pretrained language-model capability."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
