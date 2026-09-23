#!/usr/bin/env python3
"""WP6a fixed-K parity representation comparison.

Direct is the frozen baseline reference. Serial-control and latent recurrence
use identical model weights, optimizer/data configuration, and exactly matched
transformer forward-call counts for K extra recurrent steps.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.parity_saturation import (
    BIT0,
    BIT1,
    CLS,
    PAD,
    VOCAB_SIZE,
    exhaustive_examples,
    sampled_ood_examples,
)


class RecurrentParityTransformer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_positions: int = 48,
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
        # Keep shared-module initialization order identical to the admitted
        # direct ParityTransformer. The control-only pause parameter is created
        # after the classifier so it cannot perturb the frozen baseline RNG stream.
        self.classifier = nn.Linear(d_model, 2)
        self.pause_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pause_embedding, std=0.02)

    def _encode(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            embeddings.shape[1],
            dtype=torch.long,
            device=embeddings.device,
        ).unsqueeze(0)
        hidden = embeddings + self.position_embedding(positions)
        return self.encoder(hidden, src_key_padding_mask=padding_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        *,
        treatment: str,
        recurrent_steps: int,
    ) -> tuple[torch.Tensor, int]:
        base_embeddings = self.token_embedding(input_ids)
        base_padding = input_ids.eq(PAD)
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        cls_index = lengths.to(input_ids.device) - 1

        if treatment == "direct":
            hidden = self._encode(base_embeddings, base_padding)
            return self.classifier(hidden[batch, cls_index]), 1

        if treatment not in {"serial-control", "latent"}:
            raise ValueError(f"unknown treatment {treatment}")
        if recurrent_steps < 1:
            raise ValueError("recurrent treatment requires recurrent_steps >= 1")

        current = base_embeddings
        padding_mask = base_padding
        forward_calls = 0

        for step in range(recurrent_steps):
            hidden = self._encode(current, padding_mask)
            forward_calls += 1

            if treatment == "latent":
                if step == 0:
                    slot = hidden[batch, cls_index].unsqueeze(1)
                else:
                    slot = hidden[:, -1:, :]
            else:
                slot = self.pause_embedding.expand(input_ids.shape[0], -1, -1)

            current = torch.cat((current, slot), dim=1)
            padding_mask = torch.cat(
                (
                    padding_mask,
                    torch.zeros(
                        (input_ids.shape[0], 1),
                        dtype=torch.bool,
                        device=input_ids.device,
                    ),
                ),
                dim=1,
            )

        hidden = self._encode(current, padding_mask)
        forward_calls += 1
        return self.classifier(hidden[:, -1, :]), forward_calls


@torch.no_grad()
def accuracy(
    model: RecurrentParityTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    *,
    treatment: str,
    recurrent_steps: int,
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
    observed_calls = None
    for bx, by, bl in loader:
        logits, calls = model(
            bx,
            bl,
            treatment=treatment,
            recurrent_steps=recurrent_steps,
        )
        observed_calls = calls
        correct += int((logits.argmax(dim=-1) == by).sum())
        total += by.numel()
    if observed_calls is None:
        raise RuntimeError("empty evaluation set")
    return correct / total


@dataclass
class TreatmentSeedResult:
    treatment: str
    seed: int
    recurrent_steps: int
    forward_calls_per_batch: int
    stage_epochs: list[int]
    id_saturated: bool
    final_id_accuracy: float
    ood_accuracy_by_length: dict[str, float] | None
    mean_ood_accuracy: float | None
    training_wall_seconds: float
    parameter_count: int


def train_one(
    treatment: str,
    seed: int,
    *,
    recurrent_steps: int,
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
) -> TreatmentSeedResult:
    torch.manual_seed(seed)
    random.seed(seed)

    model = RecurrentParityTransformer(
        d_model=d_model,
        layers=layers,
        heads=heads,
        max_positions=max(48, ood_max_length + recurrent_steps + 4),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    stage_epochs: list[int] = []
    training_started = time.perf_counter()
    observed_calls = None
    all_stages_saturated = True

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

        for epoch in range(1, max_epochs_per_stage + 1):
            model.train()
            for bx, by, bl in loader:
                optimizer.zero_grad(set_to_none=True)
                logits, calls = model(
                    bx,
                    bl,
                    treatment=treatment,
                    recurrent_steps=recurrent_steps,
                )
                observed_calls = calls
                loss = loss_fn(logits, by)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite comparison loss")
                loss.backward()
                optimizer.step()

            if epoch % check_interval == 0 or epoch == max_epochs_per_stage:
                id_accuracy = accuracy(
                    model,
                    x,
                    y,
                    lengths,
                    treatment=treatment,
                    recurrent_steps=recurrent_steps,
                    batch_size=batch_size,
                )
                stable_checks = stable_checks + 1 if id_accuracy == 1.0 else 0
                if stable_checks >= stable_checks_required:
                    saturated = True
                    stage_epochs.append(epoch)
                    break

        if not saturated:
            stage_epochs.append(max_epochs_per_stage)
            all_stages_saturated = False
            break

    final_x, final_y, final_lengths = exhaustive_examples(stages[-1])
    final_id_accuracy = accuracy(
        model,
        final_x,
        final_y,
        final_lengths,
        treatment=treatment,
        recurrent_steps=recurrent_steps,
        batch_size=batch_size,
    )
    id_saturated = (
        all_stages_saturated
        and len(stage_epochs) == len(stages)
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
                treatment=treatment,
                recurrent_steps=recurrent_steps,
                batch_size=batch_size,
            )
        mean_ood = statistics.mean(ood_by_length.values())

    if observed_calls is None:
        raise RuntimeError("training produced no forward calls")

    return TreatmentSeedResult(
        treatment=treatment,
        seed=seed,
        recurrent_steps=recurrent_steps,
        forward_calls_per_batch=observed_calls,
        stage_epochs=stage_epochs,
        id_saturated=id_saturated,
        final_id_accuracy=final_id_accuracy,
        ood_accuracy_by_length=ood_by_length,
        mean_ood_accuracy=mean_ood,
        training_wall_seconds=time.perf_counter() - training_started,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )


def paired_deltas(results: list[TreatmentSeedResult]) -> dict:
    by_treatment = {
        treatment: {r.seed: r for r in results if r.treatment == treatment}
        for treatment in ("direct", "serial-control", "latent")
    }
    seeds = sorted(set.intersection(*(set(x) for x in by_treatment.values())))
    out = {"paired_seeds": seeds, "comparisons": {}}

    for reference in ("direct", "serial-control"):
        deltas = []
        by_seed = {}
        for seed in seeds:
            latent = by_treatment["latent"][seed]
            base = by_treatment[reference][seed]
            if latent.mean_ood_accuracy is None or base.mean_ood_accuracy is None:
                continue
            delta = latent.mean_ood_accuracy - base.mean_ood_accuracy
            deltas.append(delta)
            by_seed[str(seed)] = delta
        out["comparisons"][f"latent_minus_{reference}_mean_ood"] = {
            "by_seed": by_seed,
            "mean": statistics.mean(deltas) if deltas else None,
            "pstdev": statistics.pstdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None),
            "all_positive": bool(deltas) and all(x > 0 for x in deltas),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--stages", default="2,4,8")
    p.add_argument("--recurrent-steps", type=int, default=2)
    p.add_argument("--max-epochs-per-stage", type=int, default=1000)
    p.add_argument("--check-interval", type=int, default=5)
    p.add_argument("--stable-checks-required", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ood-min-length", type=int, default=9)
    p.add_argument("--ood-max-length", type=int, default=16)
    p.add_argument("--ood-per-length", type=int, default=256)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    stages = [int(x) for x in args.stages.split(",") if x.strip()]
    treatments = ("direct", "serial-control", "latent")

    results = []
    for treatment in treatments:
        for seed in seeds:
            results.append(
                train_one(
                    treatment,
                    seed,
                    recurrent_steps=args.recurrent_steps,
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
            )

    groups = {
        treatment: [r for r in results if r.treatment == treatment]
        for treatment in treatments
    }
    aggregate = {}
    for treatment, rows in groups.items():
        ood = [r.mean_ood_accuracy for r in rows if r.mean_ood_accuracy is not None]
        aggregate[treatment] = {
            "all_seeds_id_saturated": all(r.id_saturated for r in rows),
            "mean_final_id_accuracy": statistics.mean(r.final_id_accuracy for r in rows),
            "mean_ood_accuracy": statistics.mean(ood) if ood else None,
            "mean_training_wall_seconds": statistics.mean(r.training_wall_seconds for r in rows),
            "mean_total_epochs": statistics.mean(sum(r.stage_epochs) for r in rows),
            "forward_calls_per_batch": rows[0].forward_calls_per_batch,
            "parameter_count": rows[0].parameter_count,
        }

    if aggregate["serial-control"]["forward_calls_per_batch"] != aggregate["latent"]["forward_calls_per_batch"]:
        raise RuntimeError("serial and latent forward-call counts are not matched")

    payload = {
        "schema_version": 1,
        "evidence_class": "fixed-k-reasoning-representation-comparison",
        "family": "parity",
        "settings": {
            "seeds": seeds,
            "stages": stages,
            "recurrent_steps": args.recurrent_steps,
            "max_epochs_per_stage": args.max_epochs_per_stage,
            "learning_rate": args.learning_rate,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "ood_min_length": args.ood_min_length,
            "ood_max_length": args.ood_max_length,
            "ood_per_length": args.ood_per_length,
        },
        "results": [asdict(r) for r in results],
        "aggregate": aggregate,
        "paired_analysis": paired_deltas(results),
        "comparison_valid": (
            aggregate["serial-control"]["all_seeds_id_saturated"]
            and aggregate["latent"]["all_seeds_id_saturated"]
        ),
        "claim_boundary": (
            "Direct is a lower-compute frozen reference. Serial-control and latent "
            "use identical K and matched transformer forward-call counts. OOD metrics "
            "are emitted only for treatments/seeds that first saturate the complete ID universe."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
