#!/usr/bin/env python3
"""WP7a public-safe synthetic causal-slot smoke; no trained-model efficacy claim."""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
from torch.nn import functional as F

from experiments.register_state_representation_comparison import (
    RecurrentRegisterStateTransformer,
)
from experiments.register_state_saturation import (
    exhaustive_examples,
    sampled_ood_examples,
)


def donor_indices(labels: torch.Tensor, *, same_label: bool) -> torch.Tensor:
    """Deterministic oracle-selected pairs for an evaluation-only causal diagnostic."""
    y = labels.tolist()
    out = []
    for i, label in enumerate(y):
        candidates = [
            j for j, other in enumerate(y)
            if i != j and ((other == label) == same_label)
        ]
        if not candidates:
            raise ValueError("cannot construct causal donors for every label")
        out.append(candidates[i % len(candidates)])
    return torch.tensor(out, dtype=torch.long, device=labels.device)


@torch.no_grad()
def patched_logits(
    model: RecurrentRegisterStateTransformer,
    x: torch.Tensor,
    lengths: torch.Tensor,
    *,
    kind: str,
    donor: torch.Tensor | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Diagnostic K=1 path with a patch at the exact CLS-to-latent slot boundary."""
    model.eval()
    base = model.token_embedding(x)
    mask = x.eq(0)
    batch = torch.arange(x.shape[0], device=x.device)
    h = model._encode(base, mask)
    slot = h[batch, lengths.to(x.device) - 1].unsqueeze(1)
    if kind in {"untouched", "clone-noop"}:
        slot = slot.clone() if kind == "clone-noop" else slot
    elif kind in {"same-label", "different-label"}:
        if donor is None or donor.shape != (x.shape[0],):
            raise ValueError("donor shape mismatch")
        slot = slot.index_select(0, donor)
    elif kind == "zero":
        slot = torch.zeros_like(slot)
    elif kind == "norm-matched-noise":
        # CPU-only deterministic noise; magnitude matches each unmodified latent.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(slot.shape, generator=generator, dtype=slot.dtype).to(slot.device)
        source_norm = slot.norm(dim=-1, keepdim=True)
        slot = noise * (source_norm / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    else:
        raise ValueError(f"unsupported patch: {kind}")
    current = torch.cat((base, slot), dim=1)
    patched_mask = torch.cat(
        (mask, torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device)),
        dim=1,
    )
    final = model._encode(current, patched_mask)
    return model.classifier(final[:, -1, :])


def fit(seed: int, treatment: str, *, epochs: int, learning_rate: float):
    torch.manual_seed(seed)
    random.seed(seed)
    x, y, lengths = exhaustive_examples(1)
    model = RecurrentRegisterStateTransformer(
        d_model=32, layers=1, heads=4, max_positions=24,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    saturated = False
    stable = 0
    last_accuracy = 0.0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, calls = model(x, lengths, treatment=treatment, recurrent_steps=1)
        if calls != 2:
            raise RuntimeError("K=1 must consume two full encoder passes")
        loss = F.cross_entropy(logits, y)
        if not math.isfinite(loss.item()):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits, _ = model(x, lengths, treatment=treatment, recurrent_steps=1)
            last_accuracy = (logits.argmax(-1) == y).float().mean().item()
            stable = stable + 1 if last_accuracy == 1.0 else 0
            if stable >= 2:
                saturated = True
                break
    return model, {
        "treatment": treatment,
        "seed": seed,
        "trained_epochs": epoch,
        "id_accuracy": last_accuracy,
        "id_saturated": saturated,
        "encoder_passes_per_input": 2,
        "train_wall_seconds": time.perf_counter() - start,
    }


def evaluate(model, x, y, lengths, *, seed):
    model.eval()
    with torch.no_grad():
        original, calls = model(x, lengths, treatment="latent", recurrent_steps=1)
        clone = patched_logits(model, x, lengths, kind="clone-noop")
        if calls != 2 or not torch.equal(original, clone):
            raise RuntimeError("clone-noop failed bitwise equality against original forward")
        baseline_prediction = original.argmax(-1)
        same = donor_indices(y, same_label=True)
        different = donor_indices(y, same_label=False)
        observed = {}
        for kind in (
            "untouched", "clone-noop", "same-label",
            "different-label", "zero", "norm-matched-noise",
        ):
            donor = same if kind == "same-label" else different if kind == "different-label" else None
            logits = patched_logits(model, x, lengths, kind=kind, donor=donor, seed=seed)
            prediction = logits.argmax(-1)
            observed[kind] = {
                "accuracy": (prediction == y).float().mean().item(),
                "prediction_flip_rate": (prediction != baseline_prediction).float().mean().item(),
            }
        return observed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0")
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--ood-samples", type=int, default=64)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds or len(seeds) > 3 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be 1–3 unique integers")
    if not 1 <= args.epochs <= 300 or not 8 <= args.ood_samples <= 128:
        raise ValueError("bounded epochs/samples violated")
    if not 1 <= args.threads <= 4 or not 0 < args.learning_rate <= 0.01:
        raise ValueError("bounded threads/learning-rate violated")
    torch.set_num_threads(args.threads)
    records = []
    for seed in seeds:
        latent, latent_fit = fit(
            seed, "latent", epochs=args.epochs, learning_rate=args.learning_rate,
        )
        serial, serial_fit = fit(
            seed, "serial-control", epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        x, y, lengths = sampled_ood_examples(
            seed + 700000,
            min_steps=4,
            max_steps=4,
            per_steps=args.ood_samples,
        )[4]
        with torch.no_grad():
            serial_logits, serial_calls = serial(
                x, lengths, treatment="serial-control", recurrent_steps=1,
            )
        if serial_calls != 2:
            raise RuntimeError("serial control call count differs")
        # Always run the interface smoke, but withhold causal interpretation
        # unless *both* treatments saturated complete in-domain examples.
        interventions = evaluate(latent, x, y, lengths, seed=seed)
        records.append({
            "seed": seed,
            "latent_fit": latent_fit,
            "serial_fit": serial_fit,
            "paired_id_gate": latent_fit["id_saturated"] and serial_fit["id_saturated"],
            "serial_ood_accuracy": (serial_logits.argmax(-1) == y).float().mean().item(),
            "interventions": interventions,
            "sample_count": len(y),
        })
    print(json.dumps({
        "schema_version": 1,
        "evidence_class": "causal-slot-method-smoke",
        "family": "register-state-mini-v1",
        "resource_class": "cpu",
        "settings": {
            "seeds": seeds,
            "epochs_cap": args.epochs,
            "learning_rate": args.learning_rate,
            "ood_samples": args.ood_samples,
            "recurrent_steps": 1,
            "patch_step": 0,
            "interventions": [
                "untouched", "clone-noop", "same-label", "different-label",
                "zero", "norm-matched-noise",
            ],
        },
        "records": records,
        "all_paired_id_gates_passed": all(row["paired_id_gate"] for row in records),
        "claim_boundary": (
            "Methods/instrumentation smoke only, not independent latent-causality "
            "qualification. Oracle-selected donors use labels for a controlled "
            "counterfactual evaluation, not deployment. Learned causal claims "
            "require independently preregistered OOD/controls/multi-seed review."
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
