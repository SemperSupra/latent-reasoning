#!/usr/bin/env python3
"""ID-only common learning-rate sweep for K=2 recurrent parity treatments.

OOD is deliberately disabled. A learning rate is eligible only if BOTH
serial-control and latent saturate the complete ID universe for all seeds.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict

import torch

from experiments.parity_representation_comparison import train_one


TREATMENTS = ("serial-control", "latent")


def choose_candidate(candidates: list[dict]) -> float | None:
    qualified = [candidate for candidate in candidates if candidate["qualified"]]
    if not qualified:
        return None
    # Predeclared rule: minimize mean total curriculum epochs across both
    # recurrent treatments and all seeds; break exact ties by lower LR.
    best = min(
        qualified,
        key=lambda candidate: (
            candidate["mean_total_epochs"],
            candidate["learning_rate"],
        ),
    )
    return best["learning_rate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rates", default="0.0005,0.001,0.002,0.003,0.005")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--stages", default="2,4,8")
    parser.add_argument("--recurrent-steps", type=int, default=2)
    parser.add_argument("--max-epochs-per-stage", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    rates = [float(value) for value in args.learning_rates.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    stages = [int(value) for value in args.stages.split(",") if value.strip()]

    candidates = []

    for learning_rate in rates:
        results = []
        for treatment in TREATMENTS:
            for seed in seeds:
                result = train_one(
                    treatment,
                    seed,
                    recurrent_steps=args.recurrent_steps,
                    stages=stages,
                    max_epochs_per_stage=args.max_epochs_per_stage,
                    check_interval=5,
                    stable_checks_required=3,
                    batch_size=args.batch_size,
                    learning_rate=learning_rate,
                    d_model=args.d_model,
                    layers=args.layers,
                    heads=args.heads,
                    ood_min_length=9,
                    ood_max_length=16,
                    ood_per_length=256,
                    evaluate_ood=False,
                )
                if result.ood_accuracy_by_length is not None or result.mean_ood_accuracy is not None:
                    raise RuntimeError("OOD leakage detected during recurrent LR selection")
                results.append(result)

        by_treatment = {
            treatment: [row for row in results if row.treatment == treatment]
            for treatment in TREATMENTS
        }
        treatment_summary = {}
        for treatment, rows in by_treatment.items():
            treatment_summary[treatment] = {
                "all_seeds_id_saturated": all(row.id_saturated for row in rows),
                "mean_final_id_accuracy": statistics.mean(row.final_id_accuracy for row in rows),
                "mean_total_epochs": statistics.mean(sum(row.stage_epochs) for row in rows),
                "seeds": [
                    {
                        "seed": row.seed,
                        "id_saturated": row.id_saturated,
                        "final_id_accuracy": row.final_id_accuracy,
                        "stage_epochs": row.stage_epochs,
                        "ood_emitted": row.mean_ood_accuracy is not None,
                    }
                    for row in rows
                ],
            }

        qualified = all(
            treatment_summary[treatment]["all_seeds_id_saturated"]
            for treatment in TREATMENTS
        )
        candidates.append({
            "learning_rate": learning_rate,
            "qualified": qualified,
            "mean_total_epochs": statistics.mean(
                sum(row.stage_epochs) for row in results
            ),
            "treatments": treatment_summary,
        })

    selected = choose_candidate(candidates)

    payload = {
        "schema_version": 1,
        "evidence_class": "id-only-recurrent-treatment-hyperparameter-selection",
        "family": "parity",
        "controlled_variable": "common_recurrent_learning_rate",
        "recurrent_steps": args.recurrent_steps,
        "treatments": list(TREATMENTS),
        "seeds": seeds,
        "stages": stages,
        "candidates": candidates,
        "selected_learning_rate": selected,
        "decision": (
            "freeze-common-recurrent-learning-rate-and-rerun-k2"
            if selected is not None
            else "expand-id-only-recurrent-optimization-search"
        ),
        "selection_rule": (
            "candidate must saturate all serial-control and latent seeds; "
            "among qualified candidates minimize mean total curriculum epochs, "
            "breaking exact ties by lower learning rate"
        ),
        "ood_accessed": False,
        "claim_boundary": (
            "This sweep selects training dynamics using ID saturation only. "
            "It contains no OOD evidence and cannot support a representation claim."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
