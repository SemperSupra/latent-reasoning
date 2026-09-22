#!/usr/bin/env python3
"""WP5h learning-rate-only sweep for the frozen direct parity regime.

Selection uses ID saturation and ID training efficiency only. OOD evaluation is
explicitly disabled for every candidate learning rate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from experiments.parity_saturation import train_seed


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--learning-rates", default="0.001,0.003,0.01")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--stages", default="2,4,8")
    p.add_argument("--max-epochs-per-stage", type=int, default=1000)
    p.add_argument("--check-interval", type=int, default=5)
    p.add_argument("--stable-checks-required", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    args=p.parse_args()

    torch.set_num_threads(args.threads)
    rates=[float(x) for x in args.learning_rates.split(",") if x.strip()]
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    stages=[int(x) for x in args.stages.split(",") if x.strip()]

    candidates=[]
    started=time.perf_counter()

    for rate in rates:
        rows=[]
        for seed in seeds:
            result=train_seed(
                seed,
                stages=stages,
                max_epochs_per_stage=args.max_epochs_per_stage,
                check_interval=args.check_interval,
                stable_checks_required=args.stable_checks_required,
                batch_size=args.batch_size,
                learning_rate=rate,
                d_model=args.d_model,
                layers=args.layers,
                heads=args.heads,
                ood_min_length=stages[-1] + 1,
                ood_max_length=stages[-1] + 1,
                ood_per_length=1,
                evaluate_ood=False,
            )
            if result.ood_accuracy_by_length is not None or result.mean_ood_accuracy is not None:
                raise RuntimeError("OOD evidence leaked into learning-rate selection")
            rows.append(result)

        all_saturated=all(r.id_saturated for r in rows)
        total_epochs=[
            sum(stage.epochs_used for stage in r.stages)
            for r in rows
        ]
        final_stage_epochs=[
            r.stages[-1].epochs_used if len(r.stages) == len(stages) else args.max_epochs_per_stage
            for r in rows
        ]
        candidates.append({
            "learning_rate": rate,
            "all_seeds_id_saturated": all_saturated,
            "saturated_seed_count": sum(r.id_saturated for r in rows),
            "mean_final_id_accuracy": statistics.mean(r.final_id_accuracy for r in rows),
            "mean_total_epochs": statistics.mean(total_epochs),
            "mean_final_stage_epochs": statistics.mean(final_stage_epochs),
            "mean_wall_seconds": statistics.mean(r.total_wall_seconds for r in rows),
            "seeds": [
                {
                    "seed": r.seed,
                    "id_saturated": r.id_saturated,
                    "final_id_accuracy": r.final_id_accuracy,
                    "stage_epochs": [stage.epochs_used for stage in r.stages],
                    "ood_emitted": r.ood_accuracy_by_length is not None,
                }
                for r in rows
            ],
        })

    qualified=[c for c in candidates if c["all_seeds_id_saturated"]]
    selected=None
    if qualified:
        selected=min(
            qualified,
            key=lambda c: (
                c["mean_final_stage_epochs"],
                c["mean_total_epochs"],
                c["learning_rate"],
            ),
        )["learning_rate"]

    payload={
        "schema_version":1,
        "evidence_class":"benchmark-regime-id-training-dynamics-sweep",
        "family":"parity",
        "treatment":"direct",
        "controlled_variable":"learning_rate",
        "ood_evaluation_enabled":False,
        "frozen":{
            "architecture":{"d_model":args.d_model,"layers":args.layers,"heads":args.heads},
            "optimizer":"AdamW",
            "weight_decay":0.0,
            "stages":stages,
            "max_epochs_per_stage":args.max_epochs_per_stage,
            "check_interval":args.check_interval,
            "stable_checks_required":args.stable_checks_required,
            "batch_size":args.batch_size,
            "seeds":seeds,
        },
        "candidates":candidates,
        "selected_learning_rate":selected,
        "decision":(
            "freeze-learning-rate-and-rerun-ood"
            if selected is not None
            else "no-learning-rate-qualified"
        ),
        "wall_seconds":time.perf_counter()-started,
        "claim_boundary":(
            "Candidate selection uses complete-ID saturation and ID training efficiency only. "
            "No OOD examples are generated or evaluated in this sweep."
        ),
    }
    print(json.dumps(payload,sort_keys=True))


if __name__=="__main__":
    main()
