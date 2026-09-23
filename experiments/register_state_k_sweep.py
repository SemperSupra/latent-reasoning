#!/usr/bin/env python3
"""Bounded recurrent-depth sweep on admitted register-state regime."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict

import torch

from experiments.register_state_representation_comparison import train_one


TREATMENTS = ("serial-control", "latent")


def paired(rows):
    by_treatment = {
        treatment: {row.seed: row for row in rows if row.treatment == treatment}
        for treatment in TREATMENTS
    }
    seeds = sorted(set(by_treatment["serial-control"]) & set(by_treatment["latent"]))
    deltas = {}
    values = []
    for seed in seeds:
        serial = by_treatment["serial-control"][seed]
        latent = by_treatment["latent"][seed]
        if serial.mean_ood_accuracy is None or latent.mean_ood_accuracy is None:
            continue
        delta = latent.mean_ood_accuracy - serial.mean_ood_accuracy
        deltas[str(seed)] = delta
        values.append(delta)

    return {
        "paired_seed_count": len(values),
        "by_seed": deltas,
        "mean": statistics.mean(values) if values else None,
        "pstdev": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None),
        "direction": (
            "all-positive" if values and all(v > 0 for v in values)
            else "all-negative" if values and all(v < 0 for v in values)
            else "all-zero" if values and all(v == 0 for v in values)
            else "mixed" if values
            else "unavailable"
        ),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ks", default="1,2,4,8")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--stages", default="1,2,3")
    p.add_argument("--max-epochs-per-stage", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ood-min-steps", type=int, default=4)
    p.add_argument("--ood-max-steps", type=int, default=6)
    p.add_argument("--ood-per-steps", type=int, default=512)
    p.add_argument("--threads", type=int, default=2)
    args=p.parse_args()

    torch.set_num_threads(args.threads)
    ks=[int(x) for x in args.ks.split(",") if x.strip()]
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    stages=[int(x) for x in args.stages.split(",") if x.strip()]

    sweep=[]
    for k in ks:
        rows=[]
        for treatment in TREATMENTS:
            for seed in seeds:
                rows.append(train_one(
                    treatment,
                    seed,
                    recurrent_steps=k,
                    stages=stages,
                    max_epochs_per_stage=args.max_epochs_per_stage,
                    check_interval=5,
                    stable_checks_required=3,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    d_model=64,
                    layers=2,
                    heads=4,
                    ood_min_steps=args.ood_min_steps,
                    ood_max_steps=args.ood_max_steps,
                    ood_per_steps=args.ood_per_steps,
                ))

        aggregate={}
        for treatment in TREATMENTS:
            tr=[row for row in rows if row.treatment==treatment]
            ood=[row.mean_ood_accuracy for row in tr if row.mean_ood_accuracy is not None]
            aggregate[treatment]={
                "all_seeds_id_saturated": all(row.id_saturated for row in tr),
                "mean_final_id_accuracy": statistics.mean(row.final_id_accuracy for row in tr),
                "mean_ood_accuracy": statistics.mean(ood) if ood else None,
                "mean_total_epochs": statistics.mean(sum(row.stage_epochs) for row in tr),
                "mean_training_wall_seconds": statistics.mean(row.training_wall_seconds for row in tr),
                "forward_calls_per_batch": tr[0].forward_calls_per_batch,
            }

        if aggregate["serial-control"]["forward_calls_per_batch"] != k+1:
            raise RuntimeError(f"serial K={k} forward-call mismatch")
        if aggregate["latent"]["forward_calls_per_batch"] != k+1:
            raise RuntimeError(f"latent K={k} forward-call mismatch")

        valid=(
            aggregate["serial-control"]["all_seeds_id_saturated"]
            and aggregate["latent"]["all_seeds_id_saturated"]
        )
        sweep.append({
            "k":k,
            "comparison_valid":valid,
            "aggregate":aggregate,
            "paired_latent_minus_serial":paired(rows),
            "results":[asdict(row) for row in rows],
        })

    payload={
        "schema_version":1,
        "evidence_class":"recurrent-depth-sweep",
        "family":"register-state-mini-v1",
        "treatments":list(TREATMENTS),
        "settings":{
            "ks":ks,
            "seeds":seeds,
            "stages":stages,
            "learning_rate":args.learning_rate,
            "d_model":64,
            "layers":2,
            "heads":4,
            "ood_min_steps":args.ood_min_steps,
            "ood_max_steps":args.ood_max_steps,
            "ood_per_steps":args.ood_per_steps,
        },
        "sweep":sweep,
        "valid_ks":[row["k"] for row in sweep if row["comparison_valid"]],
        "claim_boundary":(
            "Each K is independently admissible only when serial-control and latent "
            "both saturate all ID seeds. OOD is suppressed per failed treatment/seed. "
            "This task-specific depth sweep remains descriptive with three seeds."
        ),
    }
    print(json.dumps(payload,sort_keys=True))


if __name__=="__main__":
    main()
