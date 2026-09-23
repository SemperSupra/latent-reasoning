#!/usr/bin/env python3
"""Repaired parity K=2 causal comparison.

Direct remains the admitted lower-compute contextual baseline at LR 0.001.
Serial-control and latent both use the common recurrent LR 0.002 selected by an
ID-only sweep with OOD disabled.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict

import torch

from experiments.parity_representation_comparison import train_one


TREATMENTS = ("direct", "serial-control", "latent")
DIRECT_LR = 0.001
RECURRENT_LR = 0.002


def lr_for(treatment: str) -> float:
    return DIRECT_LR if treatment == "direct" else RECURRENT_LR


def paired(results):
    by = {
        treatment: {row.seed: row for row in results if row.treatment == treatment}
        for treatment in TREATMENTS
    }
    seeds = sorted(set.intersection(*(set(rows) for rows in by.values())))
    comparisons = {}

    for reference in ("serial-control", "direct"):
        deltas = []
        by_seed = {}
        for seed in seeds:
            latent = by["latent"][seed]
            base = by[reference][seed]
            if latent.mean_ood_accuracy is None or base.mean_ood_accuracy is None:
                continue
            delta = latent.mean_ood_accuracy - base.mean_ood_accuracy
            by_seed[str(seed)] = delta
            deltas.append(delta)

        comparisons[f"latent_minus_{reference}"] = {
            "paired_seed_count": len(deltas),
            "by_seed": by_seed,
            "mean": statistics.mean(deltas) if deltas else None,
            "pstdev": statistics.pstdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None),
            "direction": (
                "all-positive" if deltas and all(x > 0 for x in deltas)
                else "all-negative" if deltas and all(x < 0 for x in deltas)
                else "all-zero" if deltas and all(x == 0 for x in deltas)
                else "mixed" if deltas
                else "unavailable"
            ),
        }
    return {"paired_seeds": seeds, "comparisons": comparisons}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--stages", default="2,4,8")
    p.add_argument("--recurrent-steps", type=int, default=2)
    p.add_argument("--max-epochs-per-stage", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ood-min-length", type=int, default=9)
    p.add_argument("--ood-max-length", type=int, default=16)
    p.add_argument("--ood-per-length", type=int, default=256)
    p.add_argument("--threads", type=int, default=2)
    args=p.parse_args()

    torch.set_num_threads(args.threads)
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    stages=[int(x) for x in args.stages.split(",") if x.strip()]

    results=[]
    for treatment in TREATMENTS:
        for seed in seeds:
            results.append(train_one(
                treatment,
                seed,
                recurrent_steps=args.recurrent_steps,
                stages=stages,
                max_epochs_per_stage=args.max_epochs_per_stage,
                check_interval=5,
                stable_checks_required=3,
                batch_size=args.batch_size,
                learning_rate=lr_for(treatment),
                d_model=64,
                layers=2,
                heads=4,
                ood_min_length=args.ood_min_length,
                ood_max_length=args.ood_max_length,
                ood_per_length=args.ood_per_length,
                evaluate_ood=True,
            ))

    aggregate={}
    for treatment in TREATMENTS:
        rows=[row for row in results if row.treatment==treatment]
        ood=[row.mean_ood_accuracy for row in rows if row.mean_ood_accuracy is not None]
        aggregate[treatment]={
            "learning_rate":lr_for(treatment),
            "all_seeds_id_saturated":all(row.id_saturated for row in rows),
            "mean_final_id_accuracy":statistics.mean(row.final_id_accuracy for row in rows),
            "mean_ood_accuracy":statistics.mean(ood) if ood else None,
            "mean_total_epochs":statistics.mean(sum(row.stage_epochs) for row in rows),
            "mean_training_wall_seconds":statistics.mean(row.training_wall_seconds for row in rows),
            "forward_calls_per_batch":rows[0].forward_calls_per_batch,
            "parameter_count":rows[0].parameter_count,
        }

    compute_match=(
        aggregate["serial-control"]["forward_calls_per_batch"]
        == aggregate["latent"]["forward_calls_per_batch"]
        == args.recurrent_steps + 1
    )
    comparison_valid=(
        compute_match
        and aggregate["serial-control"]["all_seeds_id_saturated"]
        and aggregate["latent"]["all_seeds_id_saturated"]
    )

    payload={
        "schema_version":1,
        "evidence_class":"fixed-k-reasoning-representation-comparison",
        "family":"parity",
        "comparison_revision":"repaired-common-recurrent-lr-v1",
        "settings":{
            "seeds":seeds,
            "stages":stages,
            "recurrent_steps":args.recurrent_steps,
            "direct_learning_rate":DIRECT_LR,
            "recurrent_learning_rate":RECURRENT_LR,
            "recurrent_lr_selection":{
                "source":"ID-only common recurrent LR sweep",
                "ood_accessed":False,
                "selection_rule":"joint serial+latent ID saturation then minimum mean total curriculum epochs",
            },
            "ood_min_length":args.ood_min_length,
            "ood_max_length":args.ood_max_length,
            "ood_per_length":args.ood_per_length,
        },
        "results":[asdict(row) for row in results],
        "aggregate":aggregate,
        "paired_analysis":paired(results),
        "compute_match_valid":compute_match,
        "comparison_valid":comparison_valid,
        "claim_boundary":(
            "The causal comparison is latent versus serial-control; both recurrent "
            "treatments share K, compute, architecture, initialization, data, and "
            "the ID-only-selected LR 0.002. Direct is contextual and remains at its "
            "separately frozen LR 0.001. Three-seed evidence is descriptive."
        ),
    }
    print(json.dumps(payload,sort_keys=True))


if __name__=="__main__":
    main()
