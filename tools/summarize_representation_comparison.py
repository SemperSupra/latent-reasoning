#!/usr/bin/env python3
"""Summarize a fixed-K representation-comparison receipt without overclaiming."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def classify(values: list[float]) -> str:
    if not values:
        return "unavailable"
    if all(value > 0 for value in values):
        return "all-positive"
    if all(value < 0 for value in values):
        return "all-negative"
    if all(value == 0 for value in values):
        return "all-zero"
    return "mixed"


def summarize(receipt: dict) -> dict:
    if receipt.get("evidence_class") != "fixed-k-reasoning-representation-comparison":
        raise ValueError("not a fixed-k representation-comparison receipt")

    aggregate = receipt["aggregate"]
    for treatment in ("direct", "serial-control", "latent"):
        if treatment not in aggregate:
            raise ValueError(f"missing treatment {treatment}")

    serial_calls = aggregate["serial-control"]["forward_calls_per_batch"]
    latent_calls = aggregate["latent"]["forward_calls_per_batch"]
    compute_match_valid = serial_calls == latent_calls

    results = receipt.get("results", [])
    by = {}
    for row in results:
        by[(row["treatment"], row["seed"])] = row

    paired_seeds = sorted(
        set(seed for treatment, seed in by if treatment == "latent")
        & set(seed for treatment, seed in by if treatment == "serial-control")
        & set(seed for treatment, seed in by if treatment == "direct")
    )

    comparisons = {}
    for reference in ("serial-control", "direct"):
        deltas = []
        by_seed = {}
        for seed in paired_seeds:
            latent = by[("latent", seed)]
            base = by[(reference, seed)]
            l = latent.get("mean_ood_accuracy")
            b = base.get("mean_ood_accuracy")
            if l is None or b is None:
                continue
            delta = l - b
            deltas.append(delta)
            by_seed[str(seed)] = delta

        comparisons[f"latent_minus_{reference}"] = {
            "paired_seed_count": len(deltas),
            "by_seed": by_seed,
            "mean": statistics.mean(deltas) if deltas else None,
            "median": statistics.median(deltas) if deltas else None,
            "pstdev": statistics.pstdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None),
            "direction": classify(deltas),
        }

    all_required_saturated = all(
        aggregate[treatment]["all_seeds_id_saturated"]
        for treatment in ("serial-control", "latent")
    )

    valid = bool(receipt.get("comparison_valid")) and compute_match_valid and all_required_saturated

    seed_count = comparisons["latent_minus_serial-control"]["paired_seed_count"]
    inference_boundary = (
        "descriptive-only"
        if seed_count < 5
        else "multi-seed-descriptive; inferential test requires a predeclared analysis plan"
    )

    return {
        "schema_version": 1,
        "source_family": receipt.get("family"),
        "recurrent_steps": receipt.get("settings", {}).get("recurrent_steps"),
        "comparison_valid": valid,
        "compute_match_valid": compute_match_valid,
        "serial_forward_calls": serial_calls,
        "latent_forward_calls": latent_calls,
        "treatment_aggregate": aggregate,
        "paired": comparisons,
        "inference_boundary": inference_boundary,
        "claim_boundary": (
            "This summary is descriptive. A directional fixed-K result on one task "
            "does not establish a general latent-reasoning advantage. Results are "
            "interpretable only when the underlying receipt passes ID-saturation and "
            "compute-match gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    result = summarize(receipt)

    if not result["comparison_valid"]:
        raise SystemExit(json.dumps(result, sort_keys=True))

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
