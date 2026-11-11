#!/usr/bin/env python3
"""Finite, preregistered nightly CPU register-state cohort replay.

Only public-synthetic cases. This orchestrator does NOT generate new methods,
tune on OOD, submit to Kaggle, call agents, invoke shells or write to GitHub.
It allocates fixed K and matched seed pairs from a frozen 24-night ledger.
"""
from __future__ import annotations
import argparse
from datetime import date
import hashlib
import json
import math
import statistics
from pathlib import Path

START = date(2026, 11, 12)
DEPTHS = (1, 2, 4)
COHORTS = 8
END_EXCLUSIVE = START.toordinal() + COHORTS * len(DEPTHS)
SCHEMA = "latent-cpu-overnight-assessment/v1"
ENVIRONMENT = "environments/python312-torch251-cpu.json"
MODULE = "experiments.register_state_k_sweep"

def canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def digest(obj: dict) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()

def planned(day: date) -> dict | None:
    offset = day.toordinal() - START.toordinal()
    if offset < 0 or offset >= COHORTS * len(DEPTHS):
        return None
    cohort, depth_index = divmod(offset, len(DEPTHS))
    k = DEPTHS[depth_index]
    s0 = 8 + 2 * cohort
    return {
        "schema_version": 1,
        "id": f"campaign-0001/overnight-register-state-c{cohort:02d}-k{k}-v1",
        "resource_class": "cpu",
        "environment_ref": ENVIRONMENT,
        "module": MODULE,
        "arguments": {
            "ks": str(k),
            "seeds": f"{s0},{s0+1}",
            "stages": "1,2",
            "max_epochs_per_stage": 150,
            "learning_rate": 0.001,
            "batch_size": 256,
            "ood_min_steps": 3,
            "ood_max_steps": 4,
            "ood_per_steps": 128,
            "threads": 2,
        },
        "timeout_seconds": 1080,
    }

def verify(spec: dict, science: dict, metadata: dict) -> dict:
    if (spec.get("module") != MODULE or spec.get("resource_class") != "cpu"
        or spec.get("environment_ref") != ENVIRONMENT):
        raise ValueError("not a registered overnight experiment")
    expected = False
    for offset in range(COHORTS * len(DEPTHS)):
        d = date.fromordinal(START.toordinal() + offset)
        if planned(d) == spec:
            expected = True
            break
    if not expected:
        raise ValueError("unregistered or modified overnight spec")
    if (metadata.get("replay_spec_id") != spec["id"]
        or metadata.get("replay_spec_sha256") != digest(spec)
        or metadata.get("scientific_receipt_sha256") != digest(science)
        or metadata.get("resource_class") != "cpu"):
        raise ValueError("replay provenance mismatch")
    a = spec["arguments"]
    k = int(a["ks"])
    seeds = [int(x) for x in a["seeds"].split(",")]
    if (science.get("schema_version") != 1
        or science.get("evidence_class") != "recurrent-depth-sweep"
        or science.get("family") != "register-state-mini-v1"
        or science.get("treatments") != ["serial-control", "latent"]):
        raise ValueError("unknown scientific receipt")
    settings = science.get("settings", {})
    if (settings.get("ks") != [k] or settings.get("seeds") != seeds
        or settings.get("stages") != [1, 2]):
        raise ValueError("scientific treatment mismatch")
    sweep = science.get("sweep")
    if not isinstance(sweep, list) or len(sweep) != 1 or sweep[0].get("k") != k:
        raise ValueError("missing single-K sweep")
    row = sweep[0]
    results = row.get("results")
    if not isinstance(results, list) or len(results) != 2 * len(seeds):
        raise ValueError("unexpected paired result count")
    expected_pairs = {(t, s) for t in ("serial-control", "latent") for s in seeds}
    actual_pairs = {(r["treatment"], r["seed"]) for r in results}
    if actual_pairs != expected_pairs:
        raise ValueError("duplicate, missing or extra treatment pair")
    per_seed = {}
    for seed in seeds:
        rs = [r for r in results if r["seed"] == seed]
        for r in rs:
            if r["recurrent_steps"] != k or r["forward_calls_per_batch"] != k + 1:
                raise ValueError("forward-call control failed")
            saturated = (r.get("id_saturated") is True
                         and r.get("final_id_accuracy") == 1.0)
            if not saturated and r.get("mean_ood_accuracy") is not None:
                raise ValueError("OOD leaked for unsaturated ID case")
        per_seed[str(seed)] = {
            "paired_id_saturated": all(r["id_saturated"] is True for r in rs),
            "final_id_accuracy": {r["treatment"]: r["final_id_accuracy"] for r in rs}
        }
    # Recompute treatment differences from independent per-seed measurements.
    # A self-reported aggregate cannot be its own acceptance oracle.
    deltas = {}
    for seed in seeds:
        rs = {r["treatment"]: r for r in results if r["seed"] == seed}
        left = rs["latent"].get("mean_ood_accuracy")
        right = rs["serial-control"].get("mean_ood_accuracy")
        if left is not None and right is not None:
            deltas[str(seed)] = left - right
    observed_pair = row.get("paired_latent_minus_serial", {})
    actual_map = observed_pair.get("by_seed", {})
    if (not isinstance(actual_map, dict)
        or set(actual_map) != set(deltas)
        or observed_pair.get("paired_seed_count") != len(deltas)):
        raise ValueError("independent paired OOD grid mismatch")
    for label, value in deltas.items():
        if not isinstance(actual_map[label], (int, float)) or not math.isclose(
            actual_map[label], value, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError("independent paired OOD delta mismatch")
    observed_mean = observed_pair.get("mean")
    recomputed_mean = statistics.mean(deltas.values()) if deltas else None
    if (observed_mean is None) != (recomputed_mean is None):
        raise ValueError("independent paired OOD mean presence mismatch")
    if recomputed_mean is not None and (
        not isinstance(observed_mean, (float, int)) or not math.isclose(
            observed_mean, recomputed_mean, rel_tol=1e-10, abs_tol=1e-10
        )
    ):
        raise ValueError("independent paired OOD mean mismatch")
    paired_valid = all(v["paired_id_saturated"] for v in per_seed.values())
    if bool(row.get("comparison_valid")) != paired_valid:
        raise ValueError("comparison validity gate inconsistent")
    if (science.get("valid_ks") != ([k] if paired_valid else [])):
        raise ValueError("valid K register inconsistent")
    report = {
        "schema": SCHEMA, "replay_spec_id": spec["id"],
        "replay_spec_sha256": digest(spec),
        "scientific_receipt_sha256": digest(science),
        "source_revision": metadata.get("source_revision"),
        "method_status": "VALID_RECEIPT",
        "claim_status": "DESCRIPTIVE_PAIRED_ONLY" if paired_valid else "HOLD_ID_UNSATURATED",
        "k": k, "seeds": seeds, "paired_seed_gates": per_seed,
        "paired_ood_mean": recomputed_mean if paired_valid else None,
        "claim_boundary": (
            "Public synthetic register-state stages 1-2. Matched K+1 forward "
            "calls do not establish equal FLOPs; short curriculum is not full "
            "stage-3 qualification. OOD is not valid when ID has not saturated. "
            "No result can authorize model promotion, new treatments or tuning."
        ),
    }
    return report

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--date", required=True)
    plan.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--spec", type=Path, required=True)
    check.add_argument("--scientific", type=Path, required=True)
    check.add_argument("--metadata", type=Path, required=True)
    check.add_argument("--assessment", type=Path, required=True)
    args = p.parse_args()
    if args.action == "plan":
        day = date.fromisoformat(args.date)
        spec = planned(day)
        if spec is None:
            print("CPU_OVERNIGHT_STATUS=NO_ACTIVE_CAMPAIGN_SLOT")
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
        print("CPU_OVERNIGHT_STATUS=PLANNED " + spec["id"])
        return
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    assessment = verify(read(args.spec), read(args.scientific), read(args.metadata))
    args.assessment.parent.mkdir(parents=True, exist_ok=True)
    args.assessment.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    print("CPU_OVERNIGHT_STATUS=" + assessment["claim_status"])

if __name__ == "__main__":
    main()
