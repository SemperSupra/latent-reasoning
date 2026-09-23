#!/usr/bin/env python3
"""Validate the resource-adapted Coconut GSM CoT stage-0 launch plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ALLOWED_OVERRIDE_KEYS={
    "save_path",
    "name",
    "model_id",
    "train_path",
    "val_path",
    "gradient_accumulation_steps",
}


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def effective_global_batch(world_size: int, batch_size: int, accumulation: int) -> int:
    return world_size * batch_size * accumulation


def load_yaml(path: Path) -> dict:
    value=yaml.safe_load(path.read_text())
    if not isinstance(value,dict):
        raise ValueError(f"YAML document must be an object: {path}")
    return value


def validate_plan(*, plan: dict, upstream_config: dict, derived_config: dict) -> dict:
    overrides=plan["allowed_config_overrides"]
    if set(overrides) != ALLOWED_OVERRIDE_KEYS:
        raise ValueError(
            f"allowed override set changed: {sorted(overrides)} != "
            f"{sorted(ALLOWED_OVERRIDE_KEYS)}"
        )

    if set(upstream_config) != set(derived_config):
        missing=set(upstream_config)-set(derived_config)
        extra=set(derived_config)-set(upstream_config)
        raise ValueError(f"config key set changed: missing={sorted(missing)} extra={sorted(extra)}")

    diffs={}
    for key in upstream_config:
        if upstream_config[key] != derived_config[key]:
            diffs[key]={
                "upstream":upstream_config[key],
                "derived":derived_config[key],
            }

    if set(diffs) != ALLOWED_OVERRIDE_KEYS:
        raise ValueError(
            f"actual config differences are not exactly the reviewed override set: "
            f"{sorted(diffs)}"
        )

    for key,expected in overrides.items():
        if derived_config[key] != expected:
            raise ValueError(
                f"derived override mismatch for {key}: "
                f"{derived_config[key]!r} != {expected!r}"
            )

    # Scientific stage-0 identity checks.
    required_reference={
        "coconut":False,
        "cot":True,
        "no_thoughts":False,
        "no_cot":False,
        "c_thought":0,
        "epochs_per_stage":1,
        "max_latent_stage":0,
        "pad_latent_to_max":True,
        "save_only_improve":True,
        "uniform_prob":0.0,
        "seed":0,
        "resume":0,
        "bf16":False,
        "reset_optimizer":False,
        "batch_size_training":32,
        "debug":False,
        "num_epochs":25,
        "lr":1e-4,
        "weight_decay":0.01,
    }
    for key,expected in required_reference.items():
        if upstream_config.get(key) != expected:
            raise ValueError(
                f"pinned upstream reference changed for {key}: "
                f"{upstream_config.get(key)!r} != {expected!r}"
            )
        if key not in ALLOWED_OVERRIDE_KEYS and derived_config.get(key) != expected:
            raise ValueError(f"derived config changed scientific field {key}")

    ref=plan["reference_execution"]
    target=plan["target_execution"]

    ref_batch=effective_global_batch(
        ref["world_size"],
        upstream_config["batch_size_training"],
        upstream_config["gradient_accumulation_steps"],
    )
    target_batch=effective_global_batch(
        target["world_size"],
        derived_config["batch_size_training"],
        derived_config["gradient_accumulation_steps"],
    )

    if ref_batch != ref["effective_global_batch"]:
        raise ValueError("reference effective global batch receipt is inconsistent")
    if target_batch != target["effective_global_batch"]:
        raise ValueError("target effective global batch receipt is inconsistent")
    if ref_batch != target_batch:
        raise ValueError(f"global batch changed: {ref_batch} != {target_batch}")

    if target["per_rank_batch_size"] != derived_config["batch_size_training"]:
        raise ValueError("target per-rank batch does not match derived config")
    if target["gradient_accumulation_steps"] != derived_config["gradient_accumulation_steps"]:
        raise ValueError("target accumulation does not match derived config")
    if target["launcher"] != "torchrun":
        raise ValueError("stage-0 plan must use torchrun")
    if target["wandb_mode"] != "offline":
        raise ValueError("stage-0 capsule must keep external W&B optional/offline")

    launcher=[
        "torchrun",
        f"--nproc_per_node={target['world_size']}",
        "/opt/coconut/run.py",
        "/workspace/configs/coconut-gsm-cot-stage0-2gpu.yaml",
    ]

    return {
        "schema_version":1,
        "evidence_class":"coconut-stage0-launch-plan",
        "plan_id":plan["id"],
        "reference_effective_global_batch":ref_batch,
        "target_effective_global_batch":target_batch,
        "allowed_config_differences":diffs,
        "launcher_argv":launcher,
        "launcher_uses_shell":False,
        "environment":{"WANDB_MODE":"offline"},
        "immutable_inputs":plan["immutable_inputs"],
        "claim_boundary":(
            "This validates reference-config identity, explicit two-rank resource "
            "adaptation, global-batch preservation, and launch construction. It "
            "does not execute stage-0 training or establish training efficacy."
        ),
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--upstream-root",type=Path,required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--derived-config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()

    plan=json.loads(args.plan.read_text())
    upstream_revision=plan["upstream"]["revision"]

    head=subprocess.run(
        ["git","-C",str(args.upstream_root),"rev-parse","HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if head != upstream_revision:
        raise SystemExit(f"upstream checkout mismatch: {head} != {upstream_revision}")

    upstream_path=args.upstream_root/plan["upstream"]["config_path"]
    upstream_config=load_yaml(upstream_path)
    derived_config=load_yaml(args.derived_config)

    receipt=validate_plan(
        plan=plan,
        upstream_config=upstream_config,
        derived_config=derived_config,
    )
    receipt["upstream_config_sha256"]=sha256_file(upstream_path)
    receipt["derived_config_sha256"]=sha256_file(args.derived_config)
    receipt["upstream_revision"]=head

    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,sort_keys=True)+"\n")
    print(json.dumps(receipt,sort_keys=True))


if __name__=="__main__":
    main()
