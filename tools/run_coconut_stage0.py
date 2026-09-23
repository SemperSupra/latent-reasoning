#!/usr/bin/env python3
"""Inspect or execute the immutable Coconut GSM CoT stage-0 capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml


COCONUT_REVISION="27273cb8cca4bb763c041a63b036d0c3b7cbbb48"
GPT2_REVISION="607a30d783dfa663caf39e06633721c8d4cfcd7e"
GSM_SOURCE_REVISION="e06a32ee5e4cd117171daeb4755d2a97ece62761"
GSM_TRAIN_SHA256="0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c"

ROOT=Path("/workspace")
COCONUT_ROOT=Path("/opt/coconut")
MODEL_ROOT=Path("/opt/models/gpt2")
CONFIG=ROOT/"configs"/"coconut-gsm-cot-stage0-2gpu.yaml"
PLAN=ROOT/"plans"/"coconut-gsm-cot-stage0-2gpu.json"
DATA_RECEIPT=Path("/opt/data-receipts/gsm-processed.json")
LFS_RECEIPT=Path("/opt/data-receipts/gsm-lfs-hydration.json")
OUTPUT_ROOT=Path("/out")


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"required identity file missing: {path}")
    return path.read_text().strip()


def inspect_capsule() -> dict:
    coconut_revision=read_text(COCONUT_ROOT/"SOURCE_REVISION")
    model_revision=read_text(MODEL_ROOT/"SOURCE_REVISION")
    if coconut_revision != COCONUT_REVISION:
        raise RuntimeError("Coconut source revision mismatch")
    if model_revision != GPT2_REVISION:
        raise RuntimeError("GPT-2 source revision mismatch")

    plan=json.loads(PLAN.read_text())
    config=yaml.safe_load(CONFIG.read_text())
    data=json.loads(DATA_RECEIPT.read_text())
    lfs=json.loads(LFS_RECEIPT.read_text())

    if plan["upstream"]["revision"] != COCONUT_REVISION:
        raise RuntimeError("plan/upstream Coconut revision mismatch")
    if plan["immutable_inputs"]["backbone_revision"] != GPT2_REVISION:
        raise RuntimeError("plan/backbone revision mismatch")
    if plan["immutable_inputs"]["gsm_source_revision"] != GSM_SOURCE_REVISION:
        raise RuntimeError("plan/GSM source revision mismatch")
    if plan["immutable_inputs"]["gsm_train_sha256"] != GSM_TRAIN_SHA256:
        raise RuntimeError("plan/GSM train identity mismatch")

    if lfs["revision"] != GSM_SOURCE_REVISION:
        raise RuntimeError("LFS receipt source revision mismatch")
    if lfs["lfs"]["hydrated_sha256"] != GSM_TRAIN_SHA256:
        raise RuntimeError("LFS receipt train digest mismatch")

    expected_processed={
        "train":"93a266b1f7425a00463bb775de2946f9b30f76514944b5456bbffceb50b5de30",
        "valid":"1833a3479186e4e6c0183aad32fb7ee343479c958f2a1b57631ec22f5ee3aa7f",
        "test":"46ce5b0c44fde0bfc8cf1e7492aa640b96111d9dcb80a54340d9911141a9caa9",
    }
    for split,expected in expected_processed.items():
        path=COCONUT_ROOT/"data"/f"gsm_{split}.json"
        if sha256_file(path) != expected:
            raise RuntimeError(f"processed {split} digest mismatch")
        if data["processed"][split]["sha256"] != expected:
            raise RuntimeError(f"processed receipt mismatch for {split}")

    target=plan["target_execution"]
    global_batch=(
        target["world_size"]
        * config["batch_size_training"]
        * config["gradient_accumulation_steps"]
    )
    if global_batch != 128:
        raise RuntimeError(f"effective global batch mismatch: {global_batch}")
    if target["world_size"] != 2:
        raise RuntimeError("stage-0 capsule expects exactly two ranks")
    if config["model_id"] != "/opt/models/gpt2":
        raise RuntimeError("stage-0 config is not offline-model pinned")
    if config["train_path"] != "/opt/coconut/data/gsm_train.json":
        raise RuntimeError("stage-0 train path mismatch")
    if config["val_path"] != "/opt/coconut/data/gsm_valid.json":
        raise RuntimeError("stage-0 validation path mismatch")

    offline={
        "HF_HUB_OFFLINE":os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE":os.environ.get("TRANSFORMERS_OFFLINE"),
        "WANDB_MODE":os.environ.get("WANDB_MODE"),
    }
    if offline != {
        "HF_HUB_OFFLINE":"1",
        "TRANSFORMERS_OFFLINE":"1",
        "WANDB_MODE":"offline",
    }:
        raise RuntimeError(f"offline environment mismatch: {offline}")

    return {
        "schema_version":1,
        "evidence_class":"coconut-stage0-capsule-inspection",
        "container_contract":os.environ.get("LATENT_REASONING_CONTAINER"),
        "repository_source_revision":os.environ.get("LATENT_REASONING_SOURCE_REVISION"),
        "coconut_revision":coconut_revision,
        "gpt2_revision":model_revision,
        "gsm_source_revision":lfs["revision"],
        "gsm_train_sha256":lfs["lfs"]["hydrated_sha256"],
        "processed_data_sha256":expected_processed,
        "derived_config_sha256":sha256_file(CONFIG),
        "plan_sha256":sha256_file(PLAN),
        "effective_global_batch":global_batch,
        "world_size":target["world_size"],
        "per_rank_batch_size":config["batch_size_training"],
        "gradient_accumulation_steps":config["gradient_accumulation_steps"],
        "num_epochs":config["num_epochs"],
        "learning_rate":config["lr"],
        "offline_environment":offline,
        "torch_version":torch.__version__,
        "torch_cuda_version":torch.version.cuda,
        "python_version":platform.python_version(),
        "cuda_visible":torch.cuda.is_available(),
        "cuda_device_count":torch.cuda.device_count(),
        "claim_boundary":(
            "This receipt establishes immutable capsule/input/config identity. "
            "It does not establish that stage-0 training was executed."
        ),
    }


def checkpoint_receipts() -> list[dict]:
    checkpoint_root=OUTPUT_ROOT/"checkpoints"/"gsm-cot-stage0-2gpu"
    if not checkpoint_root.is_dir():
        return []
    receipts=[]
    for path in sorted(checkpoint_root.glob("checkpoint_*")):
        if path.is_file():
            receipts.append({
                "name":path.name,
                "size_bytes":path.stat().st_size,
                "sha256":sha256_file(path),
            })
    return receipts


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+"\n")


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--inspect-only",action="store_true")
    p.add_argument(
        "--receipt",
        type=Path,
        default=OUTPUT_ROOT/"stage0-capsule-inspection.json",
    )
    args=p.parse_args()

    inspection=inspect_capsule()
    write_json(args.receipt,inspection)

    if args.inspect_only:
        print(json.dumps(inspection,sort_keys=True))
        return

    if torch.cuda.device_count() < 2:
        raise SystemExit("stage-0 apply requires at least two visible CUDA devices")

    launcher=[
        "torchrun",
        "--nproc_per_node=2",
        str(COCONUT_ROOT/"run.py"),
        str(CONFIG),
    ]

    started=time.time()
    perf=time.perf_counter()
    completed=subprocess.run(
        launcher,
        cwd=COCONUT_ROOT,
        env=os.environ.copy(),
        check=False,
    )
    elapsed=time.perf_counter()-perf

    checkpoints=checkpoint_receipts()
    execution={
        "schema_version":1,
        "evidence_class":"coconut-stage0-training-execution",
        "container_contract":inspection["container_contract"],
        "repository_source_revision":inspection["repository_source_revision"],
        "launcher_argv":launcher,
        "launcher_uses_shell":False,
        "exit_code":completed.returncode,
        "started_unix":started,
        "elapsed_seconds":elapsed,
        "cuda_device_count":torch.cuda.device_count(),
        "checkpoints":checkpoints,
        "inspection_receipt_sha256":sha256_file(args.receipt),
    }
    write_json(OUTPUT_ROOT/"stage0-execution.json",execution)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if not checkpoints:
        raise RuntimeError("stage-0 training completed without any checkpoint artifact")

    print(json.dumps(execution,sort_keys=True))


if __name__=="__main__":
    main()
