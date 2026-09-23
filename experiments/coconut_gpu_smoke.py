#!/usr/bin/env python3
"""Bounded GPU smoke for exact upstream Coconut + pinned reference GPT-2.

This measures mechanism/resource behavior for one train step. It is not a
GSM8K or paper reproduction and does not evaluate efficacy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPECTED_COCONUT_REVISION = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"
EXPECTED_GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DEFAULT_MODEL_PATH = "/opt/models/gpt2"
DEFAULT_COCONUT_PATH = "/opt/coconut"


def load_coconut_class(root: Path):
    module_path = root / "coconut.py"
    spec = importlib.util.spec_from_file_location("pinned_coconut", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import Coconut from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Coconut


def require_identity() -> None:
    coconut_revision = os.environ.get("COCONUT_REVISION")
    gpt2_revision = os.environ.get("GPT2_REVISION")
    if coconut_revision != EXPECTED_COCONUT_REVISION:
        raise RuntimeError(
            f"Coconut revision mismatch: {coconut_revision!r} != "
            f"{EXPECTED_COCONUT_REVISION}"
        )
    if gpt2_revision != EXPECTED_GPT2_REVISION:
        raise RuntimeError(
            f"GPT-2 revision mismatch: {gpt2_revision!r} != "
            f"{EXPECTED_GPT2_REVISION}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--latent-slots", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--coconut-path", default=DEFAULT_COCONUT_PATH)
    args = parser.parse_args()

    if args.latent_slots < 1 or args.latent_slots > 16:
        raise ValueError("latent-slots must be between 1 and 16")

    require_identity()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; GPU smoke must not fall back to CPU")

    device = torch.device(f"cuda:{args.device_index}")
    props = torch.cuda.get_device_properties(device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)

    model_path = Path(args.model_path)
    coconut_path = Path(args.coconut_path)
    if not model_path.is_dir():
        raise RuntimeError(f"pinned GPT-2 snapshot missing: {model_path}")
    if not (coconut_path / "coconut.py").is_file():
        raise RuntimeError(f"pinned Coconut source missing: {coconut_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    base.resize_token_embeddings(len(tokenizer))

    embedding = base.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    with torch.no_grad():
        for token_id in (latent_id, start_id, end_id):
            embedding.weight[token_id].copy_(embedding.weight[target_id])
            base.lm_head.weight[token_id].copy_(base.lm_head.weight[target_id])

    Coconut = load_coconut_class(coconut_path)
    model = Coconut(
        base,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
    ).to(device)

    prefix = tokenizer.encode("Question: 2 + 3 =")
    suffix = tokenizer.encode(" Answer: 5") + [tokenizer.eos_token_id]
    tokens = (
        prefix
        + [start_id]
        + [latent_id] * args.latent_slots
        + [end_id]
        + suffix
    )

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(
        input_ids.shape[1],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    tracked = base.transformer.h[0].attn.c_attn.weight
    before = tracked.detach().clone()

    torch.cuda.synchronize(device)
    started = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
    )

    if not torch.isfinite(outputs.loss):
        raise RuntimeError("non-finite GPU smoke loss")

    outputs.loss.backward()
    grad = tracked.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0.0:
        raise RuntimeError("gradient did not propagate through Coconut recurrence")

    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    changed = not torch.equal(before, tracked.detach())
    if not changed:
        raise RuntimeError("optimizer step did not update tracked GPT-2 parameter")

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    payload = {
        "schema_version": 1,
        "evidence_class": "pinned-upstream-reference-backbone-gpu-smoke",
        "upstream": {
            "repository": "facebookresearch/coconut",
            "revision": EXPECTED_COCONUT_REVISION,
            "module": "coconut.py"
        },
        "backbone": {
            "id": "openai-community/gpt2",
            "revision": EXPECTED_GPT2_REVISION,
            "local_snapshot": str(model_path)
        },
        "container_contract": os.environ.get("LATENT_REASONING_CONTAINER"),
        "device": {
            "index": args.device_index,
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "driver_version": os.environ.get("NVIDIA_DRIVER_VERSION"),
            "torch_cuda_version": torch.version.cuda
        },
        "latent_slots": args.latent_slots,
        "input_tokens": int(input_ids.shape[1]),
        "parameter_count": sum(p.numel() for p in base.parameters()),
        "learning_rate": args.learning_rate,
        "loss": float(outputs.loss.detach()),
        "loss_finite": True,
        "gradient_through_recurrence": True,
        "optimizer_step_changed_parameter": changed,
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "cuda_free_bytes_after": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
        "claim_boundary": (
            "This is one bounded GPU train-step/resource smoke using exact pinned "
            "Coconut source and GPT-2 weights. It is not a paper reproduction, "
            "benchmark result, or efficacy claim."
        )
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
