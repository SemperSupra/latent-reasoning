#!/usr/bin/env python3
"""Smoke the exact pinned upstream Coconut core class on CPU.

The workflow clones facebookresearch/coconut at the pinned commit and exposes it
through COCONUT_UPSTREAM. No upstream source is vendored into this repository.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel


PINNED_REVISION = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"


def load_upstream_coconut():
    root = Path(os.environ["COCONUT_UPSTREAM"])
    module_path = root / "coconut.py"
    spec = importlib.util.spec_from_file_location("pinned_coconut", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load pinned Coconut module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Coconut


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)

    Coconut = load_upstream_coconut()

    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=True,
    )
    base = GPT2LMHeadModel(config)

    latent_id = 60
    start_id = 61
    end_id = 62

    model = Coconut(
        base,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=2,
    )

    # Two latent slots force two recurrent passes before the final pass.
    input_ids = torch.tensor([[1, 7, start_id, latent_id, latent_id, end_id, 11, 2]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

    original_embeddings = model.embedding(input_ids).detach().clone()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
    )

    if not torch.isfinite(outputs.loss):
        raise RuntimeError("non-finite Coconut loss")
    if tuple(outputs.logits.shape[:2]) != tuple(input_ids.shape):
        raise RuntimeError(
            f"unexpected logits shape {tuple(outputs.logits.shape)} for input {tuple(input_ids.shape)}"
        )

    outputs.loss.backward()

    grad = base.transformer.h[0].attn.c_attn.weight.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0.0:
        raise RuntimeError("gradient did not propagate through upstream Coconut recurrence")

    latent_positions = [3, 4]
    changed = [
        not torch.equal(
            original_embeddings[:, pos, :],
            outputs.inputs_embeds[:, pos, :].detach(),
        )
        for pos in latent_positions
    ]
    if not all(changed):
        raise RuntimeError("expected latent slots were not replaced by recurrent hidden states")

    upstream_run = Path(os.environ["COCONUT_UPSTREAM"]) / "run.py"
    run_text = upstream_run.read_text(encoding="utf-8")
    trainer_gpu_bound = (
        'dist.init_process_group("nccl")' in run_text
        and "torch.cuda.set_device" in run_text
    )

    payload = {
        "schema_version": 1,
        "evidence_class": "pinned-upstream-core-preflight",
        "upstream": {
            "repository": "facebookresearch/coconut",
            "revision": PINNED_REVISION,
            "module": "coconut.py",
        },
        "device": "cpu",
        "base_model": "randomly-initialized-tiny-gpt2",
        "latent_slots": len(latent_positions),
        "expected_core_forward_passes": len(latent_positions) + 1,
        "loss_finite": True,
        "gradient_through_recurrence": True,
        "latent_slots_replaced": changed,
        "logits_shape": list(outputs.logits.shape),
        "trainer_gpu_bound_as_written": trainer_gpu_bound,
        "trainer_gpu_binding_evidence": [
            'dist.init_process_group("nccl")',
            "torch.cuda.set_device",
        ] if trainer_gpu_bound else [],
        "claim_boundary": (
            "This validates the exact pinned upstream Coconut core recurrence on CPU "
            "with a tiny random GPT-2. It is not a paper reproduction or capability result."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
