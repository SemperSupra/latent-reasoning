#!/usr/bin/env python3
"""WP4b: exact upstream Coconut + reference GPT-2 CPU training smoke.

This is a preflight of the pinned Coconut core using the reference backbone
named by the official GSM8K configuration. It is not a dataset/paper reproduction.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PINNED_REVISION = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"
MODEL_ID = "openai-community/gpt2"


def load_coconut_class():
    module_path = Path(os.environ["COCONUT_UPSTREAM"]) / "coconut.py"
    spec = importlib.util.spec_from_file_location("pinned_coconut", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import pinned upstream Coconut")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Coconut


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    base.resize_token_embeddings(len(tokenizer))

    # Mirror upstream initialization intent for the three newly added tokens.
    embedding = base.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    with torch.no_grad():
        for token_id in (latent_id, start_id, end_id):
            embedding.weight[token_id].copy_(embedding.weight[target_id])
            base.lm_head.weight[token_id].copy_(base.lm_head.weight[target_id])

    Coconut = load_coconut_class()
    model = Coconut(
        base,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prefix = tokenizer.encode("Question: 2 + 3 =")
    suffix = tokenizer.encode(" Answer: 5") + [tokenizer.eos_token_id]
    tokens = prefix + [start_id, latent_id, latent_id, end_id] + suffix
    input_ids = torch.tensor([tokens], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    tracked = base.transformer.h[0].attn.c_attn.weight
    before = tracked.detach().clone()

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
    )
    forward_seconds = time.perf_counter() - started

    if not torch.isfinite(outputs.loss):
        raise RuntimeError("non-finite loss")

    outputs.loss.backward()
    grad = tracked.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0.0:
        raise RuntimeError("reference-backbone gradient did not propagate")

    optimizer.step()
    parameter_changed = not torch.equal(before, tracked.detach())
    if not parameter_changed:
        raise RuntimeError("optimizer step did not update tracked GPT-2 parameters")

    payload = {
        "schema_version": 1,
        "evidence_class": "pinned-upstream-reference-backbone-training-smoke",
        "upstream": {
            "repository": "facebookresearch/coconut",
            "revision": PINNED_REVISION,
            "module": "coconut.py",
        },
        "backbone": MODEL_ID,
        "parameter_count": sum(p.numel() for p in base.parameters()),
        "device": "cpu",
        "latent_slots": 2,
        "input_tokens": int(input_ids.shape[1]),
        "loss": float(outputs.loss.detach()),
        "loss_finite": True,
        "gradient_through_recurrence": True,
        "optimizer_step_changed_parameter": parameter_changed,
        "forward_seconds": forward_seconds,
        "claim_boundary": (
            "This validates one CPU train step using the pinned upstream Coconut core "
            "and its reference GPT-2 backbone. It is not a GSM8K or paper reproduction."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
