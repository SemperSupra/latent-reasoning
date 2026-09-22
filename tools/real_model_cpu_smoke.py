#!/usr/bin/env python3
"""Representative pretrained-model CPU recurrence compatibility smoke.

This is an interface/mechanism test, not a capability evaluation.
"""

from __future__ import annotations

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "Return the next token after this short prompt: 2 + 2 ="


def run_forward(model, *, input_ids=None, inputs_embeds=None):
    started = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=False,
        )
    return out, time.perf_counter() - started


def recurrent(model, input_ids, steps: int, mode: str):
    if mode not in {"latent", "serial-control"}:
        raise ValueError(mode)

    embedding_layer = model.get_input_embeddings()
    current = embedding_layer(input_ids)
    pause_id = torch.tensor([[0]], dtype=input_ids.dtype)
    pause_embedding = embedding_layer(pause_id)
    calls = 0
    elapsed = 0.0

    for _ in range(steps):
        out, dt = run_forward(model, inputs_embeds=current)
        calls += 1
        elapsed += dt
        if mode == "latent":
            slot = out.hidden_states[-1][:, -1:, :]
        else:
            slot = pause_embedding
        current = torch.cat((current, slot), dim=1)

    out, dt = run_forward(model, inputs_embeds=current)
    calls += 1
    elapsed += dt
    return out, calls, elapsed, current.shape


def top_token(logits):
    return int(logits[:, -1, :].argmax(dim=-1).item())


def main():
    torch.set_num_threads(2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.eval()

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"]

    direct, direct_seconds = run_forward(model, input_ids=input_ids)

    treatments = {
        "direct": {
            "forward_calls": 1,
            "wall_seconds": direct_seconds,
            "top_token_id": top_token(direct.logits),
            "hidden_state_shape": list(direct.hidden_states[-1].shape),
        }
    }

    for steps in (1, 2, 4):
        serial, serial_calls, serial_seconds, serial_shape = recurrent(
            model, input_ids, steps, "serial-control"
        )
        latent, latent_calls, latent_seconds, latent_shape = recurrent(
            model, input_ids, steps, "latent"
        )

        if serial_calls != latent_calls:
            raise RuntimeError("serial and latent forward-call counts diverged")

        treatments[f"serial_control_k{steps}"] = {
            "forward_calls": serial_calls,
            "wall_seconds": serial_seconds,
            "top_token_id": top_token(serial.logits),
            "final_embedding_shape": list(serial_shape),
        }
        treatments[f"latent_k{steps}"] = {
            "forward_calls": latent_calls,
            "wall_seconds": latent_seconds,
            "top_token_id": top_token(latent.logits),
            "final_embedding_shape": list(latent_shape),
            "hidden_state_shape": list(latent.hidden_states[-1].shape),
        }

    payload = {
        "schema_version": 1,
        "evidence_class": "representative-pretrained-model-interface-smoke",
        "model_id": MODEL_ID,
        "model_class": model.__class__.__name__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "prompt_token_count": int(input_ids.shape[1]),
        "inputs_embeds_supported": True,
        "hidden_states_supported": True,
        "treatments": treatments,
        "claim_boundary": (
            "This receipt establishes pretrained-model interface compatibility and "
            "recurrent execution mechanics only. It is not reasoning-capability evidence."
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
