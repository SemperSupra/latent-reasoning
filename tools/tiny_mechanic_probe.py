#!/usr/bin/env python3
import json
import time

import torch

from test_latent_mechanics import build_tiny_model, direct_forward, recurrent_forward


def timed(fn):
    start = time.perf_counter()
    logits, calls = fn()
    elapsed = time.perf_counter() - start
    return {
        "forward_calls": calls,
        "wall_seconds": elapsed,
        "finite": bool(torch.isfinite(logits).all().item()),
        "shape": list(logits.shape),
    }


def main():
    torch.set_num_threads(1)
    model = build_tiny_model()
    model.eval()
    input_ids = torch.tensor([[1, 7, 11, 13, 17]], dtype=torch.long)

    with torch.no_grad():
        result = {
            "schema_version": 1,
            "evidence_class": "mechanism-smoke-only",
            "model": {
                "kind": "randomly-initialized-tiny-gpt2",
                "layers": 2,
                "embedding_dim": 32,
                "heads": 2,
                "vocab_size": 97,
            },
            "treatments": {
                "direct": timed(lambda: direct_forward(model, input_ids)),
                "serial_control_k2": timed(
                    lambda: recurrent_forward(
                        model, input_ids, 2, mode="serial-control"
                    )
                ),
                "latent_k2": timed(
                    lambda: recurrent_forward(model, input_ids, 2, mode="latent")
                ),
            },
            "claim_boundary": (
                "This receipt validates executable mechanics and accounting only; "
                "it is not model-capability evidence."
            ),
        }

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
