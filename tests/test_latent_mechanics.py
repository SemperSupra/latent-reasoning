#!/usr/bin/env python3
import math
import unittest

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel


def build_tiny_model(seed: int = 1234) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=97,
        n_positions=48,
        n_ctx=48,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        bos_token_id=1,
        eos_token_id=2,
    )
    return GPT2LMHeadModel(config)


def direct_forward(model: GPT2LMHeadModel, input_ids: torch.Tensor):
    output = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    return output.logits[:, -1, :], 1


def recurrent_forward(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    steps: int,
    *,
    mode: str,
):
    if mode not in {"latent", "serial-control"}:
        raise ValueError(f"unsupported mode: {mode}")

    embeddings = model.transformer.wte(input_ids)
    calls = 0

    for _ in range(steps):
        output = model(
            inputs_embeds=embeddings,
            use_cache=False,
            output_hidden_states=True,
        )
        calls += 1

        if mode == "latent":
            next_embedding = output.hidden_states[-1][:, -1:, :]
        else:
            # A fixed pause-like embedding preserves the extra serial forward
            # passes without claiming that hidden-state content is useful.
            pause_id = torch.zeros(
                (input_ids.shape[0], 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            next_embedding = model.transformer.wte(pause_id)

        embeddings = torch.cat((embeddings, next_embedding), dim=1)

    output = model(
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=True,
    )
    calls += 1
    return output.logits[:, -1, :], calls


class TinyLatentMechanicsTests(unittest.TestCase):
    def setUp(self):
        self.model = build_tiny_model()
        self.input_ids = torch.tensor([[1, 7, 11, 13, 17]], dtype=torch.long)

    def test_direct_is_one_forward_pass(self):
        logits, calls = direct_forward(self.model, self.input_ids)
        self.assertEqual(calls, 1)
        self.assertEqual(tuple(logits.shape), (1, 97))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_latent_and_serial_controls_use_equal_forward_calls(self):
        for steps in (1, 2, 4):
            latent_logits, latent_calls = recurrent_forward(
                self.model, self.input_ids, steps, mode="latent"
            )
            serial_logits, serial_calls = recurrent_forward(
                self.model, self.input_ids, steps, mode="serial-control"
            )
            self.assertEqual(latent_calls, steps + 1)
            self.assertEqual(serial_calls, steps + 1)
            self.assertEqual(latent_calls, serial_calls)
            self.assertEqual(latent_logits.shape, serial_logits.shape)
            self.assertTrue(torch.isfinite(latent_logits).all().item())
            self.assertTrue(torch.isfinite(serial_logits).all().item())

    def test_gradient_flows_through_latent_recurrence(self):
        self.model.zero_grad(set_to_none=True)
        logits, calls = recurrent_forward(
            self.model, self.input_ids, 3, mode="latent"
        )
        target = torch.tensor([23], dtype=torch.long)
        loss = F.cross_entropy(logits, target)
        self.assertTrue(math.isfinite(loss.item()))
        self.assertEqual(calls, 4)

        loss.backward()
        grad = self.model.transformer.h[0].attn.c_attn.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_latent_recurrence_is_deterministic_in_eval_mode(self):
        self.model.eval()
        with torch.no_grad():
            logits_a, calls_a = recurrent_forward(
                self.model, self.input_ids, 2, mode="latent"
            )
            logits_b, calls_b = recurrent_forward(
                self.model, self.input_ids, 2, mode="latent"
            )
        self.assertEqual(calls_a, calls_b)
        self.assertTrue(torch.equal(logits_a, logits_b))


if __name__ == "__main__":
    unittest.main()
