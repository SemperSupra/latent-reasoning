#!/usr/bin/env python3
import unittest

import torch

from experiments import register_state_representation_comparison as exp
from experiments.register_state_saturation import (
    RegisterStateTransformer,
    exhaustive_examples,
)


class RegisterStateComparisonTests(unittest.TestCase):
    def test_shared_weights_match_frozen_direct(self):
        torch.manual_seed(31)
        frozen = RegisterStateTransformer(d_model=32, layers=1, heads=4)
        torch.manual_seed(31)
        comparison = exp.RecurrentRegisterStateTransformer(d_model=32, layers=1, heads=4)
        comparison_state = comparison.state_dict()
        for name, value in frozen.state_dict().items():
            self.assertIn(name, comparison_state)
            self.assertTrue(torch.equal(value, comparison_state[name]), name)

    def test_forward_call_matching(self):
        model = exp.RecurrentRegisterStateTransformer(d_model=32, layers=1, heads=4)
        x, _, lengths = exhaustive_examples(1)
        _, direct = model(x, lengths, treatment="direct", recurrent_steps=2)
        _, serial = model(x, lengths, treatment="serial-control", recurrent_steps=2)
        _, latent = model(x, lengths, treatment="latent", recurrent_steps=2)
        self.assertEqual(direct, 1)
        self.assertEqual(serial, 3)
        self.assertEqual(latent, 3)

    def test_latent_gradient_flows(self):
        torch.manual_seed(7)
        model = exp.RecurrentRegisterStateTransformer(d_model=32, layers=1, heads=4)
        x, y, lengths = exhaustive_examples(1)
        logits, calls = model(x, lengths, treatment="latent", recurrent_steps=2)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        grad = model.encoder.layers[0].self_attn.in_proj_weight.grad
        self.assertEqual(calls, 3)
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
