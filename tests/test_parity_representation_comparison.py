#!/usr/bin/env python3
import unittest

import torch

from experiments import parity_representation_comparison as exp
from experiments.parity_saturation import ParityTransformer, exhaustive_examples


class ParityRepresentationComparisonTests(unittest.TestCase):
    def test_forward_call_matching(self):
        torch.manual_seed(0)
        model = exp.RecurrentParityTransformer(d_model=32, layers=1, heads=4)
        x, _, lengths = exhaustive_examples(3)

        _, direct = model(x, lengths, treatment="direct", recurrent_steps=2)
        _, serial = model(x, lengths, treatment="serial-control", recurrent_steps=2)
        _, latent = model(x, lengths, treatment="latent", recurrent_steps=2)

        self.assertEqual(direct, 1)
        self.assertEqual(serial, 3)
        self.assertEqual(latent, 3)

    def test_shared_weights_match_frozen_direct_baseline(self):
        torch.manual_seed(17)
        frozen = ParityTransformer(d_model=32, layers=1, heads=4)
        torch.manual_seed(17)
        comparison = exp.RecurrentParityTransformer(d_model=32, layers=1, heads=4)

        frozen_state = frozen.state_dict()
        comparison_state = comparison.state_dict()
        for name, value in frozen_state.items():
            self.assertIn(name, comparison_state)
            self.assertTrue(
                torch.equal(value, comparison_state[name]),
                msg=f"shared parameter diverged: {name}",
            )

    def test_recurrent_treatments_share_parameter_count(self):
        torch.manual_seed(1)
        a = exp.RecurrentParityTransformer(d_model=32, layers=1, heads=4)
        torch.manual_seed(1)
        b = exp.RecurrentParityTransformer(d_model=32, layers=1, heads=4)
        self.assertEqual(
            sum(p.numel() for p in a.parameters()),
            sum(p.numel() for p in b.parameters()),
        )

    def test_latent_gradient_flows_through_recurrence(self):
        torch.manual_seed(2)
        model = exp.RecurrentParityTransformer(d_model=32, layers=1, heads=4)
        x, y, lengths = exhaustive_examples(3)
        logits, calls = model(x, lengths, treatment="latent", recurrent_steps=2)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        grad = model.encoder.layers[0].self_attn.in_proj_weight.grad
        self.assertEqual(calls, 3)
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
