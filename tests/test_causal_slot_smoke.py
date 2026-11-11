#!/usr/bin/env python3
"""Interface-only tests; not evidence that trained latent states are causal."""
import unittest

import torch

from experiments.causal_slot_smoke import donor_indices, patched_logits
from experiments.register_state_representation_comparison import RecurrentRegisterStateTransformer
from experiments.register_state_saturation import exhaustive_examples


class CausalSlotSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = RecurrentRegisterStateTransformer(
            d_model=32, layers=1, heads=4, max_positions=24
        )
        self.model.eval()
        self.x, self.y, self.lengths = exhaustive_examples(1)

    def test_clone_noop_is_bitwise_identical_to_existing_forward(self):
        with torch.no_grad():
            original, calls = self.model(
                self.x, self.lengths, treatment="latent", recurrent_steps=1
            )
            copy = patched_logits(
                self.model, self.x, self.lengths, kind="clone-noop"
            )
        self.assertEqual(calls, 2)
        self.assertTrue(torch.equal(original, copy))

    def test_donors_have_expected_label_relation(self):
        same = donor_indices(self.y, same_label=True)
        different = donor_indices(self.y, same_label=False)
        self.assertTrue(torch.equal(self.y, self.y[same]))
        self.assertTrue(torch.all(self.y != self.y[different]))
        self.assertTrue(torch.all(same != torch.arange(len(self.y))))
        self.assertTrue(torch.all(different != torch.arange(len(self.y))))

    def test_patch_shapes_and_noise_seed(self):
        different = donor_indices(self.y, same_label=False)
        with torch.no_grad():
            original = patched_logits(
                self.model, self.x, self.lengths, kind="untouched"
            )
            other = patched_logits(
                self.model, self.x, self.lengths,
                kind="different-label", donor=different
            )
            noise1 = patched_logits(
                self.model, self.x, self.lengths,
                kind="norm-matched-noise", seed=9,
            )
            noise2 = patched_logits(
                self.model, self.x, self.lengths,
                kind="norm-matched-noise", seed=9,
            )
        self.assertEqual(original.shape, (len(self.y), 3))
        self.assertEqual(other.shape, original.shape)
        self.assertTrue(torch.equal(noise1, noise2))

    def test_bad_donor_shape_fails_closed(self):
        with self.assertRaises(ValueError):
            patched_logits(
                self.model, self.x, self.lengths,
                kind="different-label", donor=torch.tensor([0]),
            )


if __name__ == "__main__":
    unittest.main()
