#!/usr/bin/env python3
import unittest

import torch

from experiments import toy_modular_walk as toy


class ToyModularWalkTests(unittest.TestCase):
    def test_examples_are_deterministic_and_correct(self):
        x1, y1 = toy.make_examples(42, 16, action_steps=6)
        x2, y2 = toy.make_examples(42, 16, action_steps=6)
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(y1, y2))

        for row, label in zip(x1.tolist(), y1.tolist()):
            state = row[0]
            for token in row[1:]:
                delta = toy.ACTION_DELTAS[token - toy.ACTION_TOKEN_BASE]
                state = (state + delta) % toy.STATE_COUNT
            self.assertEqual(state, label)

    def test_serial_and_latent_have_matched_forward_calls(self):
        torch.manual_seed(0)
        model = toy.TinyReasoner()
        x, _ = toy.make_examples(7, 4, action_steps=6)

        _, serial_calls = model(x, "serial-control", latent_steps=3)
        _, latent_calls = model(x, "latent", latent_steps=3)

        self.assertEqual(serial_calls, 4)
        self.assertEqual(latent_calls, 4)

    def test_latent_training_path_updates_parameters(self):
        torch.manual_seed(0)
        model = toy.TinyReasoner()
        x, y = toy.make_examples(9, 32, action_steps=6)
        before = model.classifier.weight.detach().clone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        logits, _ = model(x, "latent", latent_steps=2)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(before, model.classifier.weight.detach()))


if __name__ == "__main__":
    unittest.main()
