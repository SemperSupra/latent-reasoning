#!/usr/bin/env python3
import unittest

import torch

from experiments import parity_saturation as exp


class ParitySaturationTests(unittest.TestCase):
    def test_exhaustive_universe_size(self):
        for max_length in (1, 2, 4, 8):
            x, y, lengths = exp.exhaustive_examples(max_length)
            self.assertEqual(len(y), 2 ** (max_length + 1) - 2)
            self.assertEqual(x.shape[0], len(y))
            self.assertEqual(len(lengths), len(y))

    def test_exhaustive_labels_are_exact(self):
        x, y, lengths = exp.exhaustive_examples(5)
        for row, label, length in zip(x.tolist(), y.tolist(), lengths.tolist()):
            bit_tokens = row[: length - 1]
            bits = tuple(token - exp.BIT0 for token in bit_tokens)
            self.assertEqual(exp.parity(bits), label)
            self.assertEqual(row[length - 1], exp.CLS)

    def test_ood_sampling_respects_lengths_and_labels(self):
        sampled = exp.sampled_ood_examples(
            123,
            min_length=9,
            max_length=11,
            per_length=32,
        )
        self.assertEqual(set(sampled), {9, 10, 11})
        for length, (x, y, lengths) in sampled.items():
            self.assertEqual(len(y), 32)
            self.assertTrue(torch.equal(lengths, torch.full_like(lengths, length + 1)))
            for row, label in zip(x.tolist(), y.tolist()):
                bits = tuple(token - exp.BIT0 for token in row[:-1])
                self.assertEqual(exp.parity(bits), label)

    def test_model_forward_shape(self):
        x, y, lengths = exp.exhaustive_examples(3)
        model = exp.ParityTransformer(d_model=32, layers=1, heads=4)
        logits = model(x, lengths)
        self.assertEqual(tuple(logits.shape), (len(y), 2))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
