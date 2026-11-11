"""Small GPU-only contract test against the real mlx_lm PhiMLP class.

No Phi-2 checkpoint is downloaded; this is executable framework validation.
"""
import json
from types import SimpleNamespace
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.phi import PhiMLP

from meg_mlx_bridge.controller import InterventionRequest, StaticNeuronController
from meg_mlx_bridge.phi2 import (
    Phi2NeuronPatch, Phi2NeuronWeights, attach_phi2_neuron,
    detach_phi2_neuron, parse_meg_phi2_neuron,
    serialize_meg_phi2_neuron, snapshot_phi2_neuron,
)


class Fixture:
    def __init__(self):
        mx.random.seed(11)
        self.hidden, self.intermediate = 32, 48
        self.mlp = PhiMLP(SimpleNamespace(
            hidden_size=self.hidden, intermediate_size=self.intermediate
        ))
        self.model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=self.mlp)])
        )
        self.x = mx.array(
            (np.arange(2 * 3 * 32, dtype=np.float32).reshape(2, 3, 32) % 11) / 32.0
        )


class BridgeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.set_default_device(mx.gpu)
        assert "gpu" in str(mx.default_device()).lower()

    def setUp(self):
        self.f = Fixture()
        self.original = snapshot_phi2_neuron(self.f.mlp)
        self.tuned = Phi2NeuronWeights(
            self.original.fc1_row + 0.05,
            self.original.fc1_bias + 0.03,
            self.original.fc2_column + 0.02,
        )

    def test_shape_rejection(self):
        with self.assertRaises(ValueError):
            Phi2NeuronWeights(
                mx.zeros((31,)), mx.array(1.0), mx.zeros((32,))
            ).validate(32)
        with self.assertRaises(ValueError):
            parse_meg_phi2_neuron({}, 0, hidden_size=32)
        with self.assertRaises(IndexError):
            attach_phi2_neuron(self.f.model, layer=1, weights=self.tuned)

    def test_author_phi2_json_wire_format(self):
        record = serialize_meg_phi2_neuron(self.tuned, layer=0)
        assert set(record) == {
            "model.layers.0.mlp.fc1.weight",
            "model.layers.0.mlp.fc1.bias",
            "model.layers.0.mlp.fc2.weight",
        }
        loaded = parse_meg_phi2_neuron(
            json.loads(json.dumps(record)), layer=0, hidden_size=32
        )
        for a, b in zip(
            (loaded.fc1_row, loaded.fc1_bias, loaded.fc2_column),
            (self.tuned.fc1_row, self.tuned.fc1_bias, self.tuned.fc2_column),
        ):
            mx.eval(a, b)
            self.assertTrue(np.allclose(np.asarray(a), np.asarray(b), atol=1e-6))

    def test_inject_noop_and_restore_original_object(self):
        x, model = self.f.x, self.f.model
        before = self.f.mlp(x)
        patch = attach_phi2_neuron(model, layer=0, weights=self.tuned, strength=0.0)
        self.assertIsInstance(patch, Phi2NeuronPatch)
        no_edit = model.model.layers[0].mlp(x)
        mx.eval(before, no_edit)
        self.assertTrue(np.array_equal(np.asarray(before), np.asarray(no_edit)))
        with self.assertRaises(RuntimeError):
            attach_phi2_neuron(model, layer=0, weights=self.tuned)
        detach_phi2_neuron(model, layer=0, patch=patch)
        self.assertIs(model.model.layers[0].mlp, self.f.mlp)

    def test_edit_matches_full_direct_weight_replacement(self):
        mlp, x = self.f.mlp, self.f.x
        patch = attach_phi2_neuron(
            self.f.model, layer=0, weights=self.tuned, strength=1.0
        )
        output = self.f.model.model.layers[0].mlp(x)
        # Independently construct the full fc1/fc2 matrices that MeG would
        # modify in PyTorch. Never mutate the frozen reference MLP.
        fc1 = mx.concatenate(
            [mlp.fc1.weight[:-1], self.tuned.fc1_row[None, :]], axis=0
        )
        bias = mx.concatenate(
            [mlp.fc1.bias[:-1], self.tuned.fc1_bias.reshape((1,))], axis=0
        )
        fc2 = mx.concatenate(
            [mlp.fc2.weight[:, :-1], self.tuned.fc2_column[:, None]], axis=1
        )
        expected = (nn.gelu_approx(x @ fc1.T + bias) @ fc2.T) + mlp.fc2.bias
        mx.eval(output, expected)
        self.assertTrue(
            np.allclose(
                np.asarray(output), np.asarray(expected), rtol=5e-4, atol=5e-4
            ),
            "functional MLX wrapper disagrees with full MeG-style weight edit",
        )
        detach_phi2_neuron(self.f.model, layer=0, patch=patch)

    def test_controller_and_intervention_gradient(self):
        controller = StaticNeuronController(self.tuned)
        req = InterventionRequest(task_id="public-synthetic", layer=0, hidden_size=32, seed=7)
        proposed = controller.generate(req)
        def objective(input_row):
            changed = Phi2NeuronWeights(input_row, proposed.fc1_bias, proposed.fc2_column)
            return mx.sum(Phi2NeuronPatch(self.f.mlp, changed)(self.f.x) ** 2)
        grad = mx.grad(objective)(proposed.fc1_row)
        mx.eval(grad)
        self.assertTrue(np.isfinite(np.asarray(grad)).all())
        self.assertTrue(np.any(np.abs(np.asarray(grad)) > 1e-9))


    def test_real_phi2_layer_dimensions_without_checkpoint(self):
        """2560->10240->2560 structural check, not whole model inference."""
        width, intermediate = 2560, 10240
        mlp = PhiMLP(SimpleNamespace(
            hidden_size=width, intermediate_size=intermediate
        ))
        target = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])
        )
        x = mx.ones((1, 1, width), dtype=mx.float32) / width
        baseline = mlp(x)
        original = snapshot_phi2_neuron(mlp)
        replacement = Phi2NeuronWeights(
            original.fc1_row + 0.001,
            original.fc1_bias + 0.001,
            original.fc2_column + 0.001,
        )
        patch = attach_phi2_neuron(
            target, layer=0, weights=replacement, strength=0.25
        )
        result = target.model.layers[0].mlp(x)
        mx.eval(baseline, result)
        self.assertEqual(result.shape, (1, 1, width))
        self.assertTrue(np.isfinite(np.asarray(result)).all())
        self.assertFalse(np.array_equal(np.asarray(baseline), np.asarray(result)))
        detach_phi2_neuron(target, layer=0, patch=patch)
        self.assertIs(target.model.layers[0].mlp, mlp)

if __name__ == "__main__":
    unittest.main()
