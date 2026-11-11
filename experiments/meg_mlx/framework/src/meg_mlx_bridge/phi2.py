"""MeG-style Phi-2 last-neuron replacement using a reversible MLX module.

The upstream MeG Phi-2 trainer rewrites the LAST fc1 row, LAST fc1 bias
and LAST fc2 column. This wrapper reproduces that *intervention topology*
without mutating the model's frozen original weights. It is not a trained
MeG diffusion controller nor a paper-result reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Any

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class Phi2NeuronWeights:
    fc1_row: mx.array
    fc1_bias: mx.array
    fc2_column: mx.array

    @property
    def hidden_size(self) -> int:
        return int(self.fc1_row.shape[0])

    def validate(self, hidden_size: int) -> None:
        if self.fc1_row.shape != (hidden_size,):
            raise ValueError("fc1_row must have shape [hidden_size]")
        if self.fc1_bias.shape != ():
            raise ValueError("fc1_bias must be a scalar")
        if self.fc2_column.shape != (hidden_size,):
            raise ValueError("fc2_column must have shape [hidden_size]")


def _names(layer: int) -> tuple[str, str, str]:
    if layer < 0:
        raise ValueError("layer must be a non-negative physical Phi-2 layer")
    root = f"model.layers.{layer}.mlp."
    return root + "fc1.weight", root + "fc1.bias", root + "fc2.weight"


def parse_meg_phi2_neuron(
    record: Mapping[str, Any], layer: int, *, hidden_size: int
) -> Phi2NeuronWeights:
    """Read a ONE-neuron upstream params_0.json record, not a whole checkpoint.

    Requires fc1 last-row [1,H], bias [1], fc2 last-column [H,1].
    Multi-neuron records are deliberately unsupported until separately tested.
    """
    kw, kb, kv = _names(layer)
    try:
        win, bias, wout = mx.array(record[kw]), mx.array(record[kb]), mx.array(record[kv])
    except KeyError as exc:
        raise ValueError(f"missing upstream neuron field: {exc.args[0]}") from exc
    if win.shape != (1, hidden_size) or bias.shape != (1,) or wout.shape != (hidden_size, 1):
        raise ValueError("MeG Phi-2 record has unexpected one-neuron tensor shapes")
    result = Phi2NeuronWeights(
        fc1_row=win[0],
        fc1_bias=bias.reshape(()),
        fc2_column=wout[:, 0],
    )
    result.validate(hidden_size)
    return result


def serialize_meg_phi2_neuron(weights: Phi2NeuronWeights, layer: int) -> dict:
    """Emit the original MeG three-key, one-neuron JSON record layout."""
    weights.validate(weights.hidden_size)
    kw, kb, kv = _names(layer)
    mx.eval(weights.fc1_row, weights.fc1_bias, weights.fc2_column)
    return {
        kw: [weights.fc1_row.tolist()],
        kb: [float(weights.fc1_bias.item())],
        kv: [[float(val)] for val in weights.fc2_column.tolist()],
    }


def snapshot_phi2_neuron(mlp: nn.Module) -> Phi2NeuronWeights:
    """Snapshot the original final MLP neuron for a safe no-edit control."""
    if not hasattr(mlp, "fc1") or not hasattr(mlp, "fc2"):
        raise TypeError("Expected a Phi-2-style MLP with fc1 and fc2")
    fc1, fc2 = mlp.fc1, mlp.fc2
    if fc1.weight.ndim != 2 or fc2.weight.ndim != 2:
        raise ValueError("Expected 2-D projection weights")
    width = int(fc1.weight.shape[1])
    if fc2.weight.shape != (width, fc1.weight.shape[0]):
        raise ValueError("fc1/fc2 dimensions are incompatible")
    if fc1.bias is None or fc1.bias.shape != (fc1.weight.shape[0],):
        raise ValueError("Phi-2 fc1 requires a per-neuron bias")
    return Phi2NeuronWeights(
        fc1_row=fc1.weight[-1],
        fc1_bias=fc1.bias[-1],
        fc2_column=fc2.weight[:, -1],
    )


class Phi2NeuronPatch(nn.Module):
    """Reversible Phi-2 MLP wrapper applying an exact last-neuron replacement.

    Model's original MLP is retained as 'base'. The intervention adds the
    modified neuron's contribution and removes its original contribution.
    It therefore represents REPLACEMENT, not an invented extra neuron.
    Disabled/strength=0 returns the untouched original MLP output.
    """

    def __init__(
        self, base: nn.Module, weights: Phi2NeuronWeights,
        *, strength: float = 1.0
    ) -> None:
        super().__init__()
        if not isfinite(strength) or strength < 0.0 or strength > 1.0:
            raise ValueError("strength must be finite in [0,1]")
        self.base = base
        self.original = snapshot_phi2_neuron(base)
        weights.validate(self.original.hidden_size)
        self.replacement = weights
        self.strength = float(strength)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        if self.strength == 0.0:
            return y
        old = self.original
        new = self.replacement
        old_pre = (x @ old.fc1_row) + old.fc1_bias
        new_pre = (x @ new.fc1_row) + new.fc1_bias
        old_act = nn.gelu_approx(old_pre)
        new_act = nn.gelu_approx(new_pre)
        delta = (
            new_act[..., None] * new.fc2_column -
            old_act[..., None] * old.fc2_column
        )
        return y + self.strength * delta


def attach_phi2_neuron(
    model: nn.Module, *, layer: int, weights: Phi2NeuronWeights, strength: float = 1.0
) -> Phi2NeuronPatch:
    """Attach a reversible edit to mlx_lm.models.phi.Model-compatible layout.

    Caller must retain the returned wrapper, then invoke detach_phi2_neuron
    for isolation between evaluation trials.
    """
    layers = model.model.layers
    if layer < 0 or layer >= len(layers):
        raise IndexError("Phi-2 layer out of range")
    mlp = layers[layer].mlp
    if isinstance(mlp, Phi2NeuronPatch):
        raise RuntimeError("Phi-2 MLP already patched; detach before new trial")
    patch = Phi2NeuronPatch(mlp, weights, strength=strength)
    layers[layer].mlp = patch
    return patch


def detach_phi2_neuron(model: nn.Module, *, layer: int, patch: Phi2NeuronPatch) -> None:
    """Restore the unmodified MLP; refuse to detach someone else's patch."""
    layers = model.model.layers
    if layers[layer].mlp is not patch:
        raise RuntimeError("Refusing to detach a different intervention")
    layers[layer].mlp = patch.base
