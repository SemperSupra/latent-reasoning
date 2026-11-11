"""Backend-neutral controller boundary for future generated MLX neuron weights.

A controller may use a deterministic table, a trained diffusion model, or a
document-derived encoder. No scientific method is implied by the interface.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

from .phi2 import Phi2NeuronWeights


@dataclass(frozen=True)
class InterventionRequest:
    task_id: str
    layer: int
    hidden_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.task_id or self.layer < 0 or self.hidden_size < 1 or self.seed < 0:
            raise ValueError("Invalid intervention request")


class NeuronController(Protocol):
    """All generated weight controllers must provide this deterministic API."""

    def generate(self, request: InterventionRequest) -> Phi2NeuronWeights:
        """Return one-neuron weights for a specified task/layer/seed."""
        ...


@dataclass(frozen=True)
class StaticNeuronController:
    """Known-weight reference controller, never claims diffusion capability."""

    weights: Phi2NeuronWeights

    def generate(self, request: InterventionRequest) -> Phi2NeuronWeights:
        self.weights.validate(request.hidden_size)
        return self.weights
