"""MLX-native Phi-2 neuron intervention framework.

Original bridge code only: not the authors' MeG training implementation.
"""
from .phi2 import (
    Phi2NeuronWeights,
    Phi2NeuronPatch,
    attach_phi2_neuron,
    parse_meg_phi2_neuron,
    snapshot_phi2_neuron,
    serialize_meg_phi2_neuron,
)
from .controller import StaticNeuronController, NeuronController

__all__ = [
    "Phi2NeuronWeights",
    "Phi2NeuronPatch",
    "attach_phi2_neuron",
    "parse_meg_phi2_neuron",
    "snapshot_phi2_neuron",
    "serialize_meg_phi2_neuron",
    "StaticNeuronController",
    "NeuronController",
]
