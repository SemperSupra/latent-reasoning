# latent-reasoning

Reusable implementations and reproducible workflows for **latent / continuous reasoning**: iterative computation performed in learned internal representations rather than requiring every intermediate step to be serialized as natural language.

## Scope

This repository owns promoted, reusable implementations for:

- continuous or latent chain-of-thought
- recurrent hidden-state reasoning
- latent reasoning adapters / bridges
- adaptive reasoning depth and halting
- latent-vs-explicit reasoning routing
- compute-matched controls used to distinguish representation effects from extra serial compute
- training and inference interfaces needed to reproduce those mechanisms

It does **not** own generic model fine-tuning, artifact identity/provenance, independent qualification, or actor scheduling.

## Repository pair

- `SemperSupra/latent-reasoning-private` — authoritative R&D, experiments, private holdouts, negative results, and promotion decisions.
- `SemperSupra/latent-reasoning` — promoted public implementation and reproducibility surface.

## Portfolio boundaries

- **Model Artifact Foundry** identifies and packages exact model artifacts, adapters, checkpoints, manifests, hashes, lineage, and hydration metadata.
- **Model Spelunker** independently qualifies candidate reasoning substrates and measures capability/performance envelopes.
- **Agent Dispatch** consumes qualified combinations and binds them to workloads and compute substrates.
- **Latent Reasoning** develops the reasoning mechanisms themselves.

See [docs/integration-contracts.md](docs/integration-contracts.md).

## Initial qualification question

The first campaign must try to falsify the claim that latent reasoning adds value beyond extra serial compute.

The minimum comparison is:

1. direct answer
2. explicit textual chain-of-thought
3. compute-matched recurrent / pause control
4. fixed-depth latent reasoning
5. adaptive-depth latent reasoning, after fixed-depth behavior is understood

Evaluation must include correctness, OOD behavior, latency, GPU-seconds, peak memory, generated tokens, latent iterations, estimated compute, seed variance, and perturbation robustness.

## Method layout

Method-specific implementations belong beneath `methods/` and must not define the repository architecture. Coconut, CODI, SpiralThinker, recurrent-state methods, and later approaches are interchangeable implementations of the same higher-level capability.

## Status

Bootstrap phase. The first work is reproduction, falsification, and interface stabilization before novel architecture work.
