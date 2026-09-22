# CPU-only experiment lane

Public GitHub-hosted CPU runners are used for bounded work that does not require accelerator-scale model capability.

## Evidence classes

### Mechanism evidence

Allowed on CPU CI:

- hidden-state recurrence executes;
- gradients propagate through the latent loop;
- direct, serial-control, and latent treatments expose comparable forward-call accounting;
- fixed-depth controls behave deterministically;
- schemas, manifests, probes, and verifiers execute reproducibly.

Tiny random models are intentionally used for these tests. Results are **not** evidence that latent reasoning improves real model capability.

### Procedural-evaluation evidence

CPU CI may build and validate public generators with exact intermediate state, including deterministic finite-state-machine tasks. Private holdout seeds/materialized instances remain outside the public repository.

### Capability evidence

Not accepted from the tiny CPU smoke lane. Claims about useful reasoning performance require a representative trained backbone and a qualified execution substrate.

## Current public GHA mapping

- `ubuntu-slim`: lightweight deterministic generators and verifiers.
- `ubuntu-latest`: tiny PyTorch/Transformers forward/backward mechanism tests.

The CPU lane is intended to fail fast before scarce sovereign GPU time is consumed.
