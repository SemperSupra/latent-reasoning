# CPU-only experiment lane

Public GitHub-hosted CPU runners are used primarily as a bounded **method-development and execution-learning environment**. They let the project learn how to package, orchestrate, control, validate, and reproduce experiments before moving them onto sovereign/local resources.

They are not the target deployment or final efficacy substrate.

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

Not accepted as final authority from the GHA CPU lane. GHA may produce preliminary descriptive performance evidence when a benchmark, control, and model are sufficiently representative, but the program should not spend GHA complexity or minutes merely to prove efficacy.

Stronger capability/generalization claims require replay on the intended sovereign/local execution substrate with hardware/software substrate identity preserved in the run record.

## Current public GHA mapping

- `ubuntu-slim`: lightweight deterministic generators and verifiers.
- `ubuntu-latest`: tiny PyTorch/Transformers forward/backward mechanism tests.

The CPU lane is intended to fail fast and teach the reusable experiment machinery before sovereign resources are consumed.

A successful GHA experiment should therefore leave behind enough immutable configuration, manifests, receipts, seeds, and control semantics to be replayed locally with minimal translation. Differences between GHA and sovereign/local replay are themselves useful substrate-portability evidence.
