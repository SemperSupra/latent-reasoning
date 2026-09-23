# CPU-only experiment lane

Public GitHub-hosted CPU runners are used primarily as a bounded **method-development and execution-learning environment**. They let the project learn how to package, orchestrate, control, validate, and reproduce experiments before moving them onto sovereign/local resources.

They are not the target deployment substrate, and proving efficacy is not their primary purpose. Scientifically valid results produced on GHA are nevertheless retained and interpreted normally according to the strength and limits of their experimental design.

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

GHA may produce scientifically valid capability or generalization evidence when the benchmark, controls, model, sample size, and analysis support that conclusion. Such findings are not discarded or automatically downgraded because they came from GHA.

The program should still avoid spending GHA complexity or minutes merely to maximize efficacy evidence, because method-development and execution learning are the primary goals of this substrate.

Later sovereign/local replay answers additional questions—portability, substrate sensitivity, scale behavior, and whether findings generalize to the intended execution environment. Replay strengthens or qualifies the evidence base; it is not a prerequisite that retroactively makes a valid GHA result scientific.

## Current public GHA mapping

- `ubuntu-slim`: lightweight deterministic generators and verifiers.
- `ubuntu-latest`: tiny PyTorch/Transformers forward/backward mechanism tests.

The CPU lane is intended to fail fast and teach the reusable experiment machinery before sovereign resources are consumed.

A successful GHA experiment should therefore leave behind enough immutable configuration, manifests, receipts, seeds, and control semantics to be replayed locally with minimal translation. Differences between GHA and sovereign/local replay are themselves useful substrate-portability evidence.
