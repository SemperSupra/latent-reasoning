# Integration contracts

Latent Reasoning owns development of trainable latent/continuous reasoning mechanisms. Neighboring projects consume its outputs through narrow contracts.

## Model Artifact Foundry

Foundry is the artifact identity and provenance plane.

For any promoted trained output, Latent Reasoning should provide enough information for Foundry to identify and reconstruct:

- immutable base-model identity and revision
- method / implementation revision
- adapter, reasoner, projection, controller, or checkpoint hashes
- training configuration
- dataset / split manifests or immutable references
- tokenizer and preprocessing identity
- software environment identity
- parent artifact lineage
- hydration / reconstruction instructions

Foundry does not decide whether an artifact is good, general, or operationally useful.

## Model Spelunker

Model Spelunker is an independent qualification consumer.

Latent Reasoning should expose candidates and machine-readable run configuration sufficient to compare:

- direct inference
- explicit textual reasoning
- compute-matched recurrence / pause controls
- fixed-depth latent reasoning
- adaptive-depth latent reasoning

Qualification results belong to the qualification plane rather than becoming claims embedded in the training implementation.

## Agent Dispatch

Agent Dispatch consumes already-described and, where required, already-qualified actor realizations.

A reasoning substrate is one independent dimension of an actor realization, alongside model, harness/framework, compute substrate, task class, and other execution constraints.

Agent Dispatch does not own latent-reasoning training or qualification methodology.

## Promotion

Promotion from the private R&D repository to this repository requires:

1. a reproducible implementation or recipe;
2. no dependency on private holdout contents or unpublished secrets;
3. immutable artifact references where trained outputs are involved;
4. enough metadata for independent qualification;
5. explicit statement of known limitations and unsupported claims.
