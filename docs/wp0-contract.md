# WP0 method-neutral contract

Campaign 0001 needs one experiment-side description that can travel between Latent Reasoning, Model Artifact Foundry, Model Spelunker, and Agent Dispatch without moving ownership between those projects.

## Reconciled existing interfaces

### Model Artifact Foundry

Foundry already defines strict candidate identity, exact upstream revision, file hashes, validation claims, compatibility profiles, and immutable OCI identity in `candidate-manifest-v2.schema.json`.

Latent Reasoning should reference a Foundry-managed model or promoted composite artifact through `foundry_ref` rather than copying the Foundry schema.

Current Foundry v2 is model-representation-centric. It does not yet expose a generic first-class convention for small derived components such as LoRA adapters, latent reasoners, projections, or halting controllers. Latent Reasoning therefore records component identity locally with immutable hashes and optional `artifact_ref` values, while treating broader Foundry support for derived components as an integration gap rather than inventing a competing artifact registry.

### Model Spelunker

The configured-actor schema already has an open `configuration` object. A `reasoning_substrate` descriptor can be embedded there without changing Spelunker's actor-profile schema.

The existing qualification unit remains:

`task × harness × build × model × configuration × toolset × substrate`

Latent reasoning is therefore represented as controlled configuration until experiments show that a dedicated Spelunker schema field earns its keep.

### Agent Dispatch

Agent Dispatch remains the execution/routing authority for bounded work. Its public contract currently dispatches approved task IDs and approved target IDs; it does not need a new public routing primitive merely to start Campaign 0001.

Reasoning-substrate selection should stay in bounded task/workset configuration until repeated use demonstrates a stable dispatch-level primitive.

## New local contracts

### `reasoning-substrate.schema.json`

Describes the reasoning treatment itself:

- control class: direct, textual CoT, serial-compute control, or latent;
- implementation family and revision;
- fixed/adaptive/no-extra-compute policy;
- optional adapter/reasoner/controller/projection component identities;
- method-specific configuration.

It deliberately does not contain task outcome claims.

### `run-manifest.schema.json`

Describes one train/evaluate/probe run:

- exact backbone identity when available;
- one reasoning-substrate descriptor;
- workload/split identity;
- seed and external actor/environment/harness references;
- optional compute-match relationship;
- exact Latent Reasoning repository revision.

The manifest is input/provenance configuration. Model Spelunker receipts remain the qualification evidence/output plane.

## Invariants

1. No duplicate artifact registry.
2. No new scheduler or routing layer.
3. No universal winner score.
4. A latent result is not promoted without a meaningful serial-compute control where technically possible.
5. Artifact identity and behavioral qualification remain separate authorities.
6. Private holdout contents never need to enter the public repository.
7. GitHub Actions is primarily a method-development and execution-learning substrate for Campaign 0001. This purpose does not reduce the scientific status of results obtained there: findings that satisfy the experiment's validity gates remain evidence with the strength warranted by their design. Sovereign/local replay is used to test portability, substrate sensitivity, scale, and external validity—not to retroactively legitimize otherwise valid GHA findings. Hardware/software substrate remains an explicit qualification variable.
