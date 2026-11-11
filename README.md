# Model Mechanisms

> **Proposed repository rename:** `SemperSupra/model-mechanisms`. The active repository is currently `SemperSupra/latent-reasoning`; rename of the existing repository (not recreation) is owner-controlled. Private sibling: currently `SemperSupra/latent-reasoning-private`, proposed `SemperSupra/model-mechanisms-private`.

## Mission

Reconstruct, reproduce, experimentally investigate, and develop **model-internal mechanisms that change or extend AI model capabilities**. Evaluate mechanisms against explicit controls, held-out tasks, and resource-adjusted baselines rather than assuming that an interesting architecture improves capability.

The research scope includes:
- Latent and continuous reasoning (Coconut, CODI, recurrent-state methods, adaptive compute).
- Knowledge editing and augmentation (MeG, generated neurons, deterministic and diffusion hypernetworks, conventional adapter controls).
- Recurrent model dynamics (Dory loop depth, pass-specific interventions, loop scheduling).
- Numerical implementation parity and portable experimental interfaces across CPU, PyTorch/CUDA, MLX, and explicitly qualified TPU/JAX pathways.
- Reproduction of published methods followed by pre-registered Design of Experiments (DOE), falsification, independent qualification, and documented negative results.

**Do not conflate:** faithful reconstruction of an author's published experiment, a build/portability smoke test, an independently replicated result, and a novel scientific improvement.

## Public/private authority

This **public** repository owns reproducible, public-safe implementations, typed experiment/replay contracts, benchmark adapters, validated source/packaging tools, execution workflows, and sanitized evidence.

The **private** sibling is authoritative for the actual research decisions: hypotheses, DOE designs, private holdouts, negative results, independent peer review, admission/promotion decisions, and active DLE worksets. Public workflows must never require private source checkout or private holdout data.

### Current research worksets

| Track | Private authority | Status |
| --- | --- | --- |
| Latent reasoning / extra-compute controls | [Campaign 0001](https://github.com/SemperSupra/latent-reasoning-private/issues/2) | Independent ongoing research |
| Original MeG Phi-2 PyTorch/CUDA reproduction | [Issue #5](https://github.com/SemperSupra/latent-reasoning-private/issues/5) | Source preflight; full reproduction gated on assets |
| MeG MLX framework engineering | [Issue #6](https://github.com/SemperSupra/latent-reasoning-private/issues/6) | Synthetic Phi-2 adapter build qualified; full algorithm port outstanding |
| Knowledge-capacity and Dory loop DOE | [Issue #7](https://github.com/SemperSupra/latent-reasoning-private/issues/7) | Research design; no scientific outcome claimed |
| Repository rename and layout | [Issue #8](https://github.com/SemperSupra/latent-reasoning-private/issues/8) | Proposed, awaiting owner rename |

## Existing implementation contracts

- `schemas/run-manifest.schema.json` and `schemas/reasoning-substrate.schema.json` describe present latent-reasoning runs and controls.
- `schemas/replay-spec.schema.json`, `tools/run_replay.py`, and `replays/` provide bounded, runner-neutral execution with separate scientific and environment receipts.
- `benchmarks/` tracks source and license admissibility.
- `environments/`, `profiles/`, `containers/`, and `.github/workflows/` define supported compute and smoke envelopes.
- `experiments/` currently includes legacy flat CLI modules. Existing paths stay stable until a compatibility-tested migration.

A future, versioned **knowledge-intervention descriptor** will extend the contracts without treating factual editing as `reasoning_substrate.class=latent`. See [the proposed repository structure](docs/architecture/model-mechanisms-layout.md).

## Portfolio boundaries

- **Model Artifact Foundry:** canonical model, adapter, and checkpoint identity, license/provenance, OCI distribution, immutable hydration and rollback.
- **Model Spelunker:** independent capability and performance qualification.
- **Agent Dispatch:** operational actors, workload routing, and admitted execution.
- **Model Mechanisms:** develops, reproduces, and tests scientific mechanisms, controls, and experimental implementations. It does **not** duplicate an artifact registry, generic scheduler, or the independent qualification plane.

## Way of working

**DLE means Durable, Legible, Executable.** Keep immutable source/model/dataset/experiment hashes and provider-native receipts; document assumptions, deviations and failures; maintain runnable verified commands and gated workflows. Support three audiences: humans, automation, and agentic actors.

Use independent parallel workcells for source provenance, environment compatibility, benchmark validity, and control analysis; use staged, cost-aware screening and held-out confirmation for scientific DOE. Public standard GitHub Actions are appropriate for CPU and low-memory macOS MLX build/test qualification. Original MeG GPU reproduction uses reviewed CUDA/Kaggle stages. The high-memory MLX scientific venue is a separately qualified local Mac, not the 7 GiB hosted runner.

No self-merges of draft research PRs, unreviewed Kaggle credential expansion, paid/private Actions minutes, hidden holdout publication, or promotion of smoke tests into scientific findings.
