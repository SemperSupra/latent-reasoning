# Model Mechanisms repository layout and migration contract

**Decision owner:** [private DLE #8](https://github.com/SemperSupra/latent-reasoning-private/issues/8).  
**Status:** proposed target. This document does not rename repositories, move existing modules, or admit a new experimental workload.

## What the repository pair owns

This project owns research into mechanisms that extend model capabilities: original-paper reproduction and falsification, latent/recurrent computation, dynamic knowledge editing, diffusion and deterministic weight generators, task/loop conditioning, and portable experimental mechanisms.

The public repository owns **public-safe executable implementation and mechanics**. The private sibling owns **research authority, experimental designs, negative findings, private holdouts, and promotion decisions**. Model Artifact Foundry owns content-addressed model/checkpoint provenance and distribution; Model Spelunker owns independent qualification; Agent Dispatch owns operational workload routing. Maintain those boundaries.

## Public repository target

```text
model-mechanisms/
  README.md
  AGENTS.md                         future actor/automation instructions
  .github/workflows/                CI and bounded approved resource launch
  campaigns/
    0001/                           existing latent reasoning manifests, retained until migration
    latent-reasoning/                method-neutral controls and protocol IDs
    meg-reference/                   frozen original Phi-2/CUDA reproduction
    knowledge-augmentation/          new factual-capacity DOE
    dory-loop-control/               loop-conditioned treatment DOE
  methods/
    latent/                          Coconut, CODI, recurrent/refinement
    knowledge-editing/               MeG, generated neurons, static adapters
    controls/                        compute-matched, deterministic, retrieval
  adapters/
    target-models/                   Phi-2, Dory target interfaces
    backends/                        torch, mlx, jax after qualification
  experiments/                       executable typed CLIs; retain existing flat modules
  benchmarks/                        licensed/admitted sources and task generators
  schemas/                           existing run/replay and future intervention sidecar
  replays/                           bounded replay manifests
  environments/                      pinned runtime definitions
  profiles/                          compute admission requirements
  tools/                             shared validators, replay, receipt comparison
  tests/                             public source/CPU/macOS/qualified GPU oracles
  docs/                              architecture, methodology, RDTE, provenance
```

The directory entries not yet on `main` are **targets**, not implied existing files. Avoid creating empty placeholder directories. In particular, the present flat `experiments.<module>` path is a public replay API. Preserve it until a small reviewed evolution can provide new discovery/dispatch semantics with backwards compatibility.

## Private sibling target

```text
model-mechanisms-private/
  README.md
  AGENTS.md
  WORKPLAN.md                        campaign index, no duplicated issue authority
  campaigns/
    latent-reasoning/
    meg-phi2-reproduction/
    meg-mlx-engineering/
    knowledge-capacity/
    dory-loop-condition/
  doe/
    protocols/                       immutable design, outcomes, blocking, seeds, acceptance
    holdout-manifests/               private references only
    analyses/                        paired and clustered comparisons, null results
  evidence/                          provider-native receipt and artifact indexes
  decisions/                         red/blue findings, ADRs, promote/hold/kill
  handoffs/                          minimal agent workcell batons
```

The actual research worksets continue to live in GitHub Issues, with immutable commits and artifact links. These directories are supporting indexed documents, not a second task database. Do not commit private holdout contents or large model/checkpoint bytes.

## One method-neutral experiment contract

Keep common campaign/design/cell/rep identity, immutable model+tokenizer+dataset+environment+method hashes, explicit treatment/control IDs, matched seed/edit/subject grouping, provider-native version, scores and resource footprint, and independent evidence-validation state.

**Required extension:** a small typed knowledge-intervention sidecar (neuron/adaptor generation, layer, loop-mask, intensity, controller) that can coexist with `reasoning_substrate`. Knowledge-editing a fact is **not** automatically latent reasoning. Do not build a second scheduler, artifact registry or DOE database.

## Staged migration with stop gates

| Stage | Action | Acceptance |
| --- | --- | --- |
| 0 | Inventory current clone names, workflows, Action consumers, schema URLs, issue/PR refs, private/public boundaries, `kaggle-research` environment secrets and provider notebook identifiers | Reversible migration worksheet; no secrets dumped |
| 1 | Owner renames **existing** public/private repos in GitHub settings to `SemperSupra/model-mechanisms` and `-private` | IDs/history/issues/PRs remain accessible |
| 2 | Replace canonical old-slug URLs in docs, JSON Schema `$id` fields, manifests, local SSH remotes, GHA `uses:` calls, external consumers, Pages/OCI references where used | CI green with canonical new references; connected app reachability checked |
| 3 | Add versioned intervention descriptor and DOE manifest validation | Old latent run manifests still valid; controlled MeG cells validated |
| 4 | Introduce method/adapter subtrees **one vertical slice at a time** and retain existing replay wrapper paths | Existing CPU/MLX/CUDA tests still pass; no lost historical receipts |
| 5 | Only after checking consumers, archive the separate empty misspelled bootstrap repos `latent-resoning` / `latent-resoning-private` | Historical references retained, no new repo replacing old slug |

**GitHub rename caveat:** ordinary repo links and Git clone/fetch/push redirect, but GitHub Actions cannot follow an action `uses:` reference to the old name. Project-site URLs are not automatically redirected. Do not reuse the old slug or assume all outside integrations migrate. Verify and fix exact code references after the owner rename.

## Free-tier execution separation

- **GitHub-hosted Ubuntu CPU:** data/source preflight, schema/DOE validation, small deterministic tests and analysis.
- **GitHub-hosted Apple Silicon macOS:** build installable MLX wheels, stage contract tests, synthetic numerical/gradient parity; no actual memory-intensive Phi-2/MeG science.
- **Kaggle GPU:** original CUDA/PyTorch paper reproduction, then bounded DOE after data/environment/admission gates; use reviewed fixed-workload launcher with exact provider-native receipts.
- **Sovereign high-memory macOS:** full MLX method port, complete Phi-2 and MeG model tests and later science, after build/port qualification.
- **TPU/JAX:** conditional future backend only when it earns a hypothesis and has independent numerical-parity evidence.

DLE closure is **Durable, Legible, Executable**. No scientific claim from a CI green check alone, and no implicit promotion of an unfinished port.
