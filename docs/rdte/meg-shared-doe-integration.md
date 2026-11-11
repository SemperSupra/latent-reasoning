# MeG and latent reasoning: shared DOE architecture and ownership

**Status:** design/contract reconciliation only; this document is not a scientific result, a new scheduler, or an authorization to launch Kaggle jobs.

**Private scientific worksets:** [MeG Phi-2 reference reconstruction #5](https://github.com/SemperSupra/latent-reasoning-private/issues/5), [MeG MLX engineering #6](https://github.com/SemperSupra/latent-reasoning-private/issues/6), [Latent Reasoning Campaign 0001 #2](https://github.com/SemperSupra/latent-reasoning-private/issues/2), [MeG knowledge/Dory DOE #7](https://github.com/SemperSupra/latent-reasoning-private/issues/7).

**Rationale:** one experiment/replay/evidence substrate across related research work, independent hypotheses and promotions. Do not create a second generic DOE framework or scheduler. Keep MeG's original CUDA reproduction distinct from MLX build qualification and future Dory interventions.

## Existing authoritative mechanisms to reuse

| Shared concern | Existing authority / implementation | MeG integration |
| --- | --- | --- |
| Provenance and model identity | `schemas/run-manifest.schema.json`, `docs/wp0-contract.md`; eventual Model Artifact Foundry | Reference exact upstream MeG revision, Phi-2 checkpoint, tokenizer, data split and generated components by immutable hashes; do not duplicate artifact registry |
| Experimental replay | `schemas/replay-spec.schema.json`, `tools/run_replay.py`, `docs/portable-replay.md` | Source-only CPU/MLX synthetic CLI experiments use existing bounded, no-shell replay pattern where supported; heavy Kaggle stages use a narrow provider adapter and its native receipts |
| Treatment and controls | `schemas/reasoning-substrate.schema.json`, `campaigns/0001` | Original MeG editing requires a **separate typed intervention field**; do not mislabel editing as latent-reasoning. Dory recurrence later can also reference reasoning-substrate |
| Environment | `environments/`, `profiles/`, substrate observer | Record CUDA/Kaggle, macOS GHA/MLX build, and BigMac local as distinct environments. Do not combine their benchmark scores without parity and valid controls |
| Benchmark licensing and splits | `benchmarks/sources.json`, `docs/benchmark-source-admissibility.md` | Audit ZsRE/CounterFact/MQuAKE and synthetic new-fact splits; class-E/unlicensed code remains external/methodological |
| Analysis and falsification | `docs/comparison-receipt-analysis.md` and independent Model Spelunker | Paired held-out questions, cluster/seed variance, compute/memory/cost; negative and inconclusive results are first-class |
| Execution | Public free-tier standard GHA, reviewed direct Kaggle launcher PR #45 | No Agent Dispatch admission dependency, new scheduler, private Actions minutes, silent paid resource, credentialed PR run, or bursty quota polling |

**Gaps to close rather than duplicate:** the existing `run-manifest` requires a reasoning-substrate class that does not express MeG knowledge editing; `replay-spec` supports only flat `experiments.<name>` modules, CPU/GPU classes, and 2-hour bounded processes. It does **not** provide native Kaggle notebook lifecycle or macOS/GPU placement semantics. Add a narrow, backward-compatible `intervention` sidecar or schema evolution and a provider-receipt mapper only after concrete use cases are accepted. Do not extend the general runner merely to match an imagined platform.

## Separated campaigns; common DOE envelope

| Campaign | Primary question | Design class | First gate |
| --- | --- | --- | --- |
| REF-MEG-PHI2 | Can the authors' original reported results be computationally reproduced using original source/configs/data? | Fixed original protocol; NOT a novel factorial DOE | All source/data/model/checkpoint revisions resolved; faithful AG/TF score reconstruction |
| PORT-MEG-MLX | Can we build, install, and execute the MeG-compatible framework on MLX with cross-backend numerical parity? | Engineering qualification and controlled method comparison; NOT a MeG efficacy experiment | Installable wheel, real PhiMLP synthetic semantics, then full pipeline stage parity |
| DOE-KNOWLEDGE-PHI2 | Can diffusion transfer genuinely new factual knowledge more efficiently than deterministic adapter methods or retrieval? | Stage-gated DOE: screen -> blocked controls -> held-out confirmation | Reference method and comparator implementations admitted |
| DOE-DORY-LOOPS | Does treatment identity, location and pass-specific scheduling improve access/integration of external knowledge? | 4-loop complete 16-mask screen, then selected treatment/loop-depth factorial, independent seeds | Dory baseline numerical parity and MeG intervention mechanism qualification |

**Separation rule:** Publishing a successful MLX wheel does not make MeG reproduced, and a MeG factual-editing reproduction does not prove a large-model knowledge-capacity advantage. Do not compare unpaired hardware or confound backend changes with treatment differences.

## Shared DOE identity and reporting fields

Every future experimental cell should carry these keys, as a typed sidecar pending a small, reviewed schema addition (not duplicate registries):

- `campaign_id`, `design_id`, `cell_id`, `rep_id`, `phase` (preflight / smoke / screen / confirm);
- `hypothesis_id`, `pre_registered_outcome`, `control_cell_id`, `comparison_group`;
- immutable `implementation_ref`, `backbone_ref`, `model_revision`, `tokenizer_revision`, `dataset_split_ref`, `controller_ref`, `environment_ref`, `harness_ref`;
- `factors`: e.g. controller type, task family, knowledge load, loop depth, loop mask, treatment strength, backend (only when method parity established);
- `seed` and matched `edit_id` / `subject_group` for paired designs; independent held-out lock identity;
- native `provider_job_id` + exact provider version + immutable source/receipt digests;
- `outcomes`: AG/TF reliability/generality/locality, multi-hop accuracy, unrelated-regression, latency, memory, training/inference cost;
- `state`: planned, admitted, executing, terminal, validated, promoted, blocked or failed, with machine-readable reason;
- a DLE evidence pointer and immutable run/analysis receipt hashes.

Do not put secret data, full holdout content, credentials, user-specific Kaggle IDs, or unlicensed weights in the public metadata sidecar. Separate private outcome authority from public-safe test manifests.

## Staged factor plan (no unaffordable full cross-product)

**A. Faithful reference:** run upstream Phi-2 ZsRE/CounterFact at the authors' published configurations, with a ten-item *engineering smoke* clearly distinguished from paper results. Reconstruction and new backend are not factors of this stage.

**B. MLX portability:** compare a fixed, tiny generated-neuron fixture against a PyTorch CPU reference; test shape/JSON compatibility, no-op and nonzero edit equivalence, gradient and diffusion sampler semantics after those modules are ported. Two standard macOS images are an engineering environment matrix, not scientific replicates.

**C. Knowledge-capacity pilot:** compare unchanged Phi-2, static adapter, deterministic controller, diffusion controller, and retrieval where comparable; use synthetic verifiably novel facts and frozen splits. Screen knowledge loads 100/500/1000 and direct/paraphrase/multi-hop tasks. Start with small blocked pilots and multiple independent seeds; escalate only if positive and negative controls behave as designed.

**D. Dory pass-placement:** freeze generator/knowledge payload/4-loop depth, evaluate all 16 activation masks on the *same* held-out queries and matched seeds, including zero and shuffled-conditioning controls; next compare loop-specific neuron identities and loop counts 1/2/4 in a gated confirmatory study. Normalize cumulative intervention strength and report all-loop cost.

**E. Confirmatory qualification:** pre-register primary outcomes and practical thresholds before evaluating untouched holdouts; pair tasks by ID; use edit/subject-level clustered uncertainty, multiple training seeds and independent evaluation. Do not use a 3-seed descriptive sample to claim significance. Model Spelunker qualifies promoted candidate behavior; Foundry owns derived artifact identity.

## Workflow and governance

- Public GHA Ubuntu: contract checks, DOE manifest generation, scoring, source provenance, lightweight CPU tests.
- Public GHA macOS: build/package/MLX adapter and sampler parity with tiny fixtures, no large Phi-2 checkpoint or scientific performance tests.
- Kaggle GPU: gated, queued-by-provider, original PyTorch/CUDA reproduction and later GPU science; only through the separately reviewed direct launcher with native quota and version checks, fixed source allowlist.
- BigMac high-memory MLX: local model-scale experiments after software artifact and numerical-parity acceptance. Native local provenance must be separate from hosted macOS.
- TPU: separate qualified port only if a JAX/TorchTPU hypothesis earns its cost; not part of the authors' reference reconstruction.

Each independent workcell can prepare and test concurrently. Within each scientific cell use immutable inputs, deterministic replay and idempotent publication. No broad automatic retries, no unreviewed default-branch merge or secret exposure; no new general scheduler or extra repository unless existing method-level scope actually proves insufficient.

## DLE closure rule

**Durable:** source/data/model hashes, configuration, native provider receipts and checkpoint identities.  
**Legible:** explicit hypotheses, provenance, controls, full negative results and caveats.  
**Executable:** runnable, pinned source, manifest validation and exact commands.

A PR is BUILD_COMPATIBLE only when its wheel/contract tests pass; it is not SCIENTIFICALLY_REPRODUCED. Reconcile to issues #5/#6 and Campaign 0001 only for findings that actually address their hypotheses. The next code-level integration decision is a **typed, backward-compatible intervention sidecar**, rather than bending an existing reasoning-only schema or creating a parallel experiment engine.
