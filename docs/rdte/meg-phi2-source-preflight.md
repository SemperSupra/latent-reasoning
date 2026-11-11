# MeG Phi-2 original-protocol reproduction: source preflight

**Status:** source-only preflight; not a trained or evaluated MeG result.  
**Scientific authority:** [latent-reasoning-private #5](https://github.com/SemperSupra/latent-reasoning-private/issues/5).  
**Cloud-execution dependency:** [public draft PR #45](https://github.com/SemperSupra/latent-reasoning/pull/45), which remains synthetic-hardware-only pending review.

## Exact upstream source

- Paper: ICLR 2026, [Massive Editing for Large Language Models Based on Dynamic Weight Generation](https://arxiv.org/abs/2512.14395).
- Official code: `RodeWayne/MeG-for-Knowledge-Editing`.
- Pinned source SHA: `9fa8ae10cf33c82dd2200057f70dfcb73388cd74`.
- Released Conda config: Python 3.10.14, PyTorch 2.4.1, Transformers 4.44.2, CUDA 12.1 packages.
- Target: `microsoft/phi-2` (immutable model/tokenizer revision not yet selected).
- Primary planned reproduction: ZsRE 1,024-edit condition; CounterFact follows. Original metrics are teacher-forced and autoregressive reliability, generality, and locality, plus authors' combined score.

## What this PR does

On public pull requests, run stdlib-only contract tests; fetch the immutable upstream commit; **read but never execute** its source; confirm key files and configs; hash the contents; publish a minimal source-only audit receipt. No GPU, Kaggle credential, model, dataset or author code is executed. The audit is successful if source integrity passes even when the expected external research assets are absent; the receipt explicitly says `BLOCKED_MISSING_RESEARCH_ASSETS`.

## Known scientific and runtime blockers

1. Upstream README instructs users to download data into `./data`, but the GitHub tree does not contain the derived `data/new_split_data`, `data/edit_data`, `data_paras`, `checkpoints_trained_bert`, or evaluation result checkpoint.
2. The `data/split_data.sh` process uses LLM-generated rephrases and explicitly warns that outputs may differ due to randomness.
3. Stage-1 Phi-2 ZsRE YAML has 30,000 epochs and GPU index 3. Stage-4 1,024 YAML has 1,000,000 epochs, GPU index 1 and some `...10000...` paths. Evaluation YAML references a missing trained checkpoint and example 10-case values. These must be resolved against author protocol, not silently treated as fidelity-preserving adjustments.
4. The public code tree has no obvious top-level license file; do not import its code into this repository or redistribute author's weights or derived datasets without checking rights.
5. Exact model/checkpoint hashes, original dataset split identities, actual Kaggle quota, successful GPU environment import, and provider-native run receipts remain unverified.

## Gated next work

- **Fidelity lane:** obtain raw and derived assets, hash original splits, pin model revision, verify licenses, record author deviations.
- **Environment lane:** build released Conda environment on compatible Linux/CUDA, check imports and a tiny inference smoke; no package upgrades except explicitly documented compatibility fixes.
- **Execution lane:** after public PR #45 receives independent review and explicit approval, extend its fixed allowlist with a separate, bounded, credential-gated MeG GPU workload. Prepare Kaggle-native immutable datasets for all model/source assets, disable Internet during the notebook if feasible, query account quota before admission, and refuse duplicate active jobs.
- **Measurement lane:** 10-item **engineering smoke only** -> authors' 1,024-edit protocol -> independent AG/TF scorer cross-check -> later edit counts.
- **Independent controls:** report all failure modes and compute budget; never call 10-item success a paper reproduction, and never conflate original MeG reproduction with new knowledge-capacity or Dory loop experiments.

## Reproduction terminologies

Computational reproduction = authors' same released method/code/configuration/data with traceable changes needed to execute. Independent replication = separately reimplement or independently test the mechanism. Research extension = new knowledge supply or recurrent Dory loops. Keep these claims separate.

## Operational boundaries

Public GitHub Actions standard runner only; no paid runner, secret, private corpus, broad webhook or Agent Dispatch. No automatic Kaggle retries while quota/admission unknown, and no unattended merge or secret exposure. The Kaggle controller remains the existing reviewed one-shot provider interface, not an additional scheduler.
