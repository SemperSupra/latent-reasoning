# Direct Kaggle GPU/TPU experiments — Campaign 0001

Status: draft PR / public CI only pending review and merge.
Authority: https://github.com/SemperSupra/latent-reasoning-private/issues/2
This is a one-shot GHA launcher. It does not use Agent Dispatch, sidecar, scheduler, or webhook.

## Execution boundary

- Public pull requests run offline stdlib tests and construct both source-hashed plans on free public GitHub Actions. No Kaggle credentials or provider launch are used.
- The separately reviewed main-branch workflow_dispatch supports typed plan / submit / observe.
- Credentialed runs require a repository environment named kaggle-research, protected for main and configured with KAGGLE_API_TOKEN. Secrets in another repository are not inherited.
- Every job selects one of two fixed public synthetic sources. No user-provided script, shell, repository URL, model weights, private data, or arbitrary Python source.
- Kaggle kernel metadata always specifies private notebook, Internet disabled, empty external sources, fixed accelerator and short bounded timeout.
- Submit checks authenticated account quota, scans existing targeted notebook identifiers and active known experiments, and refuses duplicates. It exits only after a provider-native version is returned. It does not claim a completed scientific result.
- Observe requires an exact provider-native version, refuses version drift and independently validates the terminal result and arithmetic/training metrics.
- The Kaggle script now includes the SHA256 of its **actual executed run.py bytes** in its native scientific receipt. The observer compares it with the exact reviewed source SHA; missing, modified or injected source identity is a hard failure, even if the numerical metrics otherwise pass. TPU cosine scores must also be finite, within [-1,1], and not boolean.
- This control is an additive defense to provider-native version verification, not proof that Kaggle itself is an independently trusted execution environment. Previously submitted notebooks are not automatically backfilled with the new evidence field.
- Only sanitized allowlisted control receipts may be uploaded to the public GitHub run. No raw provider logs, secrets or Kaggle datasets.

## Workload GPU: gpu-cuda-v1

Requests NvidiaTeslaT4. Requires actual two CUDA T4s and verifies on-device numerical operations independently on each GPU. Device enumeration by itself is not acceptance. Hardware/smoke only, no model efficacy.

## Workload TPU: tpu-bridge-v1

Requests TpuV5E8. Requires JAX backend TPU, exactly eight visible TPU devices, explicit pmap arithmetic across the eight and a small fixed-seed synthetic representation-mapping training experiment with a shuffled-target control. Bridge training uses replicated weights and eight JAX data shards with pmean gradient synchronization across all eight chips. Passing is a hardware + synthetic method sanity result, NOT a latent reasoning or cross-model capability claim.

## Follow-on TPU RDTE after observed hardware PASS

- Eight-way JAX data-parallel *scaling*, compare compile warmup and steady-state timings, XLA sharding, gradient accumulation, batch scaling and memory use.
- Recurrent K=0,1,2,4,8 and adaptive-depth sweeps against matched serial compute; per-task/seed OOD tests and causal interventions.
- Adapter/codec comparisons: no intervention, random projection, learned linear projection, nonlinear bridge and independent held-out evaluator.
- Coconut reference stage-0 and CODI only after their separate reproduction and licensing gates; CUDA PyTorch scripts are not assumed TPU-compatible without verified JAX or Torch-XLA ports.
- Immutable source/checkpoint hashes, exact device receipts and independent Model Spelunker qualification.

## Budget, secrets, prior provider history

- $0 incremental compute: public GitHub-hosted Ubuntu Actions and available Kaggle free allocation only. Actual Kaggle account quota is not assumed; provider query required.
- Never put KAGGLE_API_TOKEN in PRs, issues, source, logs or workflow inputs. Protected environment approval is a repository configuration task.
- Existing Kaggle submission ad-rdte-gpu-t4x2-20261014-080203 version 1 remains separate and unobserved. This workflow never resubmits or overwrites it.
- One active experiment at a time; no automatic retries. Public CPU CI passing is not native accelerator execution proof.

## Operator procedure after PR acceptance

1. Review source/diff, merge only with owner approval, configure kaggle-research environment for protected main and KAGGLE_API_TOKEN secret.
2. Run Actions > latent-kaggle-direct > main > action=plan, workload=tpu-bridge-v1.
3. Run action=submit with the same workload, version=0, if account-level authorization and current free quota permit.
4. Inspect returned allowlisted control receipt for the native version.
5. When the job is terminal, invoke action=observe with that exact version, retain the verified result SHA and reconcile to the private DLE.
6. Repeat separately for gpu-cuda-v1. Never claim a Kaggle pass before actual provider-native evidence.

Any unknown credential, quota, provider state, native version or hardware fails closed.
