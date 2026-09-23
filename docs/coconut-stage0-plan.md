# Coconut GSM CoT stage-0 two-GPU plan

This work package prepares the paper/reference **stage-0 CoT training** for
sovereign execution. GHA validates the plan and placement mechanics; it does not
attempt the 25-epoch GPU training.

## Reference

Pinned upstream:

- `facebookresearch/coconut@27273cb8cca4bb763c041a63b036d0c3b7cbbb48`;
- config: `args/gsm_cot.yaml`;
- backbone: `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- augmented GSM source and exact LFS object already qualified in WP4c.

The paper repository labels the config as needing four GPUs. Its reference
training batch is 32 examples per rank with accumulation 1:

`4 ranks × 32 × 1 = effective global batch 128`.

## Explicit two-rank adaptation

The first sovereign plan changes only operational/path fields plus gradient
accumulation:

- world size: 4 -> 2;
- per-rank batch: remains 32;
- gradient accumulation: 1 -> 2;
- effective global batch: remains **128**;
- model path becomes the exact offline GPT-2 snapshot in the capsule;
- train/validation paths become the exact preprocessed data paths in the capsule;
- save path becomes the mounted output volume;
- run name identifies the adapted stage-0 execution.

All other scientific config values must remain byte-semantically equivalent to
the pinned reference. The validator fails if learning rate, epochs, weight
decay, seed, CoT mode, or another unreviewed field changes.

W&B is placed in offline mode through the execution environment. That changes
logging transport, not the optimizer or model computation.

## Placement

The initial profile requires:

- at least two NVIDIA GPUs;
- at least 20,000 MiB per GPU;
- Docker with NVIDIA container runtime;
- at least 16 GiB host memory and 40 GiB free disk.

This is a conservative **admission floor for the first run**, not a measured
final resource requirement. GPU-smoke and stage-0 receipts will refine it.

Public CPU GHA must fail this placement gate. That failure is the expected
method-development result.

## Next step

After this plan is qualified, package the exact source/model/data/config into
the offline stage-0 container and exercise build/identity checks on GHA. Actual
training remains a sovereign/local apply.
