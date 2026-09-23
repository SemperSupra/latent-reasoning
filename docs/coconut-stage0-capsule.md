# Coconut GSM CoT stage-0 offline capsule

This capsule packages the next reference-reproduction step without executing the
25-epoch training on GitHub Actions.

## Embedded immutable inputs

The image contains and verifies:

- Coconut source revision
  `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`;
- GPT-2 snapshot
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- augmented GSM source revision
  `e06a32ee5e4cd117171daeb4755d2a97ece62761`;
- train Git-LFS object SHA-256
  `0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c`;
- exact Coconut-processed train/valid/test JSON digests;
- the reviewed two-rank stage-0 config and adaptation plan;
- the digest-pinned PyTorch 2.5.1 / CUDA 12.4 / cuDNN 9 base image.

The portable pinned-LFS hydrator is used during image construction, so the data
path is the same method already qualified in GHA and intended for sovereign
replay.

## Inspection vs execution

The image entrypoint is `tools/run_coconut_stage0.py`.

`--inspect-only` is CPU-safe and verifies the entire capsule identity,
configuration, effective global batch, processed data, and offline environment.

Without `--inspect-only`, the entrypoint refuses execution unless at least two
CUDA devices are visible. It then launches exactly:

`torchrun --nproc_per_node=2 /opt/coconut/run.py /workspace/configs/coconut-gsm-cot-stage0-2gpu.yaml`

without a shell.

After a successful run it records elapsed time and hashes every saved
`checkpoint_*` artifact. A run that exits successfully but produces no
checkpoint is treated as a failed stage-0 apply.

## GHA role

GHA:

1. tests the capsule contract;
2. proves the reviewed placement profile rejects the CPU runner;
3. builds the exact offline image;
4. runs CPU-safe identity inspection;
5. proves no training execution receipt exists.

It does not run stage-0 training.

## Sovereign next step

The next work package adds the fail-closed local apply command around this image:
observe → check two-GPU compatibility → build exact source revision → execute
with `docker run --gpus all` → verify checkpoint and execution receipts.

The first local run will also measure peak resource use so the conservative
two-GPU placement floor can be refined from evidence.
