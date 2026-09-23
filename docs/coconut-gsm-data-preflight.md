# Coconut GSM data-pipeline preflight

The official Coconut GSM8K recipe does not consume the original GSM8K JSON
directly. Its pinned preprocessing script consumes augmented train/validation/
test text files from:

`da03/Internalize_CoT_Step_by_Step@e06a32ee5e4cd117171daeb4755d2a97ece62761`.

That repository carries an MIT license. GSM8K itself is also MIT-licensed.

## Git LFS provenance

The pinned `data/gsm8k/train.txt` is a Git LFS pointer:

- pointer Git blob: `853a81e17863e3e42b90a856a0a1535294f8cc7e`;
- LFS object SHA-256:
  `0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c`;
- hydrated size: `87,805,358` bytes.

A historical-source wrinkle matters: the pinned data commit predates the
upstream repository's `.gitattributes` rule for this path. The object is
already an LFS pointer, but a checkout of that exact commit does not by itself
know to smudge it.

The qualified method therefore:

1. clones without an initial checkout;
2. installs Git LFS locally;
3. writes **only**
   `data/gsm8k/train.txt filter=lfs diff=lfs merge=lfs -text`
   into the clone's local `.git/info/attributes`;
4. sparse-checks out only the license and GSM train/valid/test files at the
   exact pinned revision;
5. verifies the historical pointer OID and size;
6. verifies the hydrated train file's exact byte size and SHA-256.

This local attribute changes no upstream content and does not rely on a later
upstream branch merely to activate LFS transport.

## What GHA validates

The CPU preflight:

1. checks out exact Coconut and Internalize-CoT revisions;
2. hydrates and verifies the exact augmented GSM Git LFS object;
3. executes Coconut's exact `preprocessing/gsm_icot.py`;
4. independently checks every processed record against the pinned parser
   semantics and records source/derived SHA-256 identities;
5. loads bounded train/validation samples through Coconut's exact
   `get_dataset`;
6. uses the exact GPT-2 tokenizer revision already pinned for Campaign 0001;
7. constructs scheduled-stage latent train/generation datasets;
8. runs Coconut's exact `MyCollator` and verifies shape, masking, latent-token,
   attention-mask, and position-id invariants.

The train corpus contains 385,620 physical records. Python's broader
`str.splitlines()` sees four additional control-character boundaries, so the
validator intentionally uses the upstream preprocessor's physical
`readlines()` record semantics rather than inventing four records.

Heavy PyTorch/Transformers dependencies are installed only after source
checkout, LFS hydration, and preprocessing have passed. This keeps future
transport/provenance failures cheap on GHA.

No GSM corpus content is committed to this repository.

## Boundary

This is data-path and experiment-method evidence. It removes preprocessing,
Git-LFS hydration, tokenization, staging, and collation surprises before GPU
execution; it does not train the CoT stage-0 model or reproduce GSM8K
performance.

The hashes in the receipt provide artifact identities that a later sovereign
reproduction should reconcile before training.
