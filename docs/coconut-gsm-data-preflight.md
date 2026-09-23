# Coconut GSM data-pipeline preflight

The official Coconut GSM8K recipe does not consume the original GSM8K JSON
directly. Its pinned preprocessing script downloads augmented train/validation/
test text files from:

`da03/Internalize_CoT_Step_by_Step@e06a32ee5e4cd117171daeb4755d2a97ece62761`.

That repository carries an MIT license. GSM8K itself is also MIT-licensed.

## What GHA validates

The CPU preflight:

1. checks out exact Coconut and Internalize-CoT revisions;
2. hydrates the exact augmented GSM text files using the commit-pinned media URLs
   used by Coconut;
3. executes Coconut's exact `preprocessing/gsm_icot.py`;
4. independently checks every processed record against the raw delimiter format
   and records source/derived SHA-256 identities;
5. loads bounded train/validation samples through Coconut's exact
   `get_dataset`;
6. uses the exact GPT-2 tokenizer revision already pinned for Campaign 0001;
7. constructs scheduled-stage latent train/generation datasets;
8. runs Coconut's exact `MyCollator` and verifies shape, masking, latent-token,
   attention-mask, and position-id invariants.

No GSM corpus content is committed to this repository.

## Boundary

This is data-path and experiment-method evidence. It removes preprocessing and
collation surprises before GPU execution; it does not train the CoT stage-0
model or reproduce GSM8K performance.

The hashes in the receipt provide the artifact identities that a later
sovereign reproduction should reconcile before training.
