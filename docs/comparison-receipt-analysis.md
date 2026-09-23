# Representation-comparison receipt analysis

Fixed-K experiment receipts are summarized by
`tools/summarize_representation_comparison.py`.

The analyzer first checks:

- the receipt is the expected evidence class;
- serial-control and latent forward-call counts match;
- required recurrent treatments passed their ID saturation gates;
- the originating experiment marked the comparison valid.

Only then does it summarize paired latent-minus-serial and latent-minus-direct
OOD deltas.

For fewer than five paired seeds, the analyzer deliberately labels the evidence
**descriptive-only**. It does not manufacture a p-value or confidence claim from
a three-seed campaign tranche.

A task-specific directional result is not promoted to a general latent-reasoning
claim. Cross-task replication and the later K sweep remain separate evidence
requirements.
