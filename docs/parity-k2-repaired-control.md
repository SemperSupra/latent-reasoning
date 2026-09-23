# Repaired parity K=2 representation comparison

The first full K=2 parity comparison was invalid because serial-control seed 0
failed the complete ID saturation gate at LR 0.001.

The repair was selected **without OOD access**:

- serial-control and latent were jointly swept at K=2;
- candidate LRs were judged only by complete-ID saturation and curriculum cost;
- LR 0.002 was the most efficient candidate that saturated both recurrent
  treatments across all three seeds.

This rerun freezes:

- direct contextual baseline LR: 0.001;
- serial-control LR: 0.002;
- latent LR: 0.002;
- K=2 for both recurrent treatments;
- identical shared initialization, architecture, data order, and ID/OOD sets;
- three seeds;
- complete-ID saturation before OOD.

The causal comparator is **latent versus serial-control**. Direct remains useful
lower-compute context but has separately frozen training dynamics.

The failed original K=2 run remains preserved as negative/control-development
evidence; this experiment does not overwrite it.
