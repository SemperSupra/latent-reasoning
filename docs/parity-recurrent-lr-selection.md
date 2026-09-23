# ID-only common recurrent learning-rate selection

The first full parity K=2 run was not a valid latent-vs-serial comparison because
serial-control seed 0 failed the complete ID saturation gate at the frozen direct
learning rate of 0.001.

This follow-up does **not** inspect OOD behavior.

## Scope

The sweep trains only:

- serial-control K=2;
- latent K=2.

Candidate learning rates are shared by both treatments. A candidate is eligible
only if both treatments saturate the complete lengths-1..8 ID universe for all
three selected seeds.

## Selection rule

Among jointly qualified candidates:

1. minimize mean total curriculum epochs across all six treatment/seed runs;
2. break an exact tie by selecting the lower learning rate.

No OOD examples are generated. No candidate is selected because it produces a
better extrapolation score.

## Why the direct LR may differ

The direct model remains the frozen lower-compute reference at LR 0.001.

The causal representation comparison is serial-control versus latent, so those
two recurrent treatments must share the same selected training dynamics. Direct
is contextual and must not be interpreted as a training-hyperparameter-matched
causal comparator after this step.

If no common recurrent LR produces joint saturation, the current serial-control
mechanism is not yet a valid causal control and the K sweep remains blocked.
