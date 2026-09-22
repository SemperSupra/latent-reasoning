# WP5f parity saturation

This experiment qualifies the first formal-state benchmark regime before any latent-vs-control comparison.

## ID universe

For parity, every bit string of lengths 1 through the current curriculum stage is included. The final ID stage is lengths 1–8, containing 510 unique strings.

Training progresses through cumulative exhaustive stages:

1. lengths 1–2;
2. lengths 1–4;
3. lengths 1–8.

A stage advances only after repeated checks at exactly 100% accuracy on its complete universe.

## OOD rule

Length-OOD evaluation is not run for a seed unless the final ID stage saturates. If the final stage fails, the receipt contains no OOD accuracy for that seed.

The initial OOD range is lengths 9–16 with deterministic fresh samples per length.

## Why this is cleaner than modular-walk-v0

The earlier modular-walk experiment mixed benchmark learnability with representation comparison. This regime makes learnability an explicit prerequisite.

Parity also has:

- a mathematically exact oracle;
- an enumerable ID universe;
- exact intermediate prefix state;
- a clean length extrapolation axis;
- no third-party code or dataset dependency.

Once all selected seeds repeatedly reach 100% ID, the same frozen regime can be used for compute-matched direct / serial-control / latent comparisons.


## WP5g controlled follow-up

The first three-seed main run held the final curriculum stage to 300 epochs.
Seed 0 saturated the complete lengths-1–8 ID universe, while seeds 1 and 2 did
not. Because the same architecture demonstrated that saturation is reachable,
WP5g changes **only** the maximum training budget to 1000 epochs per stage.

Frozen across WP5f -> WP5g:

- model width, depth, and attention heads;
- AdamW optimizer and learning rate;
- exhaustive curriculum stages 2, 4, 8;
- batch size;
- three consecutive 100% checks required for saturation;
- three seeds;
- OOD lengths and sampling rule.

If the remaining seeds still fail to saturate, the next experiment must change a
training/mechanism variable explicitly rather than silently increasing multiple
resources.
