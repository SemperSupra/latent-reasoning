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
