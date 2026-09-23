# WP5i frozen parity OOD baseline

WP5h selected learning rate **0.001** using only complete-ID saturation and
training efficiency. OOD examples were not generated during that selection.

WP5i freezes that choice and re-enables the untouched length-OOD evaluation.

## Frozen configuration

- treatment: direct;
- model width: 64;
- transformer layers: 2;
- attention heads: 4;
- optimizer: AdamW;
- weight decay: 0;
- learning rate: **0.001**;
- exhaustive curriculum: lengths <=2, <=4, <=8;
- three consecutive 100% ID checks required;
- seeds: 0, 1, 2;
- maximum 1000 epochs per curriculum stage;
- OOD lengths: 9-16;
- deterministic 256 fresh examples per OOD length and seed.

## Admission rule

The main-branch run fails if any selected seed does not first reach the complete
100% ID saturation gate. This is deliberate: the benchmark is not admitted for
representation comparison unless the frozen direct baseline is demonstrably
learned across all selected seeds.

No hyperparameter may be changed in response to the resulting OOD scores. The
next legal comparison uses this frozen regime for direct, serial-control, and
latent treatments.
