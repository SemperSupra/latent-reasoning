# WP5h parity learning-rate sweep

WP5f/WP5g established that the frozen direct parity architecture can represent
the full lengths-1–8 ID universe, but convergence is strongly seed-dependent.

WP5h changes one scientific variable only: **AdamW learning rate**.

Candidate rates:

- 1e-3
- 3e-3 (the existing control)
- 1e-2

Frozen:

- d_model 64;
- 2 transformer layers;
- 4 attention heads;
- weight decay 0;
- exhaustive curriculum stages 2, 4, 8;
- batch size 128;
- maximum 1000 epochs per stage;
- three consecutive 100% checks;
- seeds 0,1,2.

## No-OOD tuning rule

OOD generation/evaluation is disabled for the entire sweep. A candidate is
qualified only if all selected seeds saturate the complete ID universe.

If more than one rate qualifies, selection uses final-stage and total ID
training epochs only. After the rate is frozen, a separate run may enable OOD.

This prevents length extrapolation results from becoming an implicit
hyperparameter-selection signal.
