# Register-state fixed-K representation comparison

This is the second-task replication of the Campaign 0001 fixed-K reasoning
representation experiment.

The register-state mini-regime has already passed complete ID saturation across
all three selected seeds. WP6b freezes that task, optimizer, LR, architecture,
and OOD instances.

Treatments:

- direct: one transformer pass;
- serial-control K=2: learned constant recurrent slots;
- latent K=2: recurrent slots populated by preceding hidden state.

Serial-control and latent both use exactly three transformer forward calls per
batch and share the same parameterization, initialization, data order, and
training criterion. Direct is retained as lower-compute context.

The purpose is replication across a state-transition task whose target is a
three-way register value rather than binary parity.
