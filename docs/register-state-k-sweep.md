# Register-state recurrent-depth sweep

The register-state K=2 fixed comparison passed all causal gates and produced a
consistent positive latent-minus-serial OOD delta across three seeds.

This follow-up maps the bounded recurrent-depth envelope at:

`K = {1, 2, 4, 8}`.

The frozen direct baseline is not retrained for every K. The causal comparison is
serial-control versus latent, and both are trained with:

- identical shared initialization;
- LR 0.001;
- the same complete-universe curriculum;
- the same three seeds and data ordering;
- exactly K+1 transformer forward calls per batch;
- the same OOD instances for each seed.

Each K has its own admission gate. If either recurrent treatment fails complete
ID saturation for any selected seed, that K is marked invalid and its partial
OOD results are not used as a causal comparison.

K=2 is intentionally included again as an internal deterministic replication
point for the already observed fixed-K result.
