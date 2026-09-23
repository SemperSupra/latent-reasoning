# WP6a parity fixed-K representation comparison

Parity has now passed the frozen direct-baseline admission gate:

- complete lengths-1..8 ID universe;
- three seeds;
- learning rate 0.001 selected without OOD access;
- repeated 100% ID accuracy for all three seeds;
- untouched length-9..16 OOD baseline recorded.

WP6a is the first representation comparison.

## Treatments

- **direct** — one transformer forward pass; retained as the frozen lower-compute reference.
- **serial-control K=2** — two recurrent slots filled with a learned constant pause embedding, then a final readout pass.
- **latent K=2** — two recurrent slots filled by the model's own preceding hidden state, then a final readout pass.

Serial-control and latent therefore both use exactly **K+1 = 3 transformer forward calls per batch**.

All three treatments use:

- identical architecture and parameter count;
- identical initialization for a given seed;
- identical exhaustive curriculum and data order;
- AdamW with frozen LR 0.001;
- identical saturation criterion;
- identical OOD instances per seed.

## Interpretation gate

OOD results are emitted only for a treatment/seed that first saturates the complete ID universe.

The primary causal comparison is **latent versus serial-control** because their recurrent compute is matched. Direct is useful context but is intentionally lower-compute.

This is still a single-task, fixed-K result. It does not justify a general latent-reasoning claim. If the comparison is valid, WP6 then expands K and task families.
