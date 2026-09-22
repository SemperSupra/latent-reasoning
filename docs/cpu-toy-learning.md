# CPU toy learning experiment

`experiments/toy_modular_walk.py` is the first CPU-only **learning** experiment in Campaign 0001.

It uses a tiny randomly initialized transformer on a deterministic modular state-transition task. Every example has exact ground truth, and the OOD split increases sequential depth from six actions to ten.

The treatments are:

1. **direct** — one transformer pass;
2. **serial-control** — K extra passes using a learned pause/query embedding;
3. **latent** — K extra passes where the previous hidden state is fed back as the next continuous embedding.

Serial-control and latent use the same number of transformer forward calls. This creates an initial falsification surface for the claim that the *content* of the continuous state matters beyond merely receiving extra serial compute.

## Measurement

Receipts preserve:

- per-seed train, validation, and OOD accuracy;
- per-seed learning curves at configured epochs;
- paired latent-minus-direct and latent-minus-serial deltas;
- mean, population standard deviation, minimum, and maximum by treatment;
- forward-call count and wall time;
- a descriptive learnability status based on final training accuracy.

Poor performance is a valid experimental result and does not fail CI. CI fails only when the experiment or its evidence contract cannot execute correctly.

## Execution levels

Pull requests use a bounded two-seed/four-epoch run as an implementation gate.

Changes merged to `main` run a stronger five-seed/forty-epoch treatment set so the task's learnability is established before K-depth sweeps or representation claims are attempted.

## Claim boundary

This is a toy-system experiment. It can show that:

- the end-to-end latent training path works;
- a treatment can or cannot learn this controlled sequential task;
- latent and serial-control behavior can be compared under matched forward-call count;
- longer-depth transfer can be measured;
- an observed effect is or is not consistent across paired seeds.

It cannot establish that latent reasoning improves pretrained LLM capability.
