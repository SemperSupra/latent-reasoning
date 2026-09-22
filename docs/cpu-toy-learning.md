# CPU toy learning experiment

`experiments/toy_modular_walk.py` is the first CPU-only **learning** experiment in Campaign 0001.

It uses a tiny randomly initialized transformer on a deterministic modular state-transition task. Every example has exact ground truth, and the OOD split increases sequential depth from six actions to ten.

The treatments are:

1. **direct** — one transformer pass;
2. **serial-control** — K extra passes using a learned pause/query embedding;
3. **latent** — K extra passes where the previous hidden state is fed back as the next continuous embedding.

Serial-control and latent use the same number of transformer forward calls. This creates an initial falsification surface for the claim that the *content* of the continuous state matters beyond merely receiving extra serial compute.

## Claim boundary

This is a toy-system experiment. It can show that:

- the end-to-end latent training path works;
- a treatment can or cannot learn this controlled sequential task;
- latent and serial-control behavior can be compared under matched forward-call count;
- longer-depth transfer can be measured.

It cannot establish that latent reasoning improves pretrained LLM capability.

The PR gate uses only a bounded two-seed/four-epoch run. Longer multi-seed runs are manually dispatched after the implementation gate passes.
