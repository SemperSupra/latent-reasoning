# Coconut pinned-upstream CPU preflight

Campaign 0001 pins the official Coconut implementation at:

`facebookresearch/coconut@27273cb8cca4bb763c041a63b036d0c3b7cbbb48`

This preflight clones that exact revision during CI and imports its `coconut.py`
directly. No upstream source is copied into this repository.

The smoke uses a tiny randomly initialized GPT-2 on CPU and checks that:

- upstream latent slots are replaced with preceding hidden states;
- the recurrent forward path completes;
- the resulting loss is finite;
- gradients propagate through the recurrence.

It also records a separate implementation constraint: the upstream reference
trainer in `run.py` initializes NCCL and calls `torch.cuda.set_device`, so the
published training launcher is GPU-bound as written.

That distinction matters: successful CPU execution of the **core mechanism**
does not imply that the upstream **reference trainer** is CPU compatible, and
neither result is a paper reproduction.
