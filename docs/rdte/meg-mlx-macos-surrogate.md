# MeG-adjacent MLX surrogate on standard public macOS Actions

**Research workset:** [private MeG Phi-2 source reproduction #5](https://github.com/SemperSupra/latent-reasoning-private/issues/5).  
**Upstream:** [RodeWayne/MeG-for-Knowledge-Editing](https://github.com/RodeWayne/MeG-for-Knowledge-Editing) commit `9fa8ae10cf33c82dd2200057f70dfcb73388cd74`.  
**Classification:** MLX portability/method validation only. **Not** an execution of upstream MeG, not an independent replication, and not a Phi-2 scientific outcome.

## Why the split exists

Earlier macOS runner census and MLX qualification are authoritative in:
- [agent-dispatch-private #223](https://github.com/SemperSupra/agent-dispatch-private/issues/223): runner census;
- [#285](https://github.com/SemperSupra/agent-dispatch-private/issues/285): Metal/MLX procedural transfer limits;
- [#286](https://github.com/SemperSupra/agent-dispatch-private/issues/286): first pinned MLX-LM local inference anchor.

The hosted ARM64 `macos-26` and `xcode-27` standard runners are approximately 7-GB RAM Apple M1 guests with constrained available memory and virtual Metal. They cannot be assumed to host full-precision Phi-2 plus MeG. Larger macOS runners are out of scope.

The **original experiment** continues to use the authors' frozen Python/PyTorch/CUDA environment on Kaggle or an equivalent compatible device. An MLX implementation is a later port and an independent experimental variable; it cannot be substituted silently while claiming faithful reproduction.

## Minimal new contract

`experiments/meg_mlx/surrogate_gpu.py` executes on explicitly selected MLX GPU, with tiny fixed tensors:
- baseline two-projection MLP surrogate;
- zero-intervention bit-for-bit equality;
- generated-style low-rank/neuron intervention against independent CPU NumPy oracle;
- nonzero finite MLX autodiff gradient, numerical tolerance, and digest;
- bounded JSON receipt with architecture, Python/MLX versions, selected device and assertion results.

This is original minimal test code and does not import or redistribute the authors' source. No checkpoint downloads, Hugging Face tokens, Kaggle token, network APIs at runtime, model data, long prompts, or training tasks.

## CI admission and qualification

Public PRs use standard public `ubuntu-24.04` for syntax, then `macos-26` and `xcode-27` ARM64 sequentially for pinned `mlx==0.32.4`, `numpy==2.1.1`, selected MLX GPU compute and independent oracle. Each macOS job has a 12-minute timeout and small memory footprint. Failure is a negative environment/runtime observation, not permission to auto-upgrade or change authors' MeG training config.

Next separately gated steps: test full Phi-2 artifact size and memory admission **without downloading the model**, then a suitably quantized MLX Phi-2 **portability** baseline if a future independent protocol warrants it; original BF16/FP16 MeG remains GPU/CUDA protocol. Design any substantive MeG dynamic-neuron port only after original-paper reproduction; never claim this tiny surrogate validates factual editing.

## Data boundary
Allowlist only public original synthetic tensors and sanitized receipts. Do not import private holdouts into PR runs. Keep this lane independent of draft Kaggle PR #45 and upstream source audit PR #47.
