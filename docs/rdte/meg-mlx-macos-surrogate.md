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

## Owner clarification: MLX build first, local experiments later

The macOS public runners are a **prebuild / compatibility** lane only, not an experimental execution venue. The project builds `meg-mlx-bridge==0.1.0` from `experiments/meg_mlx/framework` as a pure Python wheel, installs that wheel in a fresh runner context, and tests actual `mlx_lm.models.phi.PhiMLP` interoperability using 32-dimensional synthetic fixtures. Both `macos-26` and `xcode-27` use bounded memory and must explicitly execute MLX GPU.

The adapter supports:
- MeG's Phi-2 one-neuron `params_0.json` schema (last `fc1` row, last bias, last `fc2` column), with strict shape validation and JSON roundtrip;
- reversible, frozen-backbone last-neuron **replacement** via an MLX-LM MLP wrapper (not addition of an extra neuron);
- a zero-strength exact no-op; direct full-weight replacement comparison; gradient flow through generated neuron weights;
- the deterministic controller protocol for a later diffusion-weight generator.

**The original MeG diffusion controller, familiarity network, text encoder and training pipeline are not yet ported.** A passing wheel build does *not* qualify those components, nor does it reproduce any author score. The scientific fidelity spine remains the original upstream PyTorch/CUDA implementation.

### Local BigMac consumption

After accepting a reviewed PR/build receipt, download the `meg-mlx-wheel-macos-26` or `meg-mlx-wheel-xcode-27` artifact from its exact GitHub Actions run. Verify the wheel SHA-256 against `build-receipt.json`. On the local Mac:

```bash
python3.11 -m venv ~/Projects/meg-mlx-venv
source ~/Projects/meg-mlx-venv/bin/activate
python -m pip install 'mlx==0.32.4' 'mlx-lm==0.31.3' 'numpy==2.1.1'
python -m pip install --no-deps ./meg_mlx_bridge-0.1.0-py3-none-any.whl
python -c 'import meg_mlx_bridge; print("MLX bridge imported")'
```

The filename above is a *wheel build output convention*, not a claim that a local artifact has been installed. The future local Phi-2 checkpoint test must pin its exact source, tokenizer and hash and use the Mac's distinct environment identity; inference/editing results may not inherit from GitHub-hosted runner evidence.

### Next engineering stage, outside the present wheel gate

Port and qualify the full **MeG diffusion-controller interface** and training/inference lifecycle to MLX against the frozen CUDA reference, using a cross-backend weight-interchange specification and numerical parity fixtures. Only after this later phase is complete should high-memory local macOS execute substantive Phi-2+MeG experiments. For now the wheel contains the verified bridge, not a full MeG reimplementation.
