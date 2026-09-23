# Coconut reference GPU smoke capsule

This package is the first GPU-bound replay prepared through the GHA
method-development path.

## Immutable inputs

- Coconut source: `facebookresearch/coconut@27273cb8cca4bb763c041a63b036d0c3b7cbbb48`.
- Reference backbone: `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Base image:
  `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.

The image downloads the exact GPT-2 snapshot at build time and then enables
offline Hugging Face/Transformers mode. The exact Coconut source is cloned,
verified, and stored without its Git metadata.

## Bounded experiment

`experiments.coconut_gpu_smoke` performs one forward/backward/optimizer step
using:

- the exact upstream Coconut core;
- the exact reference GPT-2 weights;
- two latent slots by default;
- CUDA only, with no CPU fallback.

It records device identity, model size, loss/gradient/update checks, elapsed
time, and peak CUDA allocated/reserved memory.

This is resource/mechanism evidence. It is not a GSM8K or paper reproduction.

## Placement

GHA performs three things:

1. validate the replay/profile/source contracts;
2. prove the public runner fails the GPU compatibility gate;
3. build and inspect the complete offline capsule.

GHA **does not execute** the GPU scientific replay.

A sovereign/local actor applies it only after:

```
python tools/observe_substrate.py > substrate.json
python tools/check_substrate_compatibility.py \
  substrate.json \
  profiles/coconut-reference-gpu-smoke.json \
  --output compatibility.json
```

and only when the decision is `eligible-for-bounded-apply`.

The eventual apply command uses the same replay spec and image entrypoint with
Docker/NVIDIA GPU exposure and writes the normal scientific receipt + substrate
sidecar. The measured smoke then informs the later full-reproduction resource
profile; the current 20 GB GPU-memory threshold is intentionally a conservative
admission threshold, not a full-training requirement claim.
