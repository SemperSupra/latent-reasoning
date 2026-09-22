# Pinned upstream reproduction sources

Campaign 0001 pins upstream method implementations by exact commit before any reproduction work.

## Coconut

- Repository: `facebookresearch/coconut`
- Exact revision: `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`
- Observed license: MIT
- Reference Python: 3.12
- Reference PyTorch: 2.5.1
- Reference Transformers: 4.46.2
- Paper reproduction backbone/config: `openai-community/gpt2`, using the upstream GSM CoT and Coconut YAMLs.

The upstream README states that its published commands assume four A100 80GB GPUs and explicitly allows adapting per-GPU batch size, gradient accumulation, and process count to available resources.

### Reproduction rule

First reproduce the paper-native GPT-2 path with the smallest necessary resource adaptation. Do not simultaneously port the method to a newer backbone. A later crossover can test the same method on a modern small backbone after the implementation is known-good.

## CODI

- Repository: `zhenyi4/codi`
- Exact revision: `2c2314662c63e9f482ebc46614ffe9af17a241e5`
- Reference Python: 3.12
- Reference PyTorch: 2.7.1
- Reference Transformers: 4.52.4
- Reference PEFT: 0.15.2
- Released upstream examples include GPT-2 and Llama-3.2-1B-Instruct.

### License gate

No explicit repository license file or README license statement was observed at the pinned revision. Therefore:

- do not copy or vendor CODI source into this public repository;
- do not promote a derivative of its source code as ours;
- it may be used as an external behavioral/reproduction oracle where appropriate;
- released model artifacts require their own license/provenance review before Foundry admission.

This is a provenance gate, not a technical failure of the method.

## Sequence

1. Coconut exact-source reproduction on its reference GPT-2 setup.
2. Establish direct / textual-CoT / serial-compute accounting around that reproduction.
3. CODI oracle reproduction if its external dependencies and artifact licensing permit.
4. Only then adapt one validated latent method to the campaign's modern small-backbone crossover.
