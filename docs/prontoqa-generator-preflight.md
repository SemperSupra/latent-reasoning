# PrOntoQA pinned-generator preflight

PrOntoQA is admitted in the benchmark source registry as an Apache-2.0 source.

This preflight deliberately uses only the narrow upstream generation surface:

- exact source revision: `0a6412b6fddf46324a1cb96e066dd7b3d89b87d6`;
- upstream `run_experiment.py`;
- built-in `--model-name json` mode;
- no external LLM/API;
- no copied upstream source in this repository.

The smoke generates small fictional-ontology datasets at one and two reasoning hops and verifies that the resulting JSON artifacts are structurally usable.

## Admission boundary

Passing this preflight means the generator can be wrapped as a pinned external benchmark dependency.

It does **not** mean:

- every PrOntoQA configuration is qualified;
- the original model-evaluation stack is adopted;
- public smoke examples are suitable as private holdouts;
- downstream natural-language rendering or proof parsing is correct for our use.

Those are separate gates.
