# Representative pretrained-model CPU smoke

This lane validates that a real pretrained causal LM can participate in the same recurrence interface used by Campaign 0001.

The first representative is `Qwen/Qwen2.5-0.5B-Instruct`.

The smoke checks:

- ordinary `input_ids` inference;
- `inputs_embeds` inference;
- hidden-state exposure;
- recurrent hidden-state feedback for K = 1, 2, 4;
- compute-matched fixed-embedding serial controls using the same number of model forward calls;
- bounded CPU execution on a public GitHub-hosted runner.

It does not evaluate answer correctness and must not be interpreted as capability evidence. Its purpose is to retire interface and resource-risk before sovereign GPU work.
