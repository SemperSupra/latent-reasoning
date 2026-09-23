# Sovereign Coconut stage-0 apply

The two-GPU CoT stage-0 reproduction is now exposed through one fail-closed
command:

```
python tools/apply_coconut_stage0.py
```

It implements the project's normal lifecycle:

1. **discover/observe** — record the current host as `.local/coconut-stage0/substrate.json`;
2. **plan** — evaluate the reviewed two-GPU stage-0 placement profile;
3. **apply** — only when compatible, require a clean worktree, build the exact
   stage-0 capsule at the current repository revision, and run it with Docker
   GPU exposure;
4. **verify** — require matching capsule/execution identity and at least one
   non-empty SHA-256 checkpoint receipt before writing `verified.json`.

An incompatible host exits with status 3 before Docker build or training.

## Authority boundary

This command authorizes exactly the reviewed stage-0 CoT reproduction. It does
not:

- install or modify NVIDIA drivers/runtime;
- weaken the two-GPU admission profile;
- start later Coconut latent stages;
- promote checkpoints into Foundry;
- perform qualification in Model Spelunker.

Those remain separate work packages.

## GHA qualification

GHA is expected to reject the apply because the standard public CPU runner has
no compatible two-GPU substrate. That refusal is useful method evidence: the
same command will proceed automatically on a compatible sovereign/local host
without changing scientific configuration.

## Output

Default local state is gitignored under:

`.local/coconut-stage0/`

with:

- `substrate.json`
- `compatibility.json`
- `stage0-capsule-inspection.json` after execution
- `stage0-execution.json` after training
- `verified.json` only after successful checkpoint verification

This keeps the human out of prompt shuttling: a local actor can invoke one
reviewed command and return only the compact verified/blocking receipt.
