# Sovereign Coconut GPU smoke apply

The local/sovereign apply path is now one bounded command:

```
python tools/apply_coconut_gpu_smoke.py
```

The command implements the project's normal lifecycle:

1. **discover/observe** — write `.local/coconut-gpu-smoke/substrate.json`;
2. **plan** — evaluate the reviewed Coconut GPU-smoke profile and write
   `compatibility.json`;
3. **apply** — only when compatible, require a clean Git worktree, build the
   digest-pinned offline capsule at the exact checked-out revision, and execute
   the replay with `docker run --gpus all`;
4. **verify** — require the scientific receipt and replay sidecar to agree on
   source/spec/container identity and core GPU-smoke checks, then write
   `verified.json`.

If the host is ineligible, the command exits with status 3 **before Docker build
or GPU execution**. The observation and compatibility receipts remain as the
only output so the blocker can be reconciled without transcript archaeology.

`--plan-only` can be used on an eligible host when observation/placement is
wanted without apply.

## Authority boundary

Invoking the apply command is the bounded authority to perform this one smoke.
It does not start full Coconut training, alter host configuration, install GPU
drivers/runtimes, or weaken a failed compatibility gate.

If the observer finds missing NVIDIA runtime/driver capability, the tool stops;
host remediation is a separate plan/apply decision.

## Output

Default local state is under the gitignored path:

`.local/coconut-gpu-smoke/`

containing:

- `substrate.json`
- `compatibility.json`
- `scientific-receipt.json` (only after apply)
- `replay-metadata.json` (only after apply)
- `verified.json` (only after successful verification)

This keeps the human out of the prompt-shuttling loop: a local actor can run one
command and return only the compact verified/blocking receipt through the
project's normal durable coordination channel.
