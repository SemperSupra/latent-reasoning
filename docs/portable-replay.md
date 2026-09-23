# Portable experiment replay

GitHub Actions is being used primarily to learn and harden Campaign 0001's
experiment machinery. The experiment itself therefore must not depend on GHA.

The portable replay layer is intentionally small:

1. an experiment remains a normal `python -m experiments.<module>` CLI;
2. a replay spec records only its bounded arguments, resource class, timeout,
   and environment reference;
3. `tools/run_replay.py` converts that spec to an argv list **without a shell**;
4. the experiment's JSON stdout remains the scientific receipt;
5. the runner writes a separate substrate metadata sidecar with repository head,
   Python/platform identity, elapsed time, and the scientific receipt digest.

The same replay command can run on:

- a GitHub-hosted Actions runner;
- a local shell or container;
- a sovereign worker;
- a future bounded Agent Dispatch task.

Agent Dispatch remains only the actuator. It does not need to understand the
scientific parameters beyond invoking an approved replay spec.

## Security / boundedness

The replay runner:

- accepts only Python modules under `experiments.*`;
- never invokes a shell;
- accepts scalar CLI arguments only;
- enforces a maximum 7200-second timeout;
- records a digest of both the replay spec and resulting scientific receipt;
- does not serialize ambient environment variables or credentials.

## Environment

The first portable environment is
`environments/python312-torch251-cpu.json`, backed by
`requirements-experiment-cpu.txt`.

This is deliberately a replay contract, not a universal environment manager.
Containerization, CUDA images, or additional environment backends should be
added only when sovereign/GPU reps demonstrate a concrete need.

## Current replay specs

- `register-state-k-sweep-smoke.json` — cheap portability qualification.
- `register-state-k-sweep-main.json` — exact full GHA K-sweep parameters.
- `parity-k2-repaired-main.json` — exact repaired parity K=2 parameters.

The main specs are not automatically rerun merely because they exist. They are
durable replay capsules for later local/sovereign execution.
