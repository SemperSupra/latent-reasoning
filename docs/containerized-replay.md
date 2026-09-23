# Containerized replay

The runner-neutral replay contract can execute directly in a prepared Python
environment or inside a disposable container.

The first container contract is intentionally minimal:

- base: `python:3.12-slim`;
- CPU PyTorch environment from `requirements-experiment-cpu.txt`;
- only experiment modules, replay specs, environment descriptors, and the replay
  runner are copied into the image;
- repository `.git` state is deliberately excluded;
- the exact source revision is injected as an immutable build argument and
  recorded in replay metadata.

## Why this exists

GitHub Actions is being used to learn the execution mechanics that will later
move to sovereign/local resources. A successful container replay demonstrates
that the experiment is no longer coupled to GitHub's preinstalled Python
environment or workflow shell.

The same image contract can be built on a local Linux/Docker host. The replay
command remains:

`python tools/run_replay.py <spec> --output <receipt> --metadata <sidecar>`

through the image entrypoint.

## Non-goals

This is not yet the GPU/CUDA image, an image registry, a scheduler, or an
automatic environment manager. Those additions must be justified by later
sovereign/GPU reps.
