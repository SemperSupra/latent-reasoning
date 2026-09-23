# Substrate compatibility gate

A replay must not be applied merely because an actor has access to a host.

The compatibility gate consumes:

1. the read-only substrate observation;
2. a reviewed workload requirement profile.

It emits a small decision receipt with explicit blocking reasons.

## Coconut GPU smoke profile

The first GPU profile is intentionally a **bounded smoke admission profile**, not
a claim about final training requirements:

- Linux/Python details remain observational rather than hard-coded;
- at least 4 logical CPUs;
- at least 16 GiB RAM;
- at least 30 GiB free workspace disk;
- Docker available;
- NVIDIA visible through `nvidia-smi`;
- at least one GPU with 20,000 MiB memory;
- NVIDIA container runtime declared.

The full Coconut reproduction profile will be derived only after a real
sovereign GPU smoke measures memory/runtime behavior.

## GHA role

Public GHA intentionally fails this GPU profile. That is a successful
qualification of the placement control, not a failed experiment.

The same two commands later run on the sovereign host:

`python tools/observe_substrate.py > substrate.json`

`python tools/check_substrate_compatibility.py substrate.json profiles/coconut-reference-gpu-smoke.json --output compatibility.json`

Only an `eligible-for-bounded-apply` decision authorizes the next GPU smoke.
