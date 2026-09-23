# Read-only substrate observer

Before applying an experiment to a sovereign/local host, run:

`python tools/observe_substrate.py > substrate.json`

The observer performs no installation or mutation. It records only execution
capabilities relevant to Campaign 0001:

- OS/kernel architecture and Python version;
- logical CPU count, total RAM, and free workspace disk;
- Docker availability, server version, and declared runtimes;
- NVIDIA visibility, GPU names/memory, driver version, and whether an NVIDIA
  container runtime is declared.

It intentionally does not record hostname, username, IP addresses, environment
variables, mounted paths, repository names outside this project, or credentials.

## Discover -> plan -> apply -> verify

1. **Discover/observe:** generate this receipt.
2. **Plan:** compare it with the selected replay/environment requirements.
3. **Apply:** run an approved replay spec directly or in its qualified container.
4. **Verify:** retain scientific receipt + substrate metadata and reconcile them
   into the campaign evidence plane.

GHA itself runs this observer as a qualification fixture. A GHA observation is
not expected to report an NVIDIA GPU; the purpose is to prove the observer
fails open into truthful capability data rather than assuming a target host.
