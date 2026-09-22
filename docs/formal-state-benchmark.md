# Formal-state benchmark core

This is the replacement first-line toy regime for Campaign 0001.

The implementation is independent and mathematical: no third-party benchmark code or data is copied into the repository. The design is informed by established formal-language and state-tracking methodology, with FLaRe recorded in the benchmark source registry as a clean methodological reference.

## Families

### Parity

Input: a bit sequence.

State: the running XOR after each bit.

Ground truth includes the exact state after every input symbol, not only the final answer.

Public ID difficulty is lengths 1–8. Lengths 9–32 are reserved as the initial OOD axis.

### Register state

Input: an initial vector of modular registers followed by deterministic operations:

- add a modular delta to one register;
- copy one register into another;
- swap two registers.

The target queries one register after the operation sequence. The benchmark records the full register vector after every operation.

Public ID difficulty is 1–6 operations. The initial length-OOD axis is 7–24 operations.

## Hard admission gate

A model/treatment does not earn OOD interpretation until it reaches **100% ID accuracy** under the frozen benchmark regime.

This intentionally separates three questions:

1. Can the model learn the task at all?
2. Can it extrapolate beyond the trained difficulty envelope?
3. Does the reasoning representation change either property under compute-matched controls?

Failure at question 1 blocks conclusions about questions 2 and 3.

## Evidence handling

Public fixtures and seeds exist only to test generators, oracles, schemas, and harnesses.

Qualification seeds and materialized instances remain private. Every instance carries a deterministic SHA-256 digest derived from its canonical representation.

## Why modular-walk-v0 remains

The earlier modular-walk regime is retained as a negative-result fixture. It demonstrated a real benchmark pathology: increased depth/capacity could nearly memorize the training set without producing useful validation generalization. It should not be used as the primary latent-reasoning qualification regime.
