# Register-state complete-universe mini-regime

This is the second independently implemented formal-state benchmark lane.

The initial profile is intentionally small enough to enumerate its entire ID
universe rather than relying on a random validation split.

## State

Two registers, each taking values modulo 3.

## Operation vocabulary

Five deterministic operations:

1. increment register 0 modulo 3;
2. increment register 1 modulo 3;
3. copy register 1 into register 0;
4. copy register 0 into register 1;
5. swap the two registers.

Every instance also queries either register 0 or register 1 after execution.

## Complete ID curriculum

All possible initial states, operation sequences, and queries are enumerated for:

- stage 1: one operation — 90 cumulative examples;
- stage 2: up to two operations — 540 cumulative examples;
- stage 3: up to three operations — 2,790 cumulative examples.

A stage is saturated only after repeated 100% accuracy on the complete
cumulative universe.

## OOD axis

Only saturated seeds are evaluated on fresh deterministic samples with four to
six operations.

This profile is deliberately smaller than the broader 4-register/mod-7
procedural generator. It exists to establish a complete-universe learnability
and extrapolation regime before scaling the state space or comparing reasoning
representations.
