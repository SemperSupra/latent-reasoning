# Benchmark source admissibility

Campaign 0001 treats scientific credibility, source-code licensing, and benchmark contamination as separate gates.

A paper may inform methodology without granting permission to copy its code. A repository may have a permissive top-level license while individual incorporated task implementations have different obligations. Generated tasks may be legally reusable while still being scientifically unsuitable as public holdouts.

## License classes

| Class | Meaning | Default posture |
| --- | --- | --- |
| A | Standard permissive source license with clean observed provenance (for example Apache-2.0 or MIT) | Direct pinned dependency or attributed reuse |
| B | Custom permissive terms | Pinned external dependency / adapter; retain exact license |
| C | Mixed-license repository | Selective audited modules only; no wholesale vendoring |
| D | Copyleft component relevant to the selected code | Isolate/exclude unless its obligations are deliberately accepted |
| E | No explicit usable source license | Methodology/oracle only; independent implementation |

## Admission invariants

1. Every executable external benchmark dependency is pinned to an exact 40-character commit.
2. License identity is recorded from the repository itself, not inferred from a badge or paper.
3. Mixed-license repositories require module-level lineage review before a task is admitted.
4. Class B/C/D/E sources are not vendored into this repository by default.
5. Upstream notices and copyright statements are preserved where required.
6. Scientific qualification requires an independent deterministic oracle or validator even when upstream code is trusted.
7. Public fixtures may demonstrate mechanics, but qualification holdout seeds/instances remain private.
8. Generated benchmark identity includes generator revision, configuration, and seed.
9. Benchmark reputation or venue does not override a failed licensing/provenance gate.
10. Licensing clearance does not imply benchmark validity; learnability, contamination, and oracle correctness are separate gates.

## Current clean-core candidates

The preferred initial benchmark substrate is:

- CLRS / CLRS-Text for algorithm execution and exact intermediate state;
- FLaRe for formal-language/generalization behavior;
- PrOntoQA for formal proof-chain/graph reasoning;
- selected audited CoT2 tasks for multi-path/search behavior;
- ARC-GEN later for fresh abstraction.

Reasoning Gym remains a valuable source catalog, but its mixed-license lineage means individual tasks must earn admission separately.

Multilingual Reasoning Gym is initially an external oracle/comparison source. The preferred core multilingual design is to render the same clean procedural state independently in multiple languages while retaining language-independent ground truth.

## Negative-result fixture

`modular-walk-v0` remains in the repository as a benchmark-pathology regression fixture. It is no longer the primary qualification benchmark because capacity, depth, and data-pool sweeps showed memorization without adequate validation generalization.
