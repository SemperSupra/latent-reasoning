# FLaRe reference-data preflight

FLaRe is pinned in the benchmark source registry at
`rycolab/flare@a671eb9932620cc888757612f30845efe3304831` under the MIT license.

The first integration posture is **reference-data oracle**, not vendoring.

The CI preflight clones the exact upstream revision and independently validates
all 10,000 released examples for each of two task families:

- parity;
- Dyck-2 with two bracket types and maximum nesting depth 3 (upstream directory `dyck-2-3`).

The validator recomputes the expected class directly from the mathematical task
definition. Empty-string examples are retained rather than accidentally dropped.

The receipt records exact SHA-256 identities of the upstream input and label
files so later comparisons can prove which data revision was used.

## Boundary

Passing this preflight establishes that these two pinned upstream reference sets
match our independent oracles. It does not make the public FLaRe examples
qualification holdouts. Campaign qualification uses privately generated
instances and seeds where contamination resistance matters.


The first independent oracle intentionally treated Dyck validity as unbounded
nesting and disagreed with 11 upstream negatives. Inspection showed all 11 were
balanced strings whose maximum nesting depth exceeded 3. The corrected oracle
therefore captures the upstream language definition rather than weakening the
check.
