# CLRS BFS sampler/hint preflight

CLRS is pinned in the benchmark source registry at
`google-deepmind/clrs@f8f25086f33e0b4c128583167151fc53b293c5f5`
under Apache-2.0.

This preflight deliberately validates the benchmark-generation boundary rather
than adopting CLRS's model-training baseline.

The exact pinned package is installed in an isolated public GHA job and asked to
sample four deterministic BFS instances. The preflight:

- records the sampled adjacency matrices' basic properties;
- extracts the CLRS source-node input and final `pi` parent output;
- independently recomputes the BFS parent vector with the same documented
  ascending-node tie-breaking and requires exact agreement;
- verifies that `reach_h` and `pi_h` intermediate hint trajectories exist;
- records per-sample hint lengths and tensor shapes.

No CLRS source or generated data is copied into this repository.

## Integration decision

If this gate passes, CLRS can be treated as a pinned external benchmark
dependency behind our common benchmark interface. Its JAX/TensorFlow training
stack remains upstream and is not part of Campaign 0001's model harness.
