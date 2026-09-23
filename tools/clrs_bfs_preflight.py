#!/usr/bin/env python3
"""Preflight the exact pinned CLRS BFS sampler and hint trajectory."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


PINNED_REVISION = "f8f25086f33e0b4c128583167151fc53b293c5f5"


def ordered_bfs_parent(adjacency: np.ndarray, source: int) -> np.ndarray:
    n = adjacency.shape[0]
    parent = np.arange(n)
    reached = np.zeros(n, dtype=bool)
    reached[source] = True

    # CLRS BFS updates by complete frontiers and scans node indices in ascending
    # order. Reproduce that deterministic tie-breaking independently.
    while True:
        previous = reached.copy()
        for i in range(n):
            if not previous[i]:
                continue
            for j in range(n):
                if adjacency[i, j] > 0:
                    if parent[j] == j and j != source:
                        parent[j] = i
                    reached[j] = True
        if np.array_equal(reached, previous):
            break

    return parent


def by_name(points):
    return {point.name: np.asarray(point.data) for point in points}


def main() -> None:
    import clrs  # imported only after exact pinned source is installed by CI

    sampler, _ = clrs.build_sampler(
        "bfs",
        num_samples=4,
        seed=20260923,
        length=8,
        p=(0.3,),
        track_max_steps=False,
    )
    feedback = sampler.next()

    inputs = by_name(feedback.features.inputs)
    outputs = by_name(feedback.outputs)
    hints = by_name(feedback.features.hints)

    required_inputs = {"A", "s"}
    required_outputs = {"pi"}
    required_hints = {"reach_h", "pi_h"}

    if not required_inputs <= set(inputs):
        raise RuntimeError(f"missing BFS inputs: {required_inputs - set(inputs)}")
    if not required_outputs <= set(outputs):
        raise RuntimeError(f"missing BFS outputs: {required_outputs - set(outputs)}")
    if not required_hints <= set(hints):
        raise RuntimeError(f"missing BFS hints: {required_hints - set(hints)}")

    adjacency_batch = inputs["A"]
    source_masks = inputs["s"]
    upstream_parent_batch = outputs["pi"]

    checks = []
    for batch_index in range(adjacency_batch.shape[0]):
        adjacency = adjacency_batch[batch_index]
        source = int(np.argmax(source_masks[batch_index]))
        expected_parent = ordered_bfs_parent(adjacency, source)
        upstream_parent = upstream_parent_batch[batch_index].astype(int)

        if not np.array_equal(expected_parent, upstream_parent):
            raise RuntimeError(
                f"BFS parent mismatch for sample {batch_index}: "
                f"expected={expected_parent.tolist()} "
                f"upstream={upstream_parent.tolist()}"
            )

        checks.append({
            "sample": batch_index,
            "source": source,
            "nodes": int(adjacency.shape[0]),
            "edges_undirected": int(np.count_nonzero(np.triu(adjacency, 1))),
            "parent": upstream_parent.tolist(),
        })

    lengths = np.asarray(feedback.features.lengths)
    reach_h = hints["reach_h"]
    pi_h = hints["pi_h"]

    if reach_h.shape[1] != adjacency_batch.shape[0]:
        raise RuntimeError(f"unexpected reach_h batch shape {reach_h.shape}")
    if pi_h.shape[1] != adjacency_batch.shape[0]:
        raise RuntimeError(f"unexpected pi_h batch shape {pi_h.shape}")
    if len(lengths) != adjacency_batch.shape[0]:
        raise RuntimeError(f"unexpected hint lengths shape {lengths.shape}")
    if np.any(lengths < 1):
        raise RuntimeError(f"invalid hint lengths {lengths.tolist()}")

    receipt = {
        "schema_version": 1,
        "evidence_class": "pinned-upstream-benchmark-generator-preflight",
        "source": {
            "repository": "google-deepmind/clrs",
            "revision": PINNED_REVISION,
            "license": "Apache-2.0",
        },
        "algorithm": "bfs",
        "samples": checks,
        "hint_names": sorted(hints),
        "hint_lengths": lengths.astype(int).tolist(),
        "reach_hint_shape": list(reach_h.shape),
        "parent_hint_shape": list(pi_h.shape),
        "independent_final_parent_mismatches": 0,
        "claim_boundary": (
            "This validates the pinned CLRS BFS sampler, final parent outputs, "
            "and presence of intermediate hint trajectories. It does not adopt "
            "the CLRS model-training stack or make capability claims."
        ),
    }
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
