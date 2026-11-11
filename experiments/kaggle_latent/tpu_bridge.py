#!/usr/bin/env python3
"""TPU v5e-8 JAX arithmetic and synthetic representation bridge smoke.

No private data, network, user input or model weights. This establishes
hardware and an elementary trainable latent bridge, not reasoning efficacy.
"""
from functools import partial
import hashlib
import json
import sys
from pathlib import Path

WORKLOAD = "tpu-bridge-v1"
SCHEMA = "latent-kaggle-result/v1"
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()

def probe():
    import jax
    import jax.numpy as jnp
    import numpy as np
    tpus = [d for d in jax.devices() if d.platform == "tpu"]
    metrics = {
        "requested_accelerator": "TpuV5E8",
        "jax_version": str(jax.__version__),
        "jax_backend_tpu": jax.default_backend() == "tpu",
        "tpu_count": len(tpus),
        "scope": "synthetic-method-sanity-only",
    }
    if not metrics["jax_backend_tpu"] or len(tpus) != 8:
        return metrics, "TPU_DEVICE_NOT_ATTESTED"
    # Force real arithmetic on all eight distinct physical JAX devices.
    @jax.pmap
    def transform(x):
        return x * x + jnp.float32(1.0)
    values = transform(jnp.arange(8, dtype=jnp.float32))
    observed = np.asarray(jax.device_get(values), dtype=np.float32)
    expected = np.arange(8, dtype=np.float32) ** 2 + 1
    metrics["tpu_arithmetic"] = bool(np.array_equal(observed, expected))
    metrics["tpu_arithmetic_checksum"] = float(observed.sum())
    if not metrics["tpu_arithmetic"] or metrics["tpu_arithmetic_checksum"] != 148.0:
        return metrics, "TPU_ARITHMETIC_FAILED"

    # Tiny, fixed-seed paired training control. This is a bridge sanity test.
    rng = np.random.default_rng(20261111)
    width, ntrain, ntest = 16, 512, 128
    a, _ = np.linalg.qr(rng.normal(size=(width, width)))
    b, _ = np.linalg.qr(rng.normal(size=(width, width)))
    z = rng.normal(size=(ntrain + ntest, width)).astype(np.float32)
    x = (z @ a).astype(np.float32)
    y = (z @ b).astype(np.float32)
    shuffled = y[:ntrain][rng.permutation(ntrain)].copy()
    # One data shard per TPU, replicated weights and averaged gradients:
    # verifies actual eight-device *training*, not merely device enumeration.
    shard = ntrain // len(tpus)
    with jax.default_device(tpus[0]):
        tx = jnp.asarray(x[:ntrain].reshape(8, shard, width))
        ty = jnp.asarray(y[:ntrain].reshape(8, shard, width))
        negative = jnp.asarray(shuffled.reshape(8, shard, width))
        hx, hy = jnp.asarray(x[ntrain:]), jnp.asarray(y[ntrain:])
        @partial(jax.pmap, axis_name="replica", devices=tpus)
        def step(w, features, targets):
            local_grad = jax.grad(
                lambda p: jnp.mean((features @ p - targets) ** 2))(w)
            mean_grad = jax.lax.pmean(local_grad, "replica")
            return w - jnp.float32(0.5) * mean_grad
        learned = jnp.broadcast_to(
            jnp.eye(width, dtype=jnp.float32), (8, width, width))
        control = learned
        for _ in range(160):
            learned = step(learned, tx, ty)
            control = step(control, tx, negative)
        def score(w):
            pred = hx @ w
            cosine = jnp.sum(pred * hy, axis=1) / (
                jnp.linalg.norm(pred, axis=1) * jnp.linalg.norm(hy, axis=1) + 1e-9)
            return float(jax.device_get(jnp.mean(cosine)))
        correct_score = score(learned[0])
        shuffled_score = score(control[0])
    metrics.update(
        seed=20261111, training_examples=ntrain, holdout_examples=ntest,
        updates=160, data_parallel_devices=8, heldout_cosine=round(correct_score, 6),
        shuffled_control_cosine=round(shuffled_score, 6))
    if correct_score < 0.75 or correct_score - shuffled_score < 0.45:
        return metrics, "SYNTHETIC_BRIDGE_CONTROL_FAILED"
    return metrics, None

def main():
    result = {"schema": SCHEMA, "workload": WORKLOAD, "status": "FAIL",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        metrics, failure = probe()
        result["metrics"] = metrics
        if failure:
            result["failure_class"] = failure
        else:
            result["status"] = "PASS"
    except Exception:
        result["failure_class"] = "TPU_RUNTIME_UNKNOWN"
    (OUT / "latent_kaggle_result.json").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print("LATENT_KAGGLE_RESULT=" + result["status"])
    return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__":
    sys.exit(main())
