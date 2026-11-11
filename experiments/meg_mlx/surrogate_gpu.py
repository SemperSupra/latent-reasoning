#!/usr/bin/env python3
"""Tiny public-safe MLX GPU intervention oracle for future MeG target adaptation.

This is an original synthetic surrogate (NOT author MeG code or reproduction).
Explicit GPU, bounded matrices, no models, no datasets, no credentials.
"""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys

import mlx
import mlx.core as mx
import numpy as np


def check_gpu() -> str:
    if platform.machine().lower() != "arm64":
        raise RuntimeError("Requires Apple Silicon arm64 for this macOS qualification")
    mx.set_default_device(mx.gpu)
    desc = str(mx.default_device()).lower()
    if "gpu" not in desc:
        raise RuntimeError("MLX default execution device is not GPU: " + desc)
    return desc


def run() -> dict:
    device = check_gpu()
    # Small enough to fit even the constrained public macOS-26 runner.
    x_np = (np.arange(64, dtype=np.float32).reshape(2, 32) + 1) / 128.0
    up_np = (np.arange(32 * 48, dtype=np.float32).reshape(32, 48) % 31 - 15) / 512.0
    down_np = (np.arange(48 * 32, dtype=np.float32).reshape(48, 32) % 23 - 11) / 512.0
    u_np = (np.arange(32, dtype=np.float32) + 1) / 64.0
    v_np = (np.arange(32, dtype=np.float32) + 1) / 32.0
    alpha = np.float32(0.125)

    x, up, down, u, v = (
        mx.array(x_np), mx.array(up_np), mx.array(down_np),
        mx.array(u_np), mx.array(v_np),
    )
    baseline = mx.matmul(mx.maximum(mx.matmul(x, up), 0), down)
    def injected(uvector, strength):
        gain = mx.maximum(mx.sum(x * uvector, axis=-1, keepdims=True), 0)
        return baseline + strength * gain * v

    zero = injected(u, 0.0)
    treated = injected(u, float(alpha))
    def loss(uvector):
        d = injected(uvector, float(alpha))
        return mx.sum(d * d)
    gradient = mx.grad(loss)(u)
    mx.eval(baseline, zero, treated, gradient)

    # Independent CPU NumPy oracle, not a second MLX execution.
    base_cpu = np.maximum(x_np @ up_np, 0) @ down_np
    treated_cpu = base_cpu + alpha * np.maximum(
        np.sum(x_np * u_np, axis=-1, keepdims=True), 0
    ) * v_np

    base = np.asarray(baseline)
    zero_arr = np.asarray(zero)
    treated_arr = np.asarray(treated)
    grad_arr = np.asarray(gradient)
    max_error = float(np.max(np.abs(treated_arr - treated_cpu)))
    if not np.allclose(base, base_cpu, rtol=1e-4, atol=1e-5):
        raise AssertionError("GPU baseline did not match independent CPU oracle")
    if not np.array_equal(base, zero_arr):
        raise AssertionError("Zero intervention changed the baseline")
    if not np.allclose(treated_arr, treated_cpu, rtol=1e-4, atol=1e-5):
        raise AssertionError("GPU treatment did not match independent CPU oracle")
    if not np.isfinite(grad_arr).all() or not np.any(np.abs(grad_arr) > 1e-8):
        raise AssertionError("Missing or non-finite intervention gradient")

    return {
        "schema": "meg-mlx-surrogate-gpu/v1",
        "status": "PASS",
        "claim": "tiny_synthetic_gpu_intervention_only",
        "device": device,
        "arch": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": sys.version.split()[0],
        "mlx": importlib.metadata.version("mlx"),
        "physical_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()),
        "numpy": np.__version__,
        "shapes": {"inputs": [2, 32], "up": [32, 48], "down": [48, 32]},
        "zero_intervention_exact": True,
        "gradient_finite_nonzero": True,
        "max_cpu_gpu_abs_error": max_error,
        "treated_sha256": hashlib.sha256(treated_arr.astype("<f4").tobytes()).hexdigest(),
        "model_loaded": False,
        "training_run": False,
        "meg_paper_reproduced": False,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(run(), sort_keys=True))
    except Exception as exc:
        print(json.dumps({
            "schema": "meg-mlx-surrogate-gpu/v1", "status": "FAIL",
            "exception_type": type(exc).__name__, "message": str(exc)[:240],
            "model_loaded": False, "meg_paper_reproduced": False,
        }, sort_keys=True))
        raise
