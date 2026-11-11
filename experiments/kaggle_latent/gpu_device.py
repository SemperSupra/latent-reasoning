#!/usr/bin/env python3
"""Kaggle T4x2 public-synthetic CUDA qualification; not latent efficacy evidence."""
import hashlib
import json
import sys
from pathlib import Path

WORKLOAD = "gpu-cuda-v1"
SCHEMA = "latent-kaggle-result/v1"
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()

def probe():
    import torch
    evidence = {
        "requested_accelerator": "NvidiaTeslaT4",
        "torch_version": str(torch.__version__),
        "cuda_build": bool(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
        "device_arithmetic_checksums": [],
    }
    if not evidence["cuda_build"]:
        return evidence, "CUDA_BUILD_MISSING"
    if not evidence["cuda_available"]:
        return evidence, "CUDA_NOT_AVAILABLE"
    if evidence["gpu_count"] != 2:
        return evidence, "GPU_COUNT_MISMATCH"
    names = [torch.cuda.get_device_properties(i).name for i in range(2)]
    evidence["device_models"] = names
    if not all("T4" in x for x in names):
        return evidence, "GPU_MODEL_MISMATCH"
    for i in range(2):
        with torch.cuda.device(i):
            x = torch.arange(4096, device=f"cuda:{i}", dtype=torch.float32).reshape(64, 64)
            y = x @ torch.eye(64, device=f"cuda:{i}")
            torch.cuda.synchronize(i)
            checksum = float(y.sum().item())
            if not torch.equal(x, y) or checksum != 8386560.0:
                return evidence, "CUDA_ARITHMETIC_FAILED"
            evidence["device_arithmetic_checksums"].append(checksum)
    return evidence, None

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
        result["failure_class"] = "GPU_RUNTIME_UNKNOWN"
    (OUT / "latent_kaggle_result.json").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print("LATENT_KAGGLE_RESULT=" + result["status"])
    return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__":
    sys.exit(main())
