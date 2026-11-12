#!/usr/bin/env python3
"""Direct, bounded Kaggle experiment controller for Latent Reasoning Campaign 0001.

No Agent Dispatch, long-running controller, arbitrary scripts or shell execution.
Source is selected from a fixed reviewed allowlist. Public CI only calls plan.
Credentialed submit/observe is restricted to reviewed default-branch workflow.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
WORKLOADS = {
    "gpu-cuda-v1": ("gpu_device.py", "NvidiaTeslaT4", "gpu", 600),
    "tpu-bridge-v1": ("tpu_bridge.py", "TpuV5E8", "tpu", 1200),
}
SCHEMA = "latent-kaggle-control/v1"
RESULT_SCHEMA = "latent-kaggle-result/v1"
NAMESPACE = "latent-c001-"
RECEIPT = "latent_kaggle_result.json"

class Denied(Exception):
    pass

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def plan(action: str, workload: str, version: int = 0) -> dict:
    if action not in ("plan", "submit", "observe") or workload not in WORKLOADS:
        raise Denied("UNKNOWN_TYPED_OPERATION")
    if action == "observe" and not 1 <= version <= 999999:
        raise Denied("OBSERVE_REQUIRES_EXACT_VERSION")
    if action != "observe" and version != 0:
        raise Denied("VERSION_ONLY_VALID_FOR_OBSERVE")
    filename, accelerator, quota, seconds = WORKLOADS[workload]
    path = ROOT / "experiments" / "kaggle_latent" / filename
    if not path.is_file() or path.is_symlink():
        raise Denied("ALLOWLIST_SOURCE_UNAVAILABLE")
    return {
        "action": action, "workload": workload,
        "slug": NAMESPACE + workload, "accelerator": accelerator,
        "quota_kind": quota, "limit_seconds": seconds,
        "source_sha256": sha256(path.read_bytes()), "source_path": str(path),
        "native_version": version,
    }

def metadata(ref: str, p: dict) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}/latent-c001-[a-z0-9-]{5,40}", ref):
        raise Denied("INVALID_PROVIDER_REF")
    return {
        "id": ref, "title": p["slug"], "code_file": "run.py",
        "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": p["quota_kind"] == "gpu",
        "enable_tpu": p["quota_kind"] == "tpu",
        "enable_internet": False, "machine_shape": p["accelerator"],
        "docker_image_pinning_type": "latest",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }

def identity(api) -> str:
    api.authenticate()
    user = api.get_config_value(api.CONFIG_NAME_USER)
    if not isinstance(user, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", user):
        raise Denied("KAGGLE_IDENTITY_UNKNOWN")
    return user

def quota(api, p: dict) -> int:
    snapshot = api.quota_view()
    slot = snapshot.gpu_quota if p["quota_kind"] == "gpu" else snapshot.tpu_quota
    if slot is None or slot.time_used is None or slot.total_time_allowed is None:
        raise Denied("KAGGLE_QUOTA_UNKNOWN")
    remaining = (slot.total_time_allowed - slot.time_used).total_seconds()
    if remaining < p["limit_seconds"] + 300:
        raise Denied("KAGGLE_QUOTA_INSUFFICIENT")
    return int(remaining)

def ensure_not_existing_or_active(api, target: str) -> None:
    # Kaggle quota is shared account-wide; check EVERY running notebook, not\n    # just known Latent/Agent Dispatch or shared library namespaces.
    token = None
    seen = set()
    for _ in range(20):
        opts = {"mine": True, "page_size": 100}
        if token:
            opts["page_token"] = token
        page = api.kernels_list_with_response(**opts)
        if page is None or not hasattr(page, "kernels"):
            raise Denied("KAGGLE_LIST_UNAVAILABLE")
        for entry in (page.kernels or []):
            ref = getattr(entry, "ref", None)
            if not isinstance(ref, str) or "/" not in ref:
                raise Denied("KAGGLE_LIST_UNTRUSTED")
            if ref.lower() == target.lower():
                raise Denied("TARGET_EXISTS_OBSERVE_INSTEAD")
            status = str(getattr(api.kernels_status(ref), "status", "UNKNOWN")).upper()
            if any(k in status for k in ("QUEUED", "RUNNING", "PENDING", "STARTING")):
                raise Denied("EXISTING_KAGGLE_ACCOUNT_JOB_ACTIVE")
            if not any(k in status for k in
                       ("COMPLETE", "SUCCESS", "ERROR", "FAIL", "CANCEL")):
                raise Denied("EXISTING_KAGGLE_ACCOUNT_JOB_UNKNOWN")
        token = getattr(page, "next_page_token", None)
        if not token:
            return
        if not isinstance(token, str) or token in seen:
            raise Denied("KAGGLE_PAGE_UNKNOWN")
        seen.add(token)
    raise Denied("KAGGLE_LIST_BOUND_EXCEEDED")

def submit(api, p: dict) -> dict:
    user = identity(api)
    remain = quota(api, p)
    ref = f"{user}/{p['slug']}"
    ensure_not_existing_or_active(api, ref)
    with tempfile.TemporaryDirectory(prefix="latent-kaggle-") as tmp:
        folder = Path(tmp)
        source = Path(p["source_path"])
        data = source.read_bytes()
        if sha256(data) != p["source_sha256"]:
            raise Denied("SOURCE_HASH_CHANGED")
        (folder / "run.py").write_bytes(data)
        if sha256((folder / "run.py").read_bytes()) != p["source_sha256"]:
            raise Denied("STAGED_SOURCE_HASH_CHANGED")
        (folder / "kernel-metadata.json").write_text(
            json.dumps(metadata(ref, p), sort_keys=True) + "\n", encoding="utf-8")
        native = api.kernels_push(str(folder), timeout=str(p["limit_seconds"]),
                                  acc=p["accelerator"])
    if native is None or getattr(native, "error", None):
        raise Denied("PROVIDER_SUBMISSION_REJECTED")
    version = getattr(native, "versionNumber", None)
    if not isinstance(version, int) or version < 1:
        raise Denied("PROVIDER_SUBMITTED_VERSION_UNKNOWN")
    return {"state": "SUBMITTED", "is_terminal": False,
            "native_version": version, "quota_remaining_before_seconds": remain}

def require_native_version(api, user: str, slug: str, expected: int) -> None:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = user, slug
    with api.build_kaggle_client() as client:
        details = client.kernels.kernels_api_client.get_kernel(req)
    actual = getattr(getattr(details, "metadata", None), "current_version_number", None)
    if not isinstance(actual, int) or actual != expected:
        raise Denied("NATIVE_VERSION_UNVERIFIED_OR_DRIFTED")

def verify_science(result: dict, workload: str) -> dict:
    if (not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA
        or result.get("workload") != workload or result.get("status") != "PASS"):
        raise Denied("SCIENTIFIC_RECEIPT_INVALID")
    m = result.get("metrics")
    if not isinstance(m, dict):
        raise Denied("SCIENTIFIC_METRICS_MISSING")
    if workload == "gpu-cuda-v1":
        checks = m.get("device_arithmetic_checksums")
        if (m.get("gpu_count") != 2 or m.get("requested_accelerator") != "NvidiaTeslaT4"
            or not isinstance(checks, list) or checks != [8386560.0, 8386560.0]
            or not all("T4" in s for s in m.get("device_models", []) )
            or len(m.get("device_models", [])) != 2):
            raise Denied("GPU_SCIENCE_ORACLE_FAILED")
        return {"gpu_count": 2, "device_arithmetic_checksums": checks}
    if (m.get("requested_accelerator") != "TpuV5E8"
        or m.get("tpu_count") != 8 or m.get("jax_backend_tpu") is not True
        or m.get("tpu_arithmetic") is not True
        or m.get("tpu_arithmetic_checksum") != 148.0
        or m.get("data_parallel_devices") != 8
        or m.get("scope") != "synthetic-method-sanity-only"):
        raise Denied("TPU_HARDWARE_ORACLE_FAILED")
    score, control = m.get("heldout_cosine"), m.get("shuffled_control_cosine")
    if (type(score) not in (int, float) or type(control) not in (int, float)
        or not (math.isfinite(score) and math.isfinite(control))
        or not (-1 <= score <= 1 and -1 <= control <= 1)
        or not (score >= 0.75 and score - control >= 0.45)):
        raise Denied("TPU_BRIDGE_ORACLE_FAILED")
    return {"tpu_count": 8, "tpu_arithmetic_checksum": 148.0,
            "heldout_cosine": score, "shuffled_control_cosine": control}

def observe(api, p: dict, out: Path) -> dict:
    user = identity(api)
    version, slug = p["native_version"], p["slug"]
    require_native_version(api, user, slug, version)
    ref = f"{user}/{slug}/{version}"
    status = str(getattr(api.kernels_status(ref), "status", "UNKNOWN")).upper()
    if any(s in status for s in ("RUNNING", "PENDING", "QUEUED", "STARTING")):
        return {"state": "NONTERMINAL", "is_terminal": False}
    if not any(s in status for s in ("COMPLETE", "SUCCESS", "ERROR", "FAIL", "CANCEL")):
        raise Denied("PROVIDER_STATUS_UNKNOWN")
    if any(s in status for s in ("ERROR", "FAIL", "CANCEL")):
        return {"state": "TERMINAL_FAILED", "is_terminal": True}
    out.mkdir(parents=True, exist_ok=True)
    api.kernels_output(ref, str(out), file_pattern=r"^latent_kaggle_result[.]json$",
                       force=False, quiet=True)
    require_native_version(api, user, slug, version)
    file = out / RECEIPT
    if not file.is_file() or file.stat().st_size > 65536:
        raise Denied("RESULT_MISSING_OR_OVERSIZED")
    raw = file.read_bytes()
    try:
        scientific = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise Denied("RESULT_JSON_INVALID")
    result = verify_science(scientific, p["workload"])
    return {"state": "TERMINAL_VERIFIED", "is_terminal": True,
            "native_version": version, "science_sha256": sha256(raw),
            "verified_metrics": result}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "submit", "observe"), required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--out-dir", default=".kaggle-latent")
    args = parser.parse_args()
    out = Path(args.out_dir)
    base = {"schema": SCHEMA, "action": args.action, "workload": args.workload}
    try:
        p = plan(args.action, args.workload, args.version)
        base.update({k: p[k] for k in ("source_sha256", "accelerator", "limit_seconds")})
        if args.action == "plan":
            observation = {"state": "PLANNED", "is_terminal": False}
        else:
            if not os.getenv("KAGGLE_API_TOKEN"):
                raise Denied("NO_KAGGLE_ENV_TOKEN")
            from kaggle import api
            observation = (submit(api, p) if args.action == "submit"
                           else observe(api, p, out))
        base.update(observation)
        exit_code = 0
    except Denied as exc:
        base.update(state="BLOCKED", is_terminal=False, failure_class=str(exc))
        exit_code = 2
    except Exception:
        base.update(state="BLOCKED", is_terminal=False,
                    failure_class="PROVIDER_RUNTIME_UNKNOWN")
        exit_code = 2
    # Strict allowlist to avoid accidental provider-secret or private-data exposure.
    allowed = ("schema", "action", "workload", "source_sha256", "accelerator",
               "limit_seconds", "state", "is_terminal", "failure_class",
               "native_version", "science_sha256", "verified_metrics")
    clean = {k: base[k] for k in allowed if k in base}
    out.mkdir(parents=True, exist_ok=True)
    (out / "control-receipt.json").write_text(
        json.dumps(clean, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("LATENT_KAGGLE_STATE=" + str(clean["state"]))
    if exit_code:
        print("LATENT_KAGGLE_BLOCKED=" + clean["failure_class"], file=sys.stderr)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
