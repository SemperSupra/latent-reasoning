#!/usr/bin/env python3
"""Run a bounded experiment replay spec without invoking a shell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_spec(path: Path) -> dict:
    spec=json.loads(path.read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported replay spec version")

    module=spec.get("module")
    if not isinstance(module,str) or not module.startswith("experiments."):
        raise ValueError("module must be under experiments.*")

    args=spec.get("arguments")
    if not isinstance(args,dict):
        raise ValueError("arguments must be an object")
    for key,value in args.items():
        if not key.replace("_","-").replace("-","").isalnum():
            raise ValueError(f"invalid argument name: {key}")
        if isinstance(value,bool) or not isinstance(value,(str,int,float)):
            raise ValueError(f"unsupported argument value for {key}")

    timeout=spec.get("timeout_seconds")
    if not isinstance(timeout,int) or not (1 <= timeout <= 7200):
        raise ValueError("timeout_seconds out of bounds")

    return spec


def argv_for(spec: dict) -> list[str]:
    argv=[sys.executable,"-m",spec["module"]]
    for key,value in spec["arguments"].items():
        argv.extend([f"--{key.replace('_','-')}",str(value)])
    return argv


def git_head() -> str | None:
    try:
        result=subprocess.run(
            ["git","rev-parse","HEAD"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_revision() -> str | None:
    return git_head() or os.environ.get("LATENT_REASONING_SOURCE_REVISION")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("spec",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--metadata",type=Path,required=True)
    args=parser.parse_args()

    spec=load_spec(args.spec)
    argv=argv_for(spec)

    started=time.time()
    perf=time.perf_counter()
    proc=subprocess.run(
        argv,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=spec["timeout_seconds"],
    )
    elapsed=time.perf_counter()-perf

    if proc.returncode != 0:
        sys.stderr.buffer.write(proc.stderr)
        raise SystemExit(proc.returncode)

    # Scientific experiment stdout must be one valid JSON document.
    receipt=json.loads(proc.stdout.decode("utf-8"))
    serialized=json.dumps(receipt,sort_keys=True,separators=(",",":")).encode("utf-8")

    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.metadata.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(serialized+b"\n")

    metadata={
        "schema_version":1,
        "replay_spec_id":spec["id"],
        "replay_spec_sha256":sha256_bytes(
            json.dumps(spec,sort_keys=True,separators=(",",":")).encode("utf-8")
        ),
        "scientific_receipt_sha256":sha256_bytes(serialized),
        "module":spec["module"],
        "arguments":spec["arguments"],
        "resource_class":spec["resource_class"],
        "environment_ref":spec["environment_ref"],
        "command_argv":argv[1:],
        "git_head":git_head(),
        "source_revision":source_revision(),
        "execution":{
            "started_unix":started,
            "elapsed_seconds":elapsed,
            "python_version":platform.python_version(),
            "python_implementation":platform.python_implementation(),
            "system":platform.system(),
            "release":platform.release(),
            "machine":platform.machine(),
            "github_actions":os.environ.get("GITHUB_ACTIONS")=="true",
            "runner_os":os.environ.get("RUNNER_OS"),
            "runner_arch":os.environ.get("RUNNER_ARCH"),
            "container_contract":os.environ.get("LATENT_REASONING_CONTAINER"),
        },
    }
    args.metadata.write_text(json.dumps(metadata,sort_keys=True)+"\n")

    # Only a compact non-scientific execution summary goes to stdout.
    print(json.dumps({
        "replay_spec_id":spec["id"],
        "scientific_receipt_sha256":metadata["scientific_receipt_sha256"],
        "elapsed_seconds":elapsed,
        "github_actions":metadata["execution"]["github_actions"],
    },sort_keys=True))


if __name__=="__main__":
    main()
