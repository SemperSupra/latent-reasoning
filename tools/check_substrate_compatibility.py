#!/usr/bin/env python3
"""Compare a read-only substrate observation against replay requirements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(observation: dict, profile: dict) -> dict:
    req=profile["requirements"]
    reasons=[]
    warnings=[]

    resources=observation["resources"]
    docker=observation["docker"]
    nvidia=observation["nvidia"]

    if resources["logical_cpus"] < req["min_logical_cpus"]:
        reasons.append("logical_cpu_count_below_minimum")

    memory=resources.get("memory_total_bytes")
    if memory is None:
        warnings.append("memory_total_unknown")
    elif memory < req["min_memory_bytes"]:
        reasons.append("memory_below_minimum")

    if resources["disk_free_bytes"] < req["min_disk_free_bytes"]:
        reasons.append("disk_free_below_minimum")

    if req["docker_required"] and not docker["available"]:
        reasons.append("docker_unavailable")

    if req["nvidia_required"] and not nvidia["nvidia_smi_available"]:
        reasons.append("nvidia_device_visibility_unavailable")

    gpus=nvidia.get("gpus",[])
    if len(gpus) < req["min_gpu_count"]:
        reasons.append("gpu_count_below_minimum")

    if req["min_gpu_memory_mib"] > 0:
        qualifying=[
            gpu for gpu in gpus
            if gpu.get("memory_total_mib",0) >= req["min_gpu_memory_mib"]
        ]
        if len(qualifying) < req["min_gpu_count"]:
            reasons.append("qualifying_gpu_memory_below_minimum")

    if (
        req["nvidia_container_runtime_required"]
        and not nvidia["container_runtime_declared"]
    ):
        reasons.append("nvidia_container_runtime_not_declared")

    return {
        "schema_version":1,
        "profile_id":profile["id"],
        "resource_class":profile["resource_class"],
        "compatible":not reasons,
        "blocking_reasons":reasons,
        "warnings":warnings,
        "observed":{
          "logical_cpus":resources["logical_cpus"],
          "memory_total_bytes":memory,
          "disk_free_bytes":resources["disk_free_bytes"],
          "docker_available":docker["available"],
          "docker_runtimes":docker["runtimes"],
          "nvidia_smi_available":nvidia["nvidia_smi_available"],
          "gpu_count":len(gpus),
          "gpus":gpus
        },
        "decision":(
            "eligible-for-bounded-apply"
            if not reasons
            else "do-not-apply"
        )
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("observation",type=Path)
    p.add_argument("profile",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()

    observation=json.loads(args.observation.read_text())
    profile=json.loads(args.profile.read_text())
    result=evaluate(observation,profile)
    encoded=json.dumps(result,sort_keys=True)

    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded+"\n")
    else:
        print(encoded)


if __name__=="__main__":
    main()
