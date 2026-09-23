#!/usr/bin/env python3
"""Read-only observation of an experiment execution substrate."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess


def run(argv: list[str]) -> str | None:
    try:
        p=subprocess.run(
            argv,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        return p.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def memory_total_bytes() -> int | None:
    try:
        pages=os.sysconf("SC_PHYS_PAGES")
        page_size=os.sysconf("SC_PAGE_SIZE")
        return int(pages*page_size)
    except (ValueError,OSError,AttributeError):
        return None


def docker_observation() -> dict:
    version=run(["docker","version","--format","{{.Server.Version}}"])
    runtimes_raw=run(["docker","info","--format","{{json .Runtimes}}"])
    runtimes=[]
    if runtimes_raw:
        try:
            obj=json.loads(runtimes_raw)
            if isinstance(obj,dict):
                runtimes=sorted(str(k) for k in obj)
        except json.JSONDecodeError:
            pass
    return {
        "available":version is not None,
        "server_version":version,
        "runtimes":runtimes,
    }


def nvidia_observation(docker: dict) -> dict:
    raw=run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    gpus=[]
    if raw:
        for line in raw.splitlines():
            parts=[part.strip() for part in line.split(",")]
            if len(parts)!=3:
                continue
            try:
                memory=int(float(parts[1]))
            except ValueError:
                continue
            gpus.append({
                "name":parts[0],
                "memory_total_mib":memory,
                "driver_version":parts[2],
            })
    return {
        "nvidia_smi_available":raw is not None,
        "gpus":gpus,
        "container_runtime_declared":"nvidia" in docker["runtimes"],
    }


def observe() -> dict:
    docker=docker_observation()
    disk=shutil.disk_usage(".")
    return {
        "schema_version":1,
        "platform":{
            "system":platform.system(),
            "release":platform.release(),
            "machine":platform.machine(),
            "python_version":platform.python_version(),
        },
        "resources":{
            "logical_cpus":os.cpu_count() or 1,
            "memory_total_bytes":memory_total_bytes(),
            "disk_free_bytes":disk.free,
        },
        "docker":docker,
        "nvidia":nvidia_observation(docker),
    }


def main() -> None:
    print(json.dumps(observe(),sort_keys=True))


if __name__=="__main__":
    main()
