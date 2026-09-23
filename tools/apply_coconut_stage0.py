#!/usr/bin/env python3
"""Fail-closed sovereign apply for Coconut GSM CoT stage-0."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_substrate_compatibility import evaluate  # noqa: E402
from tools.observe_substrate import observe  # noqa: E402


DEFAULT_PROFILE = ROOT / "profiles" / "coconut-gsm-cot-stage0-2gpu.json"
DEFAULT_DOCKERFILE = ROOT / "containers" / "coconut-stage0.Dockerfile"
DEFAULT_OUTPUT = ROOT / ".local" / "coconut-stage0"
DEFAULT_IMAGE = "latent-reasoning-coconut-stage0:local"


def run_checked(argv: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def repository_revision() -> str:
    return run_checked(["git", "rev-parse", "HEAD"]).stdout.strip()


def require_clean_worktree() -> None:
    status = run_checked(["git", "status", "--porcelain"]).stdout
    if status.strip():
        raise RuntimeError(
            "repository worktree is not clean; reconcile changes before stage-0 apply"
        )


def build_argv(*, image_tag: str, dockerfile: Path, source_revision: str) -> list[str]:
    return [
        "docker",
        "build",
        "--build-arg",
        f"SOURCE_REVISION={source_revision}",
        "-f",
        str(dockerfile),
        "-t",
        image_tag,
        ".",
    ]


def run_argv(*, image_tag: str, output_dir: Path) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size",
        "4g",
        "-v",
        f"{output_dir.resolve()}:/out",
        image_tag,
    ]


def verify_outputs(*, output_dir: Path, source_revision: str) -> dict:
    inspection_path = output_dir / "stage0-capsule-inspection.json"
    execution_path = output_dir / "stage0-execution.json"

    if not inspection_path.is_file() or not execution_path.is_file():
        raise RuntimeError("stage-0 apply did not produce required receipts")

    inspection = json.loads(inspection_path.read_text())
    execution = json.loads(execution_path.read_text())

    if inspection.get("evidence_class") != "coconut-stage0-capsule-inspection":
        raise RuntimeError("unexpected stage-0 inspection evidence class")
    if inspection.get("container_contract") != "coconut-stage0-v1":
        raise RuntimeError("unexpected stage-0 container contract")
    if inspection.get("repository_source_revision") != source_revision:
        raise RuntimeError("stage-0 inspection source revision mismatch")

    if execution.get("evidence_class") != "coconut-stage0-training-execution":
        raise RuntimeError("unexpected stage-0 execution evidence class")
    if execution.get("container_contract") != "coconut-stage0-v1":
        raise RuntimeError("unexpected stage-0 execution container contract")
    if execution.get("repository_source_revision") != source_revision:
        raise RuntimeError("stage-0 execution source revision mismatch")
    if execution.get("exit_code") != 0:
        raise RuntimeError("stage-0 training execution did not exit successfully")
    if execution.get("cuda_device_count", 0) < 2:
        raise RuntimeError("stage-0 execution did not observe two CUDA devices")

    checkpoints = execution.get("checkpoints") or []
    if not checkpoints:
        raise RuntimeError("stage-0 execution produced no checkpoint receipts")
    for checkpoint in checkpoints:
        if not checkpoint.get("sha256") or checkpoint.get("size_bytes", 0) <= 0:
            raise RuntimeError("invalid checkpoint receipt")

    return {
        "schema_version": 1,
        "status": "verified",
        "source_revision": source_revision,
        "container_contract": "coconut-stage0-v1",
        "cuda_device_count": execution["cuda_device_count"],
        "elapsed_seconds": execution["elapsed_seconds"],
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "inspection_receipt": str(inspection_path),
        "execution_receipt": str(execution_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--dockerfile", type=Path, default=DEFAULT_DOCKERFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-tag", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Observe and evaluate compatibility but do not build or run.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    observation = observe()
    (output_dir / "substrate.json").write_text(
        json.dumps(observation, sort_keys=True) + "\n"
    )

    profile = json.loads(args.profile.read_text())
    compatibility = evaluate(observation, profile)
    (output_dir / "compatibility.json").write_text(
        json.dumps(compatibility, sort_keys=True) + "\n"
    )

    if not compatibility["compatible"]:
        print(json.dumps({
            "status": "blocked",
            "decision": compatibility["decision"],
            "blocking_reasons": compatibility["blocking_reasons"],
            "observation": str(output_dir / "substrate.json"),
            "compatibility": str(output_dir / "compatibility.json"),
        }, sort_keys=True))
        raise SystemExit(3)

    if args.plan_only:
        print(json.dumps({
            "status": "eligible",
            "decision": compatibility["decision"],
            "apply_performed": False,
        }, sort_keys=True))
        return

    require_clean_worktree()
    source_revision = repository_revision()

    build = build_argv(
        image_tag=args.image_tag,
        dockerfile=args.dockerfile.resolve(),
        source_revision=source_revision,
    )
    run_checked(build)

    run_checked(
        run_argv(
            image_tag=args.image_tag,
            output_dir=output_dir,
        )
    )

    verified = verify_outputs(
        output_dir=output_dir,
        source_revision=source_revision,
    )
    (output_dir / "verified.json").write_text(
        json.dumps(verified, sort_keys=True) + "\n"
    )
    print(json.dumps(verified, sort_keys=True))


if __name__ == "__main__":
    main()
