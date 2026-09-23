#!/usr/bin/env python3
"""Fail-closed discover -> plan -> apply -> verify for Coconut GPU smoke."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_substrate_compatibility import evaluate  # noqa: E402
from tools.observe_substrate import observe  # noqa: E402


DEFAULT_PROFILE = ROOT / "profiles" / "coconut-reference-gpu-smoke.json"
DEFAULT_REPLAY = ROOT / "replays" / "coconut-reference-gpu-smoke.json"
DEFAULT_DOCKERFILE = ROOT / "containers" / "coconut-gpu-smoke.Dockerfile"
DEFAULT_OUTPUT = ROOT / ".local" / "coconut-gpu-smoke"
DEFAULT_IMAGE = "latent-reasoning-coconut-gpu-smoke:local"


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
            "repository worktree is not clean; reconcile changes before GPU apply"
        )


def build_argv(
    *,
    image_tag: str,
    dockerfile: Path,
    source_revision: str,
) -> list[str]:
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


def replay_argv(
    *,
    image_tag: str,
    replay_spec: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{output_dir.resolve()}:/out",
        image_tag,
        str(replay_spec.relative_to(ROOT)),
        "--output",
        "/out/scientific-receipt.json",
        "--metadata",
        "/out/replay-metadata.json",
    ]


def verify_outputs(
    *,
    output_dir: Path,
    replay_spec: dict,
    source_revision: str,
) -> dict:
    scientific_path = output_dir / "scientific-receipt.json"
    metadata_path = output_dir / "replay-metadata.json"

    if not scientific_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("GPU replay did not produce both required receipts")

    scientific = json.loads(scientific_path.read_text())
    metadata = json.loads(metadata_path.read_text())

    if scientific.get("evidence_class") != "pinned-upstream-reference-backbone-gpu-smoke":
        raise RuntimeError("unexpected scientific receipt evidence class")
    if metadata.get("replay_spec_id") != replay_spec["id"]:
        raise RuntimeError("replay metadata spec identity mismatch")
    if metadata.get("source_revision") != source_revision:
        raise RuntimeError("replay metadata source revision mismatch")
    if metadata.get("execution", {}).get("container_contract") != "coconut-gpu-smoke-v1":
        raise RuntimeError("unexpected container contract identity")
    if scientific.get("device", {}).get("total_memory_bytes", 0) <= 0:
        raise RuntimeError("GPU receipt lacks usable device memory observation")
    if not scientific.get("gradient_through_recurrence"):
        raise RuntimeError("GPU receipt did not verify recurrence gradient flow")
    if not scientific.get("optimizer_step_changed_parameter"):
        raise RuntimeError("GPU receipt did not verify optimizer update")

    return {
        "schema_version": 1,
        "status": "verified",
        "source_revision": source_revision,
        "replay_spec_id": replay_spec["id"],
        "scientific_receipt_sha256": metadata["scientific_receipt_sha256"],
        "device": scientific["device"],
        "peak_cuda_allocated_bytes": scientific["peak_cuda_allocated_bytes"],
        "peak_cuda_reserved_bytes": scientific["peak_cuda_reserved_bytes"],
        "elapsed_seconds": scientific["elapsed_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
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
    replay_spec = json.loads(args.replay.read_text())

    build = build_argv(
        image_tag=args.image_tag,
        dockerfile=args.dockerfile.resolve(),
        source_revision=source_revision,
    )
    run_checked(build)

    replay = replay_argv(
        image_tag=args.image_tag,
        replay_spec=args.replay.resolve(),
        output_dir=output_dir,
    )
    run_checked(replay)

    verified = verify_outputs(
        output_dir=output_dir,
        replay_spec=replay_spec,
        source_revision=source_revision,
    )
    (output_dir / "verified.json").write_text(
        json.dumps(verified, sort_keys=True) + "\n"
    )
    print(json.dumps(verified, sort_keys=True))


if __name__ == "__main__":
    main()
