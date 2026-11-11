#!/usr/bin/env python3
"""Read-only preflight for the original ICLR 2026 MeG Phi-2 environment.

This intentionally executes NO downloaded source, models, or training.
Missing research artifacts are reported, not silently synthesized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

UPSTREAM_COMMIT = "9fa8ae10cf33c82dd2200057f70dfcb73388cd74"
REQUIRED = {
    "environment.yml": ["python=3.10.14", "torch==2.4.1", "transformers==4.44.2"],
    "hparams/stage_1/phi2_zsre.yaml": ["model_para_type: phi2", "epochs: 30000"],
    "hparams/stage_4/phi2_zsre_1024.yaml": ["epochs: 1000000", "fi: 1024"],
    "evaluate/hparams/run_all_evaluation_config_paradit_phi2_zsre.yaml":
        ["model_path:", "microsoft/phi-2", "model_state_dir:"],
    "train_bert.py": [],
    "train_familiar.py": [],
    "train_neuron.py": [],
    "train.py": [],
    "evaluate/run_all_evaluation.py": [],
}
GENERATED_OR_EXTERNAL = [
    "data/new_split_data",
    "data/edit_data",
    "data_paras",
    "checkpoints_trained_bert",
    "results/029_20250518224950/checkpoints/model_epoch_all_30000.pth",
]

def audit(source: Path, expected_commit: str = UPSTREAM_COMMIT) -> dict:
    source = source.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != expected_commit:
        raise ValueError("upstream commit mismatch: " + commit)
    files = {}
    for rel, required_tokens in REQUIRED.items():
        path = source / rel
        if not path.is_file():
            raise FileNotFoundError("required upstream source absent: " + rel)
        data = path.read_bytes()
        text = data.decode("utf-8")
        absent = [token for token in required_tokens if token not in text]
        if absent:
            raise ValueError("unexpected upstream config in " + rel + ": " + repr(absent))
        files[rel] = hashlib.sha256(data).hexdigest()
    inventory = {
        rel: {"present": (source / rel).exists()}
        for rel in GENERATED_OR_EXTERNAL
    }
    missing = [rel for rel, info in inventory.items() if not info["present"]]
    return {
        "schema": "meg-phi2-source-preflight/v1",
        "upstream_commit": commit,
        "source_integrity": "PASS",
        "training_or_evaluation_executed": False,
        "derived_artifact_inventory": inventory,
        "derived_artifacts_missing": missing,
        "state": "BLOCKED_MISSING_RESEARCH_ASSETS" if missing else "SOURCE_AUDIT_PASS",
        "source_file_sha256": files,
        "notes": [
            "No model/checkpoint/dataset downloaded or executed",
            "Author's published benchmark is not reproduced by this source audit",
            "Exact dataset split, model revision and stage checkpoint hashes still required",
            "Upstream license/redistribution permission must be reviewed before copying code",
        ],
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "source_integrity": result["source_integrity"],
        "state": result["state"],
        "missing_count": len(result["derived_artifacts_missing"]),
        "upstream_commit": result["upstream_commit"],
    }, sort_keys=True))

if __name__ == "__main__":
    main()
