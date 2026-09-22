#!/usr/bin/env python3
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"

reasoning_schema = json.loads((SCHEMAS / "reasoning-substrate.schema.json").read_text())
run_schema = json.loads((SCHEMAS / "run-manifest.schema.json").read_text())

Draft202012Validator.check_schema(reasoning_schema)
Draft202012Validator.check_schema(run_schema)

# Resolve the repository-local reasoning-substrate reference without requiring
# network access or a schema registry.
run_schema_resolved = json.loads(json.dumps(run_schema))
run_schema_resolved["properties"]["reasoning_substrate"] = reasoning_schema

reasoning_validator = Draft202012Validator(reasoning_schema)
run_validator = Draft202012Validator(run_schema_resolved)

candidates = []
for base in (ROOT / "examples", ROOT / "campaigns"):
    if base.exists():
        candidates.extend(sorted(base.rglob("*.json")))

failures = []
validated = 0

for path in candidates:
    doc = json.loads(path.read_text())
    if "run_id" in doc:
        errors = sorted(run_validator.iter_errors(doc), key=lambda e: list(e.path))
        if "reasoning_substrate" in doc:
            errors.extend(
                sorted(
                    reasoning_validator.iter_errors(doc["reasoning_substrate"]),
                    key=lambda e: list(e.path),
                )
            )
    elif {"schema_version", "id", "class", "implementation", "compute_policy"} <= set(doc):
        errors = sorted(reasoning_validator.iter_errors(doc), key=lambda e: list(e.path))
    else:
        continue

    validated += 1
    for error in errors:
        location = ".".join(str(part) for part in error.path) or "<root>"
        failures.append(f"{path.relative_to(ROOT)}:{location}: {error.message}")

if failures:
    print("\n".join(failures))
    raise SystemExit(1)

print(f"validated {validated} contract document(s)")
