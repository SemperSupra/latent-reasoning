#!/usr/bin/env python3
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"

reasoning_schema = json.loads((SCHEMAS / "reasoning-substrate.schema.json").read_text())
run_schema = json.loads((SCHEMAS / "run-manifest.schema.json").read_text())
benchmark_source_schema = json.loads(
    (SCHEMAS / "benchmark-source-registry.schema.json").read_text()
)

Draft202012Validator.check_schema(reasoning_schema)
Draft202012Validator.check_schema(run_schema)
Draft202012Validator.check_schema(benchmark_source_schema)

# Resolve the repository-local reasoning-substrate reference without requiring
# network access or a schema registry.
run_schema_resolved = json.loads(json.dumps(run_schema))
run_schema_resolved["properties"]["reasoning_substrate"] = reasoning_schema

reasoning_validator = Draft202012Validator(reasoning_schema)
run_validator = Draft202012Validator(run_schema_resolved)
benchmark_source_validator = Draft202012Validator(benchmark_source_schema)

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

registry_path = ROOT / "benchmarks" / "sources.json"
if registry_path.exists():
    registry = json.loads(registry_path.read_text())
    errors = sorted(
        benchmark_source_validator.iter_errors(registry),
        key=lambda e: list(e.path),
    )
    validated += 1
    for error in errors:
        location = ".".join(str(part) for part in error.path) or "<root>"
        failures.append(f"{registry_path.relative_to(ROOT)}:{location}: {error.message}")

    seen_ids = set()
    for source in registry.get("sources", []):
        source_id = source["id"]
        if source_id in seen_ids:
            failures.append(f"benchmarks/sources.json: duplicate source id {source_id}")
        seen_ids.add(source_id)

        license_class = source["license"]["class"]
        posture = source["integration"]["posture"]
        vendoring = source["integration"]["vendoring_allowed"]
        module_audit = source["integration"]["module_audit_required"]

        if license_class in {"B", "C", "D", "E"} and vendoring:
            failures.append(
                f"benchmarks/sources.json:{source_id}: "
                f"class {license_class} source cannot be vendor-enabled by default"
            )
        if license_class == "C" and not module_audit:
            failures.append(
                f"benchmarks/sources.json:{source_id}: "
                "mixed-license source requires module audit"
            )
        if license_class == "E" and posture != "methodology-only":
            failures.append(
                f"benchmarks/sources.json:{source_id}: "
                "unresolved-license source must remain methodology-only"
            )

if failures:
    print("\n".join(failures))
    raise SystemExit(1)

print(f"validated {validated} contract document(s)")
