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
replay_schema = json.loads((SCHEMAS / "replay-spec.schema.json").read_text())
substrate_requirements_schema = json.loads(
    (SCHEMAS / "substrate-requirements.schema.json").read_text()
)

for schema in (
    reasoning_schema,
    run_schema,
    benchmark_source_schema,
    replay_schema,
    substrate_requirements_schema,
):
    Draft202012Validator.check_schema(schema)

# Resolve the repository-local reasoning-substrate reference without requiring
# network access or a schema registry.
run_schema_resolved = json.loads(json.dumps(run_schema))
run_schema_resolved["properties"]["reasoning_substrate"] = reasoning_schema

reasoning_validator = Draft202012Validator(reasoning_schema)
run_validator = Draft202012Validator(run_schema_resolved)
benchmark_source_validator = Draft202012Validator(benchmark_source_schema)
replay_validator = Draft202012Validator(replay_schema)
substrate_requirements_validator = Draft202012Validator(substrate_requirements_schema)

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

replay_dir = ROOT / "replays"
if replay_dir.exists():
    seen_replay_ids = set()
    for path in sorted(replay_dir.glob("*.json")):
        spec = json.loads(path.read_text())
        errors = sorted(replay_validator.iter_errors(spec), key=lambda e: list(e.path))
        validated += 1

        for error in errors:
            location = ".".join(str(part) for part in error.path) or "<root>"
            failures.append(f"{path.relative_to(ROOT)}:{location}: {error.message}")

        replay_id = spec.get("id")
        if replay_id in seen_replay_ids:
            failures.append(f"{path.relative_to(ROOT)}: duplicate replay id {replay_id}")
        seen_replay_ids.add(replay_id)

        environment_ref = spec.get("environment_ref")
        if isinstance(environment_ref, str):
            env_path = ROOT / environment_ref
            if not env_path.is_file():
                failures.append(
                    f"{path.relative_to(ROOT)}: environment_ref does not exist: "
                    f"{environment_ref}"
                )
            else:
                environment = json.loads(env_path.read_text())
                requirements_file = environment.get("requirements_file")
                if not isinstance(requirements_file, str):
                    failures.append(
                        f"{env_path.relative_to(ROOT)}: requirements_file missing"
                    )
                elif not (ROOT / requirements_file).is_file():
                    failures.append(
                        f"{env_path.relative_to(ROOT)}: requirements file does not exist: "
                        f"{requirements_file}"
                    )

        module = spec.get("module")
        if isinstance(module, str) and module.startswith("experiments."):
            module_path = ROOT / (module.replace(".", "/") + ".py")
            if not module_path.is_file():
                failures.append(
                    f"{path.relative_to(ROOT)}: experiment module does not exist: "
                    f"{module}"
                )

profiles_dir = ROOT / "profiles"
if profiles_dir.exists():
    seen_profile_ids = set()
    for path in sorted(profiles_dir.glob("*.json")):
        profile = json.loads(path.read_text())
        errors = sorted(
            substrate_requirements_validator.iter_errors(profile),
            key=lambda e: list(e.path),
        )
        validated += 1
        for error in errors:
            location = ".".join(str(part) for part in error.path) or "<root>"
            failures.append(f"{path.relative_to(ROOT)}:{location}: {error.message}")

        profile_id = profile.get("id")
        if profile_id in seen_profile_ids:
            failures.append(f"{path.relative_to(ROOT)}: duplicate profile id {profile_id}")
        seen_profile_ids.add(profile_id)

        req = profile.get("requirements", {})
        if profile.get("resource_class") == "gpu":
            if not req.get("nvidia_required"):
                failures.append(
                    f"{path.relative_to(ROOT)}: GPU profile must require NVIDIA visibility"
                )
            if req.get("min_gpu_count", 0) < 1:
                failures.append(
                    f"{path.relative_to(ROOT)}: GPU profile must require at least one GPU"
                )

if failures:
    print("\n".join(failures))
    raise SystemExit(1)

print(f"validated {validated} contract document(s)")
