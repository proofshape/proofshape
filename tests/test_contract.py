"""Tests for contracts/openapi.yaml (F-03).

Written alongside the spec, section by section, per D-029 — not batched at the end. Each test
guards one piece of the acceptance criteria or one decision the spec is required to encode.
"""

import json
import re
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft7Validator, RefResolver

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "contracts" / "openapi.yaml"
EXAMPLES_DIR = REPO_ROOT / "contracts" / "examples"

# Matches both the plain convention (createSession.request.json,
# createSession.response-201.json) and a variant example that adds a descriptive slug
# (getVerdict.response-200-rescan-unobserved.json) for a second, non-happy-path case.
EXAMPLE_FILENAME_RE = re.compile(
    r"^(?P<op>[A-Za-z0-9]+)\.(?:request|response-(?P<status>\d{3}))(?:-[a-z0-9-]+)?\.json$"
)


@pytest.fixture(scope="module")
def spec() -> dict:
    with open(CONTRACT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_operations(spec: dict):
    """Yield (operation_id, method, path, operation_dict) for every operation in the spec."""
    for path, path_item in spec["paths"].items():
        for method, operation in path_item.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            yield operation["operationId"], method, path, operation


def _request_schema(operation: dict) -> dict | None:
    request_body = operation.get("requestBody")
    if not request_body:
        return None
    for media in request_body.get("content", {}).values():
        return media.get("schema")
    return None


def _response_schemas(operation: dict) -> dict:
    """Map status code (str) -> schema, for every response that documents a body."""
    result = {}
    for status, response in operation.get("responses", {}).items():
        for media in response.get("content", {}).values():
            result[status] = media.get("schema")
            break
    return result


def _parse_example_filenames():
    """Return (path, operation_id, kind, status) for every file under contracts/examples."""
    parsed = []
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        match = EXAMPLE_FILENAME_RE.match(path.name)
        if match is None:
            parsed.append((path, None, None, None))
            continue
        kind = "response" if match.group("status") else "request"
        parsed.append((path, match.group("op"), kind, match.group("status")))
    return parsed


def test_spec_is_valid_openapi(spec):
    """contracts/openapi.yaml must be a structurally valid OpenAPI 3.1 document."""
    try:
        from openapi_spec_validator import validate as validate_spec
    except ImportError:  # pragma: no cover - older releases exposed a different name
        from openapi_spec_validator import validate_spec
    validate_spec(spec)


def test_every_operation_has_an_example(spec):
    """Acceptance criterion 2: one example request and response per endpoint."""
    examples = _parse_example_filenames()
    by_op_kind_status = {}
    for _path, op, kind, status in examples:
        by_op_kind_status.setdefault((op, kind, status), []).append(_path)

    missing = []
    for op_id, _method, _path, operation in _iter_operations(spec):
        if (
            _request_schema(operation) is not None
            and (op_id, "request", None) not in by_op_kind_status
        ):
            missing.append(f"{op_id}: no request example")
        for status in _response_schemas(operation):
            if (op_id, "response", status) not in by_op_kind_status:
                missing.append(f"{op_id}: no response-{status} example")

    assert not missing, "missing examples:\n" + "\n".join(missing)


def test_examples_validate_against_their_schema(spec):
    """Every committed example must satisfy the schema its own endpoint declares."""
    schemas_by_op = {}
    for op_id, _method, _path, operation in _iter_operations(spec):
        schemas_by_op[op_id] = {
            "request": _request_schema(operation),
            "responses": _response_schemas(operation),
        }

    resolver = RefResolver.from_schema(spec)
    checked = 0
    for path, op_id, kind, status in _parse_example_filenames():
        if op_id is None:
            continue  # covered separately by test_no_example_file_is_orphaned
        op_schemas = schemas_by_op.get(op_id)
        if op_schemas is None:
            continue
        schema = (
            op_schemas["request"]
            if kind == "request"
            else op_schemas["responses"].get(status)
        )
        if schema is None:
            continue
        with open(path, encoding="utf-8") as f:
            instance = json.load(f)
        Draft7Validator(schema, resolver=resolver).validate(instance)
        checked += 1

    assert checked > 0, (
        "no examples were actually validated — check the naming convention"
    )


def test_unobserved_regions_never_contribute_to_a_verdict():
    """D-005, the hard rule, made structural: enforced over every committed verdict example."""
    verdict_examples = list(EXAMPLES_DIR.glob("getVerdict.response-*.json"))
    assert verdict_examples, "expected at least one getVerdict example to check"

    violations = []
    for path in verdict_examples:
        with open(path, encoding="utf-8") as f:
            body = json.load(f)
        for region in body.get("regions", []):
            if (
                region["provenance"] == "unobserved"
                and region["contributes_to_verdict"] is not False
            ):
                violations.append(f"{path.name}: region {region['name']!r}")

    assert not violations, (
        "unobserved region(s) marked as contributing to a verdict — this violates the hard "
        "rule (D-005): " + ", ".join(violations)
    )


def test_verdict_enum_is_exactly_the_four_values(spec):
    """A fifth verdict value must fail the build, not silently pass through."""
    verdict_schema = spec["components"]["schemas"]["Verdict"]
    assert verdict_schema["enum"] == ["Pass", "Fail", "Rescan", "Unverifiable"]


def test_no_example_file_is_orphaned(spec):
    """Every example file must map to a real operationId, so a rename can't leave dead files."""
    known_op_ids = {op_id for op_id, _m, _p, _o in _iter_operations(spec)}

    orphans = []
    for path, op_id, _kind, _status in _parse_example_filenames():
        if op_id is None or op_id not in known_op_ids:
            orphans.append(path.name)

    assert not orphans, "example file(s) with no matching operationId: " + ", ".join(
        orphans
    )
