"""G-06: every caliper readings file in fixtures/ matches the format in fixtures/README.md.

The ground-truth stories (stories/G/) record raw caliper readings, one file per story, as
fixtures/caliper_<story-id>.json. Those readings are the truth every later accuracy figure is
judged against (G-04), so a typo in one of them would quietly move every number built on top of
it. This file is what catches that before it lands: it checks every caliper_*.json against the
format, including that each summary figure (mean, min, max, standard deviation) really is the
summary of the readings beside it, and that each reading is a whole number of the instrument's
displayed steps.

The checks live in caliper_file_problems(), which returns a list of human-readable problems
rather than stopping at the first one, so a file with three mistakes reports all three. The
synthetic-file tests below pin down each check on its own; the parametrised test runs the same
function over the real files.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

# Straight from the field tables in fixtures/README.md. Exact key sets, not minimums: an extra
# key is as likely to be a misspelt required one as a deliberate addition, and the format has no
# optional fields.
TOP_LEVEL_KEYS = {"version", "story", "measured_by", "date", "instrument", "features"}
INSTRUMENT_KEYS = {"type", "resolution_mm", "accuracy_spec"}
FEATURE_KEYS = {
    "object",
    "kind",
    "cad_vertices",
    "nominal_mm",
    "nominal_source",
    "square_count",
    "where",
    "orientation_on_board",
    "readings_mm",
    "mean_mm",
    "min_mm",
    "max_mm",
    "std_mm",
}
KINDS = {"edge", "height", "board_span", "cube_axis"}
NOMINAL_SOURCES = {"cad", "D-034", "cube"}
READINGS_PER_FEATURE = 10  # the Ground truth protocol: 10 independent placements
HANDLE_RE = re.compile(r"^@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STORY_RE = re.compile(r"^G-\d\d$")

# Summary figures are written rounded to four decimals (fixtures/README.md), so they are compared
# with half of 0.0001 mm. Three decimals would not be enough: a caliper's spread is typically
# about 0.01 mm (PR #35's ten readings ranged 0.02 mm), and at that size the population standard
# deviation differs from the sample one by only about 0.0005 mm, the same as three-decimal
# rounding, so the wrong formula would pass unnoticed.
SUMMARY_TOLERANCE_MM = 0.00005


def caliper_file_problems(data: object, file_name: str) -> list[str]:
    """Everything wrong with one parsed caliper file, as messages. Empty when it is valid."""
    if not isinstance(data, dict):
        return [f"{file_name}: the top level must be a JSON object"]
    problems = _key_problems(data, TOP_LEVEL_KEYS, file_name)

    if data.get("version") != 1:
        problems.append(f"{file_name}: version must be 1, got {data.get('version')!r}")
    story = data.get("story")
    if not isinstance(story, str) or not STORY_RE.match(story):
        problems.append(f"{file_name}: story must look like 'G-06', got {story!r}")
    elif file_name != f"caliper_{story.lower()}.json":
        # One file per story, named after it (fixtures/README.md), so a reader can find a
        # story's readings without opening every file.
        problems.append(
            f"{file_name}: file name should be caliper_{story.lower()}.json for story {story}"
        )
    measured_by = data.get("measured_by")
    if not isinstance(measured_by, str) or not HANDLE_RE.match(measured_by):
        problems.append(
            f"{file_name}: measured_by must be an @github-handle, got {measured_by!r}"
        )
    date = data.get("date")
    if not isinstance(date, str) or not DATE_RE.match(date):
        problems.append(f"{file_name}: date must be YYYY-MM-DD, got {date!r}")

    resolution_mm = _instrument_problems(data.get("instrument"), file_name, problems)

    features = data.get("features")
    if not isinstance(features, list) or not features:
        problems.append(f"{file_name}: features must be a non-empty list")
        return problems
    for index, feature in enumerate(features):
        label = f"{file_name}: features[{index}]"
        if not isinstance(feature, dict):
            problems.append(f"{label} must be an object")
            continue
        problems.extend(_feature_problems(feature, label, resolution_mm))
    return problems


def _key_problems(obj: dict, expected: set[str], label: str) -> list[str]:
    problems = []
    for key in sorted(expected - obj.keys()):
        problems.append(f"{label}: missing {key}")
    for key in sorted(obj.keys() - expected):
        problems.append(
            f"{label}: unexpected {key} (not in fixtures/README.md's format)"
        )
    return problems


def _instrument_problems(instrument: object, file_name: str, problems: list[str]):
    """Append the instrument's problems. Returns its resolution when usable, else None."""
    label = f"{file_name}: instrument"
    if not isinstance(instrument, dict):
        problems.append(f"{label} must be an object")
        return None
    problems.extend(_key_problems(instrument, INSTRUMENT_KEYS, label))
    if not isinstance(instrument.get("type"), str) or not instrument.get("type"):
        problems.append(f"{label}.type must be a non-empty string")
    # Resolution is not accuracy (fixtures/README.md): the spec is the maker's stated figure, or
    # null when the instrument states none, which the Ground truth protocol says out loud.
    spec = instrument.get("accuracy_spec")
    if spec is not None and (not isinstance(spec, str) or not spec):
        problems.append(
            f"{label}.accuracy_spec must be the stated figure as text, or null"
        )
    resolution_mm = instrument.get("resolution_mm")
    if not _is_number(resolution_mm) or resolution_mm <= 0:
        problems.append(
            f"{label}.resolution_mm must be a positive number, got {resolution_mm!r}"
        )
        return None
    return float(resolution_mm)


def _feature_problems(
    feature: dict, label: str, resolution_mm: float | None
) -> list[str]:
    problems = _key_problems(feature, FEATURE_KEYS, label)

    if not isinstance(feature.get("object"), str) or not feature.get("object"):
        problems.append(f"{label}.object must be a non-empty string")
    kind = feature.get("kind")
    if kind not in KINDS:
        problems.append(f"{label}.kind must be one of {sorted(KINDS)}, got {kind!r}")
    if feature.get("nominal_source") not in NOMINAL_SOURCES:
        problems.append(
            f"{label}.nominal_source must be one of {sorted(NOMINAL_SOURCES)}, "
            f"got {feature.get('nominal_source')!r}"
        )
    nominal_mm = feature.get("nominal_mm")
    if not _is_number(nominal_mm) or nominal_mm <= 0:
        problems.append(
            f"{label}.nominal_mm must be a positive number, got {nominal_mm!r}"
        )
    if not isinstance(feature.get("where"), str) or not feature.get("where"):
        problems.append(f"{label}.where must say how it was read")
    orientation = feature.get("orientation_on_board")
    if orientation is not None and (
        not isinstance(orientation, str) or not orientation
    ):
        problems.append(f"{label}.orientation_on_board must be text or null")

    # An edge is named by the CAD vertex pair it runs between (the Ground truth protocol);
    # nothing else has one, so a pair on anything else is a sign the kind is wrong.
    cad_vertices = feature.get("cad_vertices")
    if kind == "edge":
        if not _is_vertex_pair(cad_vertices):
            problems.append(
                f"{label}.cad_vertices must be [[x, y], [x, y]] for an edge"
            )
    elif cad_vertices is not None:
        problems.append(f"{label}.cad_vertices must be null unless kind is edge")

    square_count = feature.get("square_count")
    if kind == "board_span":
        if not isinstance(square_count, int) or isinstance(square_count, bool):
            problems.append(
                f"{label}.square_count must be a whole number for a board_span"
            )
        elif square_count < 1:
            problems.append(f"{label}.square_count must be at least 1")
    elif square_count is not None:
        problems.append(f"{label}.square_count must be null unless kind is board_span")

    problems.extend(_readings_problems(feature, label, resolution_mm))
    return problems


def _is_vertex_pair(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    for vertex in value:
        if not isinstance(vertex, list) or len(vertex) != 2:
            return False
        for coordinate in vertex:
            if not _is_number(coordinate):
                return False
    return True


def _readings_problems(
    feature: dict, label: str, resolution_mm: float | None
) -> list[str]:
    readings = feature.get("readings_mm")
    if not isinstance(readings, list):
        return [f"{label}.readings_mm must be a list"]
    problems = []
    if len(readings) != READINGS_PER_FEATURE:
        problems.append(
            f"{label}: needs exactly {READINGS_PER_FEATURE} readings, got {len(readings)}"
        )
    for reading in readings:
        if not _is_number(reading) or reading <= 0:
            problems.append(
                f"{label}.readings_mm has a non-positive or non-number {reading!r}"
            )
            return problems
    if len(readings) < 2:
        return problems  # nothing below can be computed

    # A displayed reading is a whole number of the instrument's steps. Anything between steps
    # can't have come off the display, so it's a typing slip.
    if resolution_mm is not None:
        for reading in readings:
            steps = reading / resolution_mm
            if abs(steps - round(steps)) > 1e-6:
                problems.append(
                    f"{label}: reading {reading} is not a whole number of "
                    f"{resolution_mm} mm steps (the instrument's resolution)"
                )

    # Each summary figure has to be the summary of these readings. The sample standard deviation,
    # as the format says, not the population one.
    expected = {
        "mean_mm": statistics.fmean(readings),
        "min_mm": min(readings),
        "max_mm": max(readings),
        "std_mm": statistics.stdev(readings),
    }
    for key, value in expected.items():
        stated = feature.get(key)
        if not _is_number(stated):
            problems.append(f"{label}.{key} must be a number, got {stated!r}")
        elif abs(stated - value) > SUMMARY_TOLERANCE_MM:
            problems.append(
                f"{label}.{key} is {stated}, but its readings give {value:.4f}"
            )
    return problems


def _is_number(value: object) -> bool:
    # bool is a subclass of int in Python, and a true/false where a millimetre value belongs is a
    # mistake, not a number.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --- a valid synthetic file, then one broken thing at a time ----------------------------------


def _feature(**overrides: object) -> dict:
    readings = [20.01, 20.02, 20.0, 20.01, 20.03, 20.01, 20.02, 20.0, 20.01, 20.02]
    feature = {
        "object": "board",
        "kind": "board_span",
        "cad_vertices": None,
        "nominal_mm": 20.0,
        "nominal_source": "D-034",
        "square_count": 1,
        "where": "corner to corner along a row",
        "orientation_on_board": None,
        "readings_mm": readings,
        "mean_mm": statistics.fmean(readings),
        "min_mm": min(readings),
        "max_mm": max(readings),
        "std_mm": statistics.stdev(readings),
    }
    feature.update(overrides)
    return feature


def _edge_feature(**overrides: object) -> dict:
    feature = _feature(
        object="s-04",
        kind="edge",
        cad_vertices=[[0.0, 0.0], [60.0, 0.0]],
        nominal_mm=60.0,
        nominal_source="cad",
        square_count=None,
        where="mid-height, corner to corner",
    )
    feature.update(overrides)
    return feature


def _valid_file(**overrides: object) -> dict:
    data = {
        "version": 1,
        "story": "G-06",
        "measured_by": "@member",
        "date": "2026-10-26",
        "instrument": {
            "type": "digital caliper",
            "resolution_mm": 0.01,
            "accuracy_spec": "±0.02 mm",
        },
        "features": [_feature(), _edge_feature()],
    }
    data.update(overrides)
    return data


def _problems_mentioning(
    data: dict, text: str, file_name: str = "caliper_g-06.json"
) -> list:
    problems = caliper_file_problems(data, file_name)
    return [problem for problem in problems if text in problem]


def test_a_valid_file_has_no_problems():
    assert caliper_file_problems(_valid_file(), "caliper_g-06.json") == []


def test_a_mean_that_does_not_match_its_readings_is_reported():
    broken = _valid_file(features=[_feature(mean_mm=20.5)])
    assert _problems_mentioning(broken, "mean_mm")


def test_a_standard_deviation_from_the_wrong_formula_is_reported():
    """The format asks for the sample standard deviation. The population one is a plausible slip
    and differs by about 5% at 10 readings, far more than rounding."""
    readings = _feature()["readings_mm"]
    broken = _valid_file(features=[_feature(std_mm=statistics.pstdev(readings))])
    assert _problems_mentioning(broken, "std_mm")


def test_min_and_max_must_match_the_readings():
    broken = _valid_file(features=[_feature(min_mm=19.0, max_mm=21.0)])
    assert _problems_mentioning(broken, "min_mm")
    assert _problems_mentioning(broken, "max_mm")


def test_exactly_ten_readings_are_required():
    nine = _feature()["readings_mm"][:9]
    broken = _valid_file(
        features=[
            _feature(
                readings_mm=nine,
                mean_mm=statistics.fmean(nine),
                min_mm=min(nine),
                max_mm=max(nine),
                std_mm=statistics.stdev(nine),
            )
        ]
    )
    assert _problems_mentioning(broken, "10 readings")


def test_a_reading_between_the_displayed_steps_is_reported():
    """A 0.01 mm caliper can't display 20.015. A value like that is a typing slip."""
    readings = [20.015] + _feature()["readings_mm"][1:]
    broken = _valid_file(
        features=[
            _feature(
                readings_mm=readings,
                mean_mm=statistics.fmean(readings),
                min_mm=min(readings),
                max_mm=max(readings),
                std_mm=statistics.stdev(readings),
            )
        ]
    )
    assert _problems_mentioning(broken, "resolution")


def test_an_unknown_kind_or_nominal_source_is_reported():
    broken = _valid_file(features=[_feature(kind="width", nominal_source="guess")])
    assert _problems_mentioning(broken, "kind")
    assert _problems_mentioning(broken, "nominal_source")


def test_an_edge_needs_its_cad_vertex_pair_and_nothing_else_has_one():
    missing = _valid_file(features=[_edge_feature(cad_vertices=None)])
    assert _problems_mentioning(missing, "cad_vertices")
    stray = _valid_file(features=[_feature(cad_vertices=[[0, 0], [20, 0]])])
    assert _problems_mentioning(stray, "cad_vertices")


def test_a_board_span_needs_a_square_count_and_nothing_else_has_one():
    missing = _valid_file(features=[_feature(square_count=None)])
    assert _problems_mentioning(missing, "square_count")
    stray = _valid_file(features=[_edge_feature(square_count=3)])
    assert _problems_mentioning(stray, "square_count")


def test_missing_and_misspelt_keys_are_both_reported():
    feature = _feature()
    feature["were"] = feature.pop("where")
    broken = _valid_file(features=[feature])
    assert _problems_mentioning(broken, "missing where")
    assert _problems_mentioning(broken, "unexpected were")


def test_the_file_name_must_match_the_story():
    assert _problems_mentioning(
        _valid_file(), "file name", file_name="caliper_g-01.json"
    )


def test_top_level_fields_are_checked():
    broken = _valid_file(version=2, measured_by="member", date="26/10/2026")
    assert _problems_mentioning(broken, "version")
    assert _problems_mentioning(broken, "measured_by")
    assert _problems_mentioning(broken, "date")


def test_resolution_is_required_and_accuracy_may_be_null():
    no_resolution = _valid_file(
        instrument={
            "type": "digital caliper",
            "resolution_mm": 0,
            "accuracy_spec": None,
        }
    )
    assert _problems_mentioning(no_resolution, "resolution_mm")
    no_spec = _valid_file(
        instrument={
            "type": "digital caliper",
            "resolution_mm": 0.01,
            "accuracy_spec": None,
        }
    )
    assert caliper_file_problems(no_spec, "caliper_g-06.json") == []


# --- the real files ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", sorted(FIXTURES_DIR.glob("caliper_*.json")), ids=lambda path: path.name
)
def test_every_caliper_file_matches_the_format(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert caliper_file_problems(data, path.name) == []
