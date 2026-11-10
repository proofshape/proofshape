from __future__ import annotations

import base64
from pathlib import Path

from scripts.generate_c14_fixture import generate


def test_c14_fixture_matches_generator_byte_for_byte(tmp_path, capsys):
    generated = tmp_path / "model-deviation.glb"
    generate(generated)

    fixture = Path("capture/stub/fixtures/model-deviation.glb")
    actual = generated.read_bytes()
    expected = fixture.read_bytes()

    if actual != expected:
        print(
            "GENERATED_C14_FIXTURE_BASE64="
            + base64.b64encode(actual).decode("ascii")
        )

    assert actual == expected
