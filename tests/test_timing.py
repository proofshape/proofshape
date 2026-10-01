"""Unit tests for R-13's per-stage timing table. Pure dict-in/string-out -- no GPU, no fixtures."""

from __future__ import annotations

import pytest

from recon.timing import format_timing_table


def test_renders_every_stage_and_a_share_column():
    table = format_timing_table(
        {"model_load_s": 1.0, "inference_s": 3.0, "total_s": 4.0}
    )
    assert "model_load_s" in table
    assert "inference_s" in table
    assert "total_s" in table
    assert "share" in table
    # model_load_s is 1/4 of total, inference_s is 3/4.
    assert "25.0%" in table
    assert "75.0%" in table
    assert "100.0%" in table


def test_stage_order_is_preserved_not_sorted():
    table = format_timing_table(
        {"z_last_added": 1.0, "a_first_added": 1.0, "total_s": 2.0}
    )
    assert table.index("z_last_added") < table.index("a_first_added")


def test_raises_without_total_s():
    with pytest.raises(ValueError, match="total_s"):
        format_timing_table({"model_load_s": 1.0})


def test_zero_total_does_not_divide_by_zero():
    table = format_timing_table({"model_load_s": 0.0, "total_s": 0.0})
    assert "0.0%" in table


def test_column_width_adapts_to_longest_stage_name():
    short = format_timing_table({"a": 1.0, "total_s": 1.0})
    long_name = "a_much_longer_stage_name_than_usual_s"
    long = format_timing_table({long_name: 1.0, "total_s": 1.0})
    # The long name's own line should not be truncated, and the table should be wider than the
    # short-named one's.
    assert long_name in long
    assert len(long.splitlines()[0]) > len(short.splitlines()[0])


def test_single_total_only_stage():
    table = format_timing_table({"total_s": 2.5})
    assert "total_s" in table
    assert "100.0%" in table
