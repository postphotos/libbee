"""Unit tests for libbee.inflation — the nominal→real (CPI-U) deflator."""

from __future__ import annotations

import polars as pl
import pytest

from libbee import inflation as inf


def test_deflator_base_is_identity():
    """A dollar in the base year is worth exactly one base-year dollar."""
    assert inf.deflator(2023, base=2023) == 1.0


def test_deflator_older_dollars_scale_up():
    """Pre-base years deflate to >1 (an old dollar buys more of today's), matching CPI[base]/CPI[year]."""
    d = inf.deflator(2007, base=2023)
    assert d > 1.0
    assert d == pytest.approx(inf.CPI_U[2023] / inf.CPI_U[2007])


def test_deflator_unknown_years_raise():
    with pytest.raises(KeyError):
        inf.deflator(1850)
    with pytest.raises(KeyError):
        inf.deflator(2019, base=1850)


def test_deflate_frame_converts_to_constant_dollars():
    """deflate() multiplies the named columns by each row's deflator, leaving others untouched."""
    df = pl.DataFrame({"year": [2007, 2023], "funding_pc": [100.0, 100.0], "visits_pc": [5.0, 5.0]})
    out = inf.deflate(df, cols=["funding_pc"], base=2023)
    # 2023 row unchanged (base); 2007 row scaled up by its deflator; visits_pc passes through
    row07 = out.filter(pl.col("year") == 2007).row(0, named=True)
    row23 = out.filter(pl.col("year") == 2023).row(0, named=True)
    assert row23["funding_pc"] == pytest.approx(100.0)
    assert row07["funding_pc"] == pytest.approx(100.0 * inf.deflator(2007, base=2023))
    assert row07["visits_pc"] == 5.0 and "_defl" not in out.columns
