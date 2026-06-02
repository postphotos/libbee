"""Shared helpers for the analysis cuts (private)."""

from __future__ import annotations

import polars as pl


def _pair_years(panel, year_a, year_b, value_cols, *, key="fips"):
    """Helper: per-``key`` wide join of two years' value columns (suffix 0/1)."""
    keep = [key, "year", *value_cols]
    j = panel.filter(pl.col("year").is_in([year_a, year_b])).select(keep)
    a = j.filter(pl.col("year") == year_a).select([key, *[pl.col(c).alias(f"{c}0") for c in value_cols]])
    b = j.filter(pl.col("year") == year_b).select([key, *[pl.col(c).alias(f"{c}1") for c in value_cols]])
    return a.join(b, on=key, how="inner")
