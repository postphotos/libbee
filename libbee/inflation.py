"""Turn nominal dollars into constant (real) dollars.

Every dollar figure in the IMLS/HUD data is **nominal** — a 2007 dollar and a 2023 dollar wear the same
label but don't buy the same thing. Comparing `funding_pc` across years without deflating credits
inflation as if it were real investment. This module owns one job: a published **CPI-U** index plus the
deflator math to express any year's dollars in a chosen base year's dollars, via
$\\text{real}_y = \\text{nominal}_y \\cdot \\mathrm{CPI}_{base}/\\mathrm{CPI}_y$.

The public surface is:

- `CPI_U` — the index table, BLS series CUUR0000SA0 (1982-84 = 100), annual averages.
- `BASE_YEAR` — the latest year in the table, the natural default base for "today's dollars".
- `deflator` — the scalar multiplier $\\mathrm{CPI}_{base}/\\mathrm{CPI}_y$ for one year.
- `deflate` — apply that multiplier row by row across a Polars frame.

Source: U.S. Bureau of Labor Statistics, **CPI-U, U.S. city average, all items** (1982-84 = 100),
annual averages. These are public, fixed, historical values — no network call, fully reproducible.
"""

from __future__ import annotations

import polars as pl

# CPI-U (1982-84 = 100), annual average. BLS series CUUR0000SA0.
CPI_U: dict[int, float] = {
    1992: 140.3,
    1993: 144.5,
    1994: 148.2,
    1995: 152.4,
    1996: 156.9,
    1997: 160.5,
    1998: 163.0,
    1999: 166.6,
    2000: 172.2,
    2001: 177.1,
    2002: 179.9,
    2003: 184.0,
    2004: 188.9,
    2005: 195.3,
    2006: 201.6,
    2007: 207.342,
    2008: 215.303,
    2009: 214.537,
    2010: 218.056,
    2011: 224.939,
    2012: 229.594,
    2013: 232.957,
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2020: 258.811,
    2021: 270.970,
    2022: 292.655,
    2023: 304.702,
    2024: 313.689,
}

#: The most recent year with a published index — the natural default base for "today's dollars".
BASE_YEAR: int = max(CPI_U)


def deflator(year: int, *, base: int = BASE_YEAR) -> float:
    """Compute the multiplier that converts a nominal `year` dollar into a constant `base` dollar.

    Returns the scalar deflator so that $\\text{real} = \\text{nominal} \\cdot d$ where
    $d = \\mathrm{CPI}_{base}/\\mathrm{CPI}_{year}$. It is **>1** for years before the base (older dollars
    scale **up** to today's), **<1** for years after the base, and exactly $1.0$ when `year == base`.

    Args:
        year: A year present in `CPI_U`; the year whose nominal dollars are being converted.
        base: A year present in `CPI_U` to express dollars in; defaults to `BASE_YEAR`.

    Returns:
        The deflation factor $\\mathrm{CPI}_{base}/\\mathrm{CPI}_{year}$ as a `float`.

    Raises:
        KeyError: when `year` or `base` has no published CPI-U index in `CPI_U`.

    Example:
        ```python
        deflator(2007)            # ~1.51 — a 2007 dollar in latest-year dollars
        deflator(2007, base=2019) # express a 2007 dollar in 2019 dollars instead
        ```
    """
    if year not in CPI_U:
        raise KeyError(f"no CPI-U index for {year} (have {min(CPI_U)}–{max(CPI_U)})")
    if base not in CPI_U:
        raise KeyError(f"no CPI-U index for base year {base} (have {min(CPI_U)}–{max(CPI_U)})")
    return CPI_U[base] / CPI_U[year]


def deflate(df: pl.DataFrame, *, cols: list[str], year: str = "year", base: int = BASE_YEAR) -> pl.DataFrame:
    """Convert nominal dollar columns to constant-`base` dollars, row by row, by each row's `year`.

    For every column `c` in `cols`, replaces the value with
    $\\text{real}_y = \\text{nominal}_y \\cdot \\mathrm{CPI}_{base}/\\mathrm{CPI}_y$, where $y$ is the
    value in the `year` column for that row. The deflator is **left-joined** from `CPI_U`, so:

    - Other columns pass through **untouched**.
    - Within-year rankings are unchanged — only cross-year **levels** become comparable.
    - Any year absent from `CPI_U` yields a `null` deflator (hence `null` output) rather than a silent
      wrong number.

    Args:
        df: A Polars `DataFrame` containing the `year` column and every column named in `cols`.
        cols: Names of the nominal-dollar columns to deflate in place.
        year: Name of the integer year column to join the index on; defaults to `"year"`.
        base: A year present in `CPI_U` to express dollars in; defaults to `BASE_YEAR`.

    Returns:
        A new `DataFrame` with the same columns as `df`, where each column in `cols` now holds
        constant-`base` dollars and the temporary `_defl` helper column has been dropped.

    Example:
        ```python
        cp = libbee.load("county_panel")                            # funding_pc is NOMINAL
        real = libbee.inflation.deflate(cp, cols=["funding_pc"])    # now constant BASE_YEAR dollars
        libbee.inflation.deflate(cp, cols=["funding_pc"], base=2019)  # ...or a different base year
        ```
    """
    factor = pl.DataFrame({year: list(CPI_U), "_defl": [CPI_U[base] / CPI_U[y] for y in CPI_U]}).with_columns(pl.col(year).cast(df.schema[year]))
    return df.join(factor, on=year, how="left").with_columns(*[(pl.col(c) * pl.col("_defl")).alias(c) for c in cols]).drop("_defl")
