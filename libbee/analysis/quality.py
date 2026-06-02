"""Data-quality cuts — the zero/missing scan and the coverage denominator."""

from __future__ import annotations

import polars as pl

_QA_CHECKS = {
    "county_equity": ["visits", "funding", "hours", "popu_lsa"],
    "state_panel": ["visits", "funding", "popu_lsa"],
    "county_panel": ["visits", "funding", "popu_lsa"],
}


def qa_report(frames) -> pl.DataFrame:
    """Data-quality scan over the loaded ``frames`` dict: count zero/missing values in metrics that
    should be positive for an operating system (the kind of artifact that made Hawaii read $0-funded).
    Returns ``[frame, column, zero_or_null, total, pct]`` worst-first — surfacing issues at build time
    instead of in a chart.

    Example::

        d = libbee.frames.load()
        libbee.analysis.qa_report({"county_equity": d["county_equity"], "state_panel": d["state_panel"]})
        # the 2019 cross-section is clean (0%); historical zeros are pre-1998 metrics that don't exist yet
    """
    rows = []
    for name, cols in _QA_CHECKS.items():
        df = frames.get(name)
        if df is None:
            continue
        for c in cols:
            if c not in df.columns:
                continue
            z = df.filter((pl.col(c) <= 0) | pl.col(c).is_null()).height
            rows.append({"frame": name, "column": c, "zero_or_null": z, "total": df.height, "pct": round(z / df.height * 100, 1) if df.height else 0.0})
    return pl.DataFrame(rows).sort("pct", descending=True)


def coverage_report(county_equity, *, us_counties=3143, us_pop=328_239_523, year=2019) -> dict:
    """What share of the country the county cross-section actually represents. IMLS only reaches counties
    with a reporting library system, so the modelled set is a subset of all ~3,143 US county-equivalents
    (``us_counties``) and of the ~328M US population (``us_pop``, 2019 Census). Returns the county count
    and the share of counties and of population covered — the denominator behind every 'counties' figure.

    Example::

        cov = libbee.analysis.coverage_report(libbee.load("county_equity"))
        f"{cov['counties']:,} of {cov['us_counties']:,} counties ({cov['pct_counties']}%)"
        # -> '2,758 of 3,143 counties (87.8%)', covering ~cov['pct_pop']% of the US population
    """
    n = county_equity["fips"].n_unique()
    pop = float(county_equity["total_pop"].sum())
    return {
        "year": year,
        "counties": n,
        "us_counties": us_counties,
        "pct_counties": round(n / us_counties * 100, 1),
        "pop_covered": pop,
        "us_pop": us_pop,
        "pct_pop": round(pop / us_pop * 100, 1),
    }
