"""Homelessness cuts (§6) — the scatter + the hours×spend cohort heatmap."""

from __future__ import annotations

import polars as pl

from ._common import _pair_years


def homeless_scatter(county_equity, county_homeless, *, year=2019) -> tuple[pl.DataFrame, dict]:
    """§6a — county homelessness rate vs visits_pc, with a closed-form OLS line. Returns
    ``(points[fips,homeless_per10k,visits_pc], fit{slope,intercept,r,x0,x1,year})``. r will be ≈0.

    Example::

        pts, fit = libbee.analysis.homeless_scatter(county_equity, county_homeless, year=2019)
        round(fit["r"], 2)        # ~0.07 — the "libraries are just shelters" story fails
        # draw the fit line between (fit["x0"], …) and (fit["x1"], …) using fit["slope"]/["intercept"]
    """
    ch = county_homeless.filter(pl.col("year") == year).select(["fips", "homeless_per10k"])
    pts = county_equity.join(ch, on="fips", how="inner").filter((pl.col("homeless_per10k") > 0) & (pl.col("visits_pc") > 0)).select(["fips", "homeless_per10k", "visits_pc"])
    s = pts.select(
        pl.corr("homeless_per10k", "visits_pc").alias("r"),
        pl.cov("homeless_per10k", "visits_pc").alias("cov"),
        pl.var("homeless_per10k").alias("vx"),
        pl.mean("homeless_per10k").alias("mx"),
        pl.mean("visits_pc").alias("my"),
        pl.min("homeless_per10k").alias("x0"),
        pl.max("homeless_per10k").alias("x1"),
    ).row(0, named=True)
    slope = s["cov"] / s["vx"] if s["vx"] else 0.0
    fit = {
        "slope": slope,
        "intercept": s["my"] - slope * s["mx"],
        "r": s["r"],
        "x0": s["x0"],
        "x1": s["x1"],
        "year": year,
    }
    return pts, fit


def homeless_cohorts(county_panel, county_homeless, *, year_a=2018, year_b=2023, thresh=0.05) -> pl.DataFrame:
    """§6b — the 2×3 cohort heatmap that separates *open hours* from *homeless spending* as drivers of use.

    Takes the SAME counties and splits them two ways over ``[year_a, year_b]``: by what happened to their
    open **hours** (fewer / same / more) and to their **homeless grant dollars** (less / more), then reports
    how each cell's library visits moved. The load-bearing trick is ``visits_chg_demeaned`` — each cohort's
    change **minus the all-county mean change** — which cancels the giant COVID common shock so the cells
    show *relative* performance instead of "everyone fell in 2020." Returns
    ``[hcat, scat, visits_chg, visits_chg_demeaned, p25, p75, n]`` (``p25``/``p75`` = the within-cohort
    spread of the visit change). The story it tells: the **hours** rows separate sharply; the **spending**
    rows barely do — hours is the lever, not homeless money.

    Example::

        coh = libbee.analysis.homeless_cohorts(county_panel, county_homeless, year_a=2018, year_b=2023)
        coh.pivot("scat", index="hcat", values="visits_chg_demeaned")   # the 3×2 heatmap, as a grid
        # chart: x="hcat" (fewer/same/more hours), y="scat" (less/more $), colour="visits_chg_demeaned"
    """
    cp = county_panel.select(["fips", "year", "hours_pc", "visits_pc"])
    ch = county_homeless.select(["fips", "year", "spend_pc"])
    j = cp.join(ch, on=["fips", "year"], how="inner")
    w = (
        _pair_years(j, year_a, year_b, ["hours_pc", "visits_pc", "spend_pc"])
        .filter((pl.col("hours_pc0") > 0) & (pl.col("spend_pc0") > 0) & (pl.col("visits_pc0") > 0) & (pl.col("visits_pc1") > 0))
        .with_columns(
            (pl.col("hours_pc1") / pl.col("hours_pc0") - 1).alias("hchg"),
            (pl.col("spend_pc1") / pl.col("spend_pc0") - 1).alias("schg"),
            (pl.col("visits_pc1") / pl.col("visits_pc0") - 1).alias("vchg"),
        )
        .with_columns(
            pl.when(pl.col("hchg") > thresh).then(pl.lit("more hours")).when(pl.col("hchg") < -thresh).then(pl.lit("fewer hours")).otherwise(pl.lit("same hours")).alias("hcat"),
            pl.when(pl.col("schg") > 0).then(pl.lit("more homeless $")).otherwise(pl.lit("less homeless $")).alias("scat"),
        )
    )
    grand = w["vchg"].mean()
    return (
        w.group_by("hcat", "scat")
        .agg(
            (pl.col("vchg").mean() * 100).alias("visits_chg"),
            pl.len().alias("n"),
            # within-cohort spread of the visit change, so a single cell number isn't over-read
            (pl.col("vchg").quantile(0.25) * 100).alias("p25"),
            (pl.col("vchg").quantile(0.75) * 100).alias("p75"),
        )
        .with_columns((pl.col("visits_chg") - grand * 100).alias("visits_chg_demeaned"))
    )
