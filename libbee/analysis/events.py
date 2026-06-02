"""Causal cuts (§7, Part II) — the event study and difference-in-differences.

These cuts go beyond correlation: they classify units by what they *did* (cut or extended hours, raised or
cut funding) and track the outcome around that event. The **event study** indexes each cohort to
$\\text{baseline} = 100$ and reads the dip; the **difference-in-differences** panel feeds
$\\text{visits\\_pc} \\sim \\text{treated} \\times \\text{post}$, whose interaction coefficient is the causal
effect estimate. Treatment flags are NaN-safe so a missing baseline never silently lands a unit in the
control group.
"""

from __future__ import annotations

import polars as pl

from ._common import _pair_years
from .series import index_to_100


def event_cohorts(panel, *, baseline, horizon, unit="fips", thresh=5.0, fourth_cohort=True) -> pl.DataFrame:
    """Classify each ``unit`` by its hours change over [baseline, baseline+horizon] → ``[unit, cohort]``.
    Cohorts: cut / extended / held; with ``fourth_cohort`` a "cut then recovered" group (dipped at the
    midpoint, back near baseline by the end) is split out of "held steady"."""
    end = baseline + horizon
    mid = baseline + horizon // 2
    ends = (
        panel.filter(pl.col("year").is_in([baseline, mid, end]) & (pl.col("hours_pc") > 0))
        .group_by(unit)
        .agg(
            pl.col("hours_pc").filter(pl.col("year") == baseline).first().alias("h0"),
            pl.col("hours_pc").filter(pl.col("year") == mid).first().alias("hm"),
            pl.col("hours_pc").filter(pl.col("year") == end).first().alias("h1"),
        )
        .filter(pl.col("h0").is_not_null() & pl.col("h1").is_not_null() & (pl.col("h0") > 0))
        .with_columns(
            ((pl.col("h1") / pl.col("h0") - 1) * 100).alias("hchg"),
            ((pl.col("hm") / pl.col("h0") - 1) * 100).alias("hmid"),
        )
    )
    recovered = (pl.col("hmid") < -thresh) if fourth_cohort else pl.lit(False)
    cohort = (
        pl.when(pl.col("hchg") < -thresh)
        .then(pl.lit("cut hours"))
        .when(pl.col("hchg") > thresh)
        .then(pl.lit("extended hours"))
        .when(recovered)
        .then(pl.lit("cut then recovered"))
        .otherwise(pl.lit("held steady"))
        .alias("cohort")
    )
    return ends.with_columns(cohort).select([unit, "cohort"])


def event_study(
    panel,
    *,
    baseline,
    horizon,
    unit="fips",
    thresh=5.0,
    fourth_cohort=True,
    measures=("visits_pc", "checkouts_pc", "wifi_pc"),
) -> tuple[pl.DataFrame, dict]:
    """Index each measure to **baseline = 100** within each open-hours cohort and trace it over time (§7).

    Units are first classified by their hours change over the window (via `event_cohorts`: cut / extended /
    held / optionally "cut then recovered"), then each measure's cohort mean is re-based so the baseline
    year equals 100 — i.e. $\\text{idx}_t = 100 \\cdot \\bar{m}_t / \\bar{m}_{\\text{baseline}}$. Plotting the
    cohorts side by side shows the **dip** after an hours cut against the flat "held steady" line. Runs on
    `county_panel` (`unit="fips"`) for statistically robust cohorts.

    Args:
        panel: Long panel with `[unit, year, hours_pc, *measures]`, one row per unit-year.
        baseline: The index-100 year; the window runs `[baseline, baseline + horizon]`.
        horizon: Number of years after `baseline` to follow.
        unit: Unit-of-analysis column. Default `"fips"`.
        thresh: Percent hours change defining cut vs extended cohorts. Default `5.0`.
        fourth_cohort: If `True`, split a "cut then recovered" cohort out of "held steady".
        measures: Outcome columns to index. Default `("visits_pc", "checkouts_pc", "wifi_pc")`.

    Returns:
        A 2-tuple `(es, counts)`. `es` is a long Polars frame `[cohort, year, t, measure, idx]` where
        `t = year − baseline`; `counts` is `{cohort: n_units}`.

    Example:
        ```python
        es, counts = libbee.analysis.event_study(county_panel, baseline=2014, horizon=5)
        counts                                  # {'cut hours': 412, 'held steady': 1880, ...}
        es.filter((pl.col("measure") == "visits_pc") & (pl.col("cohort") == "cut hours"))  # the dip
        ```
    """
    end = baseline + horizon
    labels = event_cohorts(panel, baseline=baseline, horizon=horizon, unit=unit, thresh=thresh, fourth_cohort=fourth_cohort)
    w = panel.filter(pl.col("year").is_between(baseline, end) & (pl.col("hours_pc") > 0)).select([unit, "year", *measures]).join(labels, on=unit, how="inner")
    agg = w.group_by("cohort", "year").agg([pl.col(c).mean().alias(c) for c in measures]).sort("year")
    es = index_to_100(agg, list(measures), by="cohort").with_columns((pl.col("year") - baseline).alias("t"))
    _cn = labels.group_by("cohort").agg(pl.len().alias("n"))
    counts = {r["cohort"]: r["n"] for r in _cn.iter_rows(named=True)}
    return es, counts


def cohort_cost_curve(panel, cohort_labels, *, baseline, horizon, unit="fips") -> pl.DataFrame:
    """§7 companion — for each event-study cohort, index cost_per_visit to baseline=100 over time.
    ``cohort_labels`` is a frame ``[unit, cohort]`` (from ``event_study``'s classification)."""
    end = baseline + horizon
    w = panel.filter(pl.col("year").is_between(baseline, end) & (pl.col("cost_per_visit") > 0)).select([unit, "year", "cost_per_visit"]).join(cohort_labels, on=unit, how="inner")
    agg = w.group_by("cohort", "year").agg(pl.col("cost_per_visit").mean().alias("cost_per_visit")).sort("year")
    return index_to_100(agg, ["cost_per_visit"], by="cohort").with_columns((pl.col("year") - baseline).alias("t"))


def _assign_treated(panel, baseline, post, treat_thresh, unit, treat_col="hours_pc", rising=False):
    """Shared treatment assignment for the DiD / event-study cuts. By default a ``unit`` is **treated** (1) if
    it *cut* ``treat_col`` (open hours) by more than ``treat_thresh`` percent between ``baseline`` and
    ``post``. Pass ``treat_col`` to define treatment on a different action (e.g. ``"funding_pc"``) and
    ``rising=True`` to flag a big *increase* instead of a cut — that's the "did ADDING funding move the
    outcome?" lens. Returns ``(renamed_panel, [unit, treated])``. Private; covered via did_panel/event_panel.
    """
    renamed = panel.rename({unit: "unit"})
    c0, c1 = f"{treat_col}0", f"{treat_col}1"
    # NaN-safe positivity gate: Polars treats NaN > 0 as True, so an unguarded `> 0` would let a NaN baseline
    # unit slip into the CONTROL group (NaN < thresh is False → treated=0) and silently bias the estimate.
    w = _pair_years(renamed, baseline, post, [treat_col], key="unit").filter((pl.col(c0) > 0) & pl.col(c0).is_finite())
    chg = (pl.col(c1) / pl.col(c0) - 1) * 100
    flag = (chg > treat_thresh) if rising else (chg < treat_thresh)  # rising → a big increase counts as treated
    treated = w.with_columns(chg.alias("_chg")).with_columns(flag.cast(pl.Int8).alias("treated")).select(["unit", "treated"])
    return renamed, treated


def did_panel(panel, *, baseline, post, treat_thresh=-5.0, unit="metro", treat_col="hours_pc", rising=False) -> pl.DataFrame:
    """Build the long **difference-in-differences** panel for §M3 (two periods, treated vs control).

    A unit is `treated` (1) if it moved `treat_col` (default open hours) past `treat_thresh` percent between
    `baseline` and `post` — a *cut* by default, or an *increase* when `rising=True` (e.g.
    `treat_col="funding_pc", rising=True` asks "did adding funding move the outcome?"). Each surviving unit
    contributes two rows (baseline, post) with a `post` indicator, so a regression of
    $\\text{visits\\_pc} \\sim \\text{treated} \\times \\text{post}$ recovers the DiD estimate as the
    `treated:post` interaction. Treatment assignment is NaN-safe (a non-finite baseline can't slip into
    control).

    Args:
        panel: Long panel `[unit, year, visits_pc, treat_col]`, one row per unit-year.
        baseline: The pre-treatment year.
        post: The post-treatment year (the second period).
        treat_thresh: Percent-change cutoff defining treatment. Default `-5.0` (a hours cut of >5%).
        unit: Unit-of-analysis column. Default `"metro"`.
        treat_col: Column whose change defines treatment. Default `"hours_pc"`.
        rising: If `True`, a large *increase* (rather than a cut) flags treatment.

    Returns:
        A Polars frame `[unit, year, visits_pc, treated, post]` — two rows per surviving unit, sorted by
        `unit` then `year`.

    Example:
        ```python
        import statsmodels.formula.api as smf
        did = libbee.analysis.did_panel(county_panel, baseline=2009, post=2014, unit="fips")
        fit = smf.ols("visits_pc ~ treated * post", did.to_pandas()).fit()
        fit.params["treated:post"]    # the causal effect of cutting hours (≈ -0.77 visits/res, p=0.001)
        ```
    """
    renamed, treated = _assign_treated(panel, baseline, post, treat_thresh, unit, treat_col, rising)
    base = (
        renamed.filter(pl.col("year").is_in([baseline, post]) & (pl.col("visits_pc") > 0))
        .select(["unit", "year", "visits_pc"])
        .join(treated, on="unit", how="inner")
        .with_columns((pl.col("year") == post).cast(pl.Int8).alias("post"))
    )
    return base.sort("unit", "year")


def event_panel(panel, *, baseline, post, treat_thresh=-5.0, unit="fips", measure="visits_pc", leads=(), treat_col="hours_pc", rising=False) -> pl.DataFrame:
    """Build a regression-ready **event-study** panel around a common treatment year (``post``) — the
    deeper, dynamic cousin of ``did_panel``. Treatment is assigned exactly as in the DiD (a unit cut hours
    by more than ``treat_thresh``% between ``baseline`` and ``post``), then every year in
    ``leads ∪ [baseline … post]`` is stamped with its **event time** ``evt = year − post`` (0 = the
    treatment year, negatives = pre-periods). Regressing ``measure ~ treated * C(evt)`` then yields one
    coefficient per lead/lag: the pre-period (``evt < 0``) terms are the **parallel-trends test** (should be
    ≈0), the post terms the **dynamic treatment effects**. Returns ``[unit, year, <measure>, treated, evt]``.

    Honest caveat: with only a few annual pre-periods the pre-trend test is weak — read it as suggestive,
    not decisive.

    Example::

        import statsmodels.formula.api as smf
        ep = libbee.analysis.event_panel(county_panel, baseline=2009, post=2014,
                                         measure="visits_pc", leads=[2005, 2007])
        fit = smf.ols("visits_pc ~ treated * C(evt)", ep.to_pandas()).fit()
        {k: fit.params[k] for k in fit.params.index if k.startswith("treated:C(evt)")}  # per-event effects
    """
    renamed, treated = _assign_treated(panel, baseline, post, treat_thresh, unit, treat_col, rising)
    years = sorted(set(leads) | set(range(baseline, post + 1)))
    base = (
        renamed.filter(pl.col("year").is_in(years) & (pl.col(measure) > 0))
        .select(["unit", "year", measure])
        .join(treated, on="unit", how="inner")
        .with_columns((pl.col("year") - post).alias("evt"))
    )
    return base.sort("unit", "year")


def did_means(panel, *, baseline, post, treat_thresh=-5.0, unit="fips") -> pl.DataFrame:
    """2×2 group means for the DiD chart. Returns ``[group∈{treated,control}, year, period∈{pre,post},
    visits_pc]`` (reuses ``did_panel``'s treatment assignment)."""
    dp = did_panel(panel, baseline=baseline, post=post, treat_thresh=treat_thresh, unit=unit)
    return (
        dp.group_by("treated", "year", "post")
        .agg(pl.col("visits_pc").mean().alias("visits_pc"))
        .with_columns(
            pl.when(pl.col("treated") == 1).then(pl.lit("treated")).otherwise(pl.lit("control")).alias("group"),
            pl.when(pl.col("post") == 1).then(pl.lit("post")).otherwise(pl.lit("pre")).alias("period"),
        )
        .select(["group", "year", "period", "visits_pc"])
        .sort("group", "year")
    )


def did_trends(panel, *, baseline, post, treat_thresh=-5.0, unit="fips", show_years=None) -> pl.DataFrame:
    """Treated/control mean visits across several years — the parallel-trends VISUAL behind the DiD.
    Treatment is assigned by the hours change baseline→post (as in ``did_panel``); means are computed for
    every year in ``show_years`` (default ``[baseline, post]``), so pre-baseline years show whether the
    groups moved in parallel *before* treatment. Returns ``[group, year, visits_pc]``.

    Example::

        # include pre-baseline years to *show* the parallel-trends assumption holds before treatment
        tr = libbee.analysis.did_trends(county_panel, baseline=2009, post=2014,
                                       show_years=[2007, 2008, 2009, 2014])
        tr.pivot("group", index="year", values="visits_pc")   # treated vs control, year by year
    """
    years = show_years or [baseline, post]
    renamed, treated = _assign_treated(panel, baseline, post, treat_thresh, unit)
    base = renamed.filter(pl.col("year").is_in(years) & (pl.col("visits_pc") > 0)).select(["unit", "year", "visits_pc"]).join(treated, on="unit", how="inner")
    return (
        base.group_by("treated", "year")
        .agg(pl.col("visits_pc").mean().alias("visits_pc"))
        .with_columns(pl.when(pl.col("treated") == 1).then(pl.lit("treated")).otherwise(pl.lit("control")).alias("group"))
        .select(["group", "year", "visits_pc"])
        .sort("group", "year")
    )
