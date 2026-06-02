"""Leaderboard & funding cuts — state rankings, growth, funding deciles/tiers, the dose-response.

These cuts answer the **ranking** and **dose-response** questions: who uses libraries most per resident, who
is growing, and does more funding buy more visits? Every rate here is population-weighted —
$\\text{rate} = \\sum_i c_i / \\sum_i p_i$ over the units $i$ in a group — so counts and population are summed
*before* dividing and a tiny county never outvotes a large one.
"""

from __future__ import annotations

import polars as pl

from ..geo import CENSUS_REGION, FIPS2ST, ST2FIPS
from ._common import _pair_years


def state_leaderboard(county_equity) -> pl.DataFrame:
    """Roll counties up to one **population-weighted** row per state for the map and the rankings table.

    Counties are grouped by the two-digit FIPS state prefix; within each state the raw counts and population
    are summed and *then* divided, so a per-capita figure is $\\sum_i c_i / \\sum_i p_i$ — the honest national
    average, not a mean of county rates. Adds an integer FIPS `id` (for the topojson map join) and a Census
    `region` label. States whose FIPS prefix can't be mapped are dropped.

    Args:
        county_equity: One row per county carrying raw counts (`visits`, `checkouts`, `funding`, `hours`),
            the `popu_lsa` population, and a `fips` string.

    Returns:
        A Polars frame with columns `[st, id, visits_pc, checkouts_pc, funding_pc, hours_pc,
        cost_per_visit, pop, region]` — one row per state. `cost_per_visit` is
        $\\sum \\text{funding} / \\sum \\text{visits}$.

    Example:
        ```python
        board = libbee.analysis.state_leaderboard(libbee.load("county_equity"))
        board.sort("visits_pc", descending=True).head(5)["st"]    # the most-visited states
        ```
    """
    return (
        county_equity.with_columns(pl.col("fips").str.slice(0, 2).replace_strict(FIPS2ST, default=None).alias("st"))
        .filter(pl.col("st").is_not_null())
        .group_by("st")
        .agg(
            (pl.col("visits").sum() / pl.col("popu_lsa").sum()).alias("visits_pc"),
            (pl.col("checkouts").sum() / pl.col("popu_lsa").sum()).alias("checkouts_pc"),
            (pl.col("funding").sum() / pl.col("popu_lsa").sum()).alias("funding_pc"),
            (pl.col("hours").sum() / pl.col("popu_lsa").sum()).alias("hours_pc"),
            (pl.col("funding").sum() / pl.col("visits").sum()).alias("cost_per_visit"),
            pl.col("popu_lsa").sum().alias("pop"),
        )
        .with_columns(
            pl.col("st").replace_strict(ST2FIPS, default=None).cast(pl.Int64).alias("id"),
            pl.col("st").replace_strict(CENSUS_REGION, default="—").alias("region"),
        )
    )


def state_growth(state_panel, *, years=5) -> pl.DataFrame:
    """Per-state mean annual % change in visits_pc over the trailing ``years`` window → ``[st, chg]``."""
    hi = state_panel["year"].max()
    lo = hi - years
    sp = (
        state_panel.filter(pl.col("year").is_between(lo, hi) & (pl.col("visits_pc") > 0))
        .sort("st", "year")
        .with_columns((pl.col("visits_pc").pct_change().over("st") * 100).alias("_yoy"))
    )
    return sp.group_by("st").agg(pl.col("_yoy").mean().alias("chg")).filter(pl.col("chg").is_not_null())


def quantile_bins(df, *, value, measures, bins=10) -> tuple[pl.DataFrame, float]:
    """Rank rows by ``value`` into ``bins`` equal groups (1..bins, Int32) and average each measure.
    Returns ``(frame, r)`` where r = Pearson corr of value vs the first measure."""
    d = df.filter(pl.col(value) > 0)
    binned = (pl.col(value).rank() / pl.len() * bins).ceil().clip(1, bins).cast(pl.Int32).alias("bin")
    # mean per measure + the within-bin spread (p25/p75) of the first measure, so a bar isn't read as if
    # every county in the bin sat on the mean. n is the bin's county count.
    agg = [pl.col(m).mean().alias(m) for m in measures] + [
        pl.col(value).mean().alias(value),
        pl.len().alias("n"),
        pl.col(measures[0]).quantile(0.25).alias("p25"),
        pl.col(measures[0]).quantile(0.75).alias("p75"),
    ]
    frame = d.with_columns(binned).group_by("bin").agg(agg).sort("bin")
    r = float(d.select(pl.corr(value, measures[0])).item())
    return frame, r


def funding_deciles(county_equity, *, bins=10) -> tuple[pl.DataFrame, float]:
    """§4 — counties binned by funding_pc → mean visits_pc per decile. Returns ``(frame, r)``; the
    ``decile`` column is Int32 so the chart sorts 1..10 numerically.

    Example::

        bins, r = libbee.analysis.funding_deciles(libbee.load("county_equity"))
        round(r, 2)                     # ~0.5 — more funding, more visits, monotone
        bins.select(["decile", "visits_pc", "p25", "p75", "n"])   # mean + IQR + count per decile
    """
    frame, r = quantile_bins(county_equity, value="funding_pc", measures=["visits_pc"], bins=bins)
    return frame.rename({"bin": "decile"}), r


def funding_tiers(county_equity) -> pl.DataFrame:
    """Split counties into **low / med / high funding tiers** and pool each tier's efficiency (§5).

    Counties with positive funding and visits are cut at the funding-per-resident **tertiles** ($q_{1/3}$,
    $q_{2/3}$) into three tiers. Within each tier the figures are pooled — `cost_per_visit` is
    $\\sum \\text{funding} / \\sum \\text{visits}$, a single number — and the per-county spread of that
    figure is reported as median / p25 / p75 so "a dollar goes further" isn't misread as every county
    matching the tier. Rows are sorted by ascending `funding_pc` (low → high).

    Args:
        county_equity: One row per county with `funding_pc`, raw `funding`, `visits`, and `popu_lsa`. Rows
            with non-positive funding, visits, or funding_pc are filtered out.

    Returns:
        A Polars frame `[tier, cost_per_visit, visits_pc, funding_pc, n, cpv_median, cpv_p25, cpv_p75]` —
        one row per tier, where `n` is the tier's county count.

    Example:
        ```python
        tiers = libbee.analysis.funding_tiers(libbee.load("county_equity"))
        ```
    """
    d = county_equity.filter((pl.col("funding_pc") > 0) & (pl.col("visits") > 0) & (pl.col("funding") > 0))
    q1, q2 = d["funding_pc"].quantile(1 / 3), d["funding_pc"].quantile(2 / 3)
    tier = pl.when(pl.col("funding_pc") <= q1).then(pl.lit("low spend")).when(pl.col("funding_pc") <= q2).then(pl.lit("med spend")).otherwise(pl.lit("high spend")).alias("tier")
    return (
        d.with_columns(tier)
        .group_by("tier")
        .agg(
            (pl.col("funding").sum() / pl.col("visits").sum()).alias("cost_per_visit"),
            (pl.col("visits").sum() / pl.col("popu_lsa").sum()).alias("visits_pc"),
            (pl.col("funding").sum() / pl.col("popu_lsa").sum()).alias("funding_pc"),
            pl.len().alias("n"),
            # the pooled cost_per_visit is one number; these show the within-tier spread of the
            # per-county figure, so "a dollar goes further" isn't read as if every county matched the tier.
            (pl.col("funding") / pl.col("visits")).median().alias("cpv_median"),
            (pl.col("funding") / pl.col("visits")).quantile(0.25).alias("cpv_p25"),
            (pl.col("funding") / pl.col("visits")).quantile(0.75).alias("cpv_p75"),
        )
        .sort("funding_pc")
    )


def funding_response_longitudinal(county_panel, *, year_a, year_b, thresh=0.05) -> tuple[pl.DataFrame, dict]:
    """§5 within-unit — among the SAME counties, did those that raised funding between year_a and
    year_b see visits rise? Returns ``(group_frame[fund_group, visits_chg, n], {"r": corr})``."""
    w = (
        _pair_years(county_panel, year_a, year_b, ["funding_pc", "visits_pc"])
        .filter((pl.col("funding_pc0") > 0) & (pl.col("visits_pc0") > 0))
        .with_columns(
            (pl.col("funding_pc1") / pl.col("funding_pc0") - 1).alias("fchg"),
            (pl.col("visits_pc1") / pl.col("visits_pc0") - 1).alias("vchg"),
        )
        .with_columns(
            pl.when(pl.col("fchg") > thresh)
            .then(pl.lit("raised funding"))
            .when(pl.col("fchg") < -thresh)
            .then(pl.lit("cut funding"))
            .otherwise(pl.lit("held funding"))
            .alias("fund_group")
        )
    )
    grp = w.group_by("fund_group").agg((pl.col("vchg").mean() * 100).alias("visits_chg"), pl.len().alias("n"))
    r = float(w.select(pl.corr("fchg", "vchg")).item()) if w.height > 1 else float("nan")
    return grp, {"r": r}


def cohort_compare(frame, groups, metrics, *, pop="popu_lsa") -> pl.DataFrame:
    """Compare arbitrary cohorts of rows on **population-weighted per-capita** metrics.

    Every "do X-type places differ from Y-type places?" question is the *same* operation: pick some rows,
    add up a raw count (visits, dollars, wifi sessions), divide by the people those rows serve, and compare.
    The subtlety is the **weighting** — averaging per-county *rates* would let a 500-person county count as
    much as Los Angeles. The honest comparison sums first, then divides, so this always returns

    $$\\text{per\\_capita} = \\frac{\\sum_i \\text{metric}_i}{\\sum_i \\text{pop}_i}$$

    over the rows $i$ in a cohort. Polars `.sum()` skips nulls, so a metric absent for some rows (e.g. wifi
    before 2014) contributes nothing rather than poisoning the cohort.

    Args:
        frame: One row per unit (e.g. a county) carrying the `pop` column and every `metrics` column as a
            **raw count**, not a pre-divided rate — pass `visits`, not `visits_pc`.
        groups: Mapping of cohort label → a boolean Polars expression selecting that cohort's rows. Cohorts
            may overlap; "everybody" is `pl.lit(True)`. Insertion order is preserved, so the dict order *is*
            the chart's x-axis order.
        metrics: The raw-count columns to turn into per-capita rates, e.g. `["visits", "funding", "wifi"]`.
        pop: The population column used as the denominator (IMLS "legal service-area population").

    Returns:
        A tidy/long Polars frame `[cohort, metric, per_capita, n, pop]` — one row per `(cohort, metric)`.
        `per_capita` is `None` if the cohort is empty, `n` is the unit count, `pop` the cohort's total
        population. The long shape lets Altair colour/facet by `metric` with no reshaping.

    Example:
        ```python
        county = libbee.load("county_equity").with_columns(
            pl.col("fips").str.slice(0, 2).replace_strict(libbee.geo.FIPS2ST, default=None).alias("st"))
        groups = {"Liberal (CA·MA)": pl.col("st").is_in(["CA", "MA"]),
                  "Conservative (WY·AL)": pl.col("st").is_in(["WY", "AL"]),
                  "National": pl.lit(True)}
        libbee.analysis.cohort_compare(county, groups, ["visits", "funding"], pop="popu_lsa")
        ```
    """
    rows = []
    # One pass per cohort. `.sum()` skips nulls in Polars, so a metric that simply didn't exist for some
    # rows (e.g. wifi sessions before 2014) contributes nothing rather than poisoning the cohort with a
    # null. We divide the summed count by the summed population => a true population-weighted rate.
    for label, mask in groups.items():
        sub = frame.filter(mask)
        total_pop = sub[pop].sum()  # the denominator: everyone this cohort's libraries serve
        for metric in metrics:
            rows.append(
                {
                    "cohort": label,
                    "metric": metric,
                    # guard the empty-cohort case (total_pop == 0) so we emit a clean null, not a divide error
                    "per_capita": (sub[metric].sum() / total_pop) if total_pop else None,
                    "n": sub.height,
                    "pop": float(total_pop) if total_pop is not None else 0.0,
                }
            )
    return pl.DataFrame(rows)
