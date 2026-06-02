"""Unit tests for libbee.analysis — the pure data-prep / "cuts" layer.

Every function is exercised against tiny synthetic Polars frames (no disk IO), including the defensive
branches, so the module stays at 100% line coverage.
"""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest

from libbee import analysis as A

# ── synthetic frames ────────────────────────────────────────────────────────
# Real state abbrs so geo.CENSUS_REGION maps: 08=CO(West) 36=NY(Northeast) 17=IL(Midwest) 13=GA(South)
_FIPS = [
    "08001",
    "08003",
    "36001",
    "36003",
    "17001",
    "17003",
    "13001",
    "13003",
    "06001",
    "06003",
    "48001",
    "48003",
]


def _state_panel() -> pl.DataFrame:
    rows = []
    for st, mult in [("CO", 1.0), ("NY", 1.5), ("IL", 1.2)]:
        for i, yr in enumerate(range(2015, 2021)):
            pop = 1000.0
            rows.append(
                dict(
                    st=st,
                    year=yr,
                    popu_lsa=pop,
                    visits=(100 + 5 * i) * mult,
                    checkouts=(80 - 2 * i) * mult,
                    wifi=(10 + 8 * i) * mult,
                    visits_pc=(100 + 5 * i) * mult / pop,
                    funding_pc=(50 + i) * mult,
                    funding=(50 + i) * mult * pop,
                )
            )
    return pl.DataFrame(rows)


def _county_equity() -> pl.DataFrame:
    rows = []
    for i, fips in enumerate(_FIPS):
        pop = 1000.0 + i * 10
        funding = 20000.0 + i * 4000  # spread so deciles/tiers split
        visits = 2000.0 + i * 150
        hours_pc = 2.0 + 0.1 * i
        hours = hours_pc * pop
        rows.append(
            dict(
                fips=fips,
                NAME=f"County {i}, State",
                popu_lsa=pop,
                total_pop=pop + 100,  # Census county total ≥ library service-area pop
                visits=visits,
                checkouts=visits * 0.8,
                wifi=visits * 0.3,
                funding=funding,
                hours=hours,
                visits_pc=visits / pop,
                checkouts_pc=visits * 0.8 / pop,  # per-capita outcomes for multi-target model_matrix
                wifi_pc=visits * 0.3 / pop,
                funding_pc=funding / pop,
                cost_per_visit=funding / visits,
                hours_pc=hours_pc,
                poverty=10.0 + i,
                median_income=40000.0 + i * 1000,
                no_broadband=20.0 - i,
                density=5.0 + i,
            )
        )
    return pl.DataFrame(rows)


def _county_panel() -> pl.DataFrame:
    rows = []
    for i, fips in enumerate(_FIPS):
        for yr in [2005, 2009, 2011, 2014, 2018, 2019, 2023]:  # 2005 = a pre-baseline lead for event_panel
            # county i "cuts hours" if even, "extends" if odd; visits move with hours (stay positive)
            hours = 3.0 - 0.05 * (yr - 2009) if i % 2 == 0 else 3.0 + 0.05 * (yr - 2009)
            visits_pc = 2.0 + (hours - 3.0)
            # funding shifts between 2018 and 2019, differently per county → raised/held/cut groups
            fund = 40.0 + ((i - 6) * 2.0 if yr >= 2019 else 0.0)
            rows.append(
                dict(
                    fips=fips,
                    year=yr,
                    popu_lsa=1000.0,
                    hours_pc=hours,
                    visits_pc=visits_pc,
                    checkouts=1500.0,
                    checkouts_pc=1.5,
                    wifi=500.0,
                    wifi_pc=0.5,
                    visits=visits_pc * 1000.0,
                    funding_pc=fund,
                    cost_per_visit=10.0 + 0.1 * (yr - 2009),
                )
            )
    return pl.DataFrame(rows)


def _county_homeless() -> pl.DataFrame:
    rows = []
    for i, fips in enumerate(_FIPS):
        for yr in [2018, 2019, 2023]:
            rows.append(dict(fips=fips, year=yr, homeless_per10k=5.0 + i, spend_pc=10.0 + (2.0 if yr >= 2019 else 0.0)))
    return pl.DataFrame(rows)


def _metro_panel() -> pl.DataFrame:
    rows = []
    metros = ["Denver-Aurora-Lakewood", "Seattle-Tacoma-Bellevue", "Chicago-Naperville-Elgin"]
    for m, cut in zip(metros, [True, False, True], strict=True):
        for yr in [2009, 2011, 2014]:
            hours = 3.0 - 0.2 * (yr - 2009) if cut else 3.0 + 0.1 * (yr - 2009)
            rows.append(
                dict(
                    cbsa="1",
                    metro=m,
                    year=yr,
                    popu_lsa=1000.0,
                    visits=2000.0,
                    visits_pc=2.0 + (hours - 3.0),
                    hours_pc=hours,
                )
            )
    return pl.DataFrame(rows)


def _facts() -> pl.DataFrame:
    rows = []
    for geo in ["Colorado", "California"]:
        for yr in [2018, 2019, 2020]:
            rows.append(
                dict(
                    geo_level="state",
                    geo_id="08",
                    geo_name=geo,
                    state="CO",
                    year=yr,
                    metric="visits_pc",
                    value=3.0 + yr - 2018,
                    source="IMLS",
                )
            )
    return pl.DataFrame(rows)


# ── scope_series ─────────────────────────────────────────────────────────────
def test_scope_series_country():
    out, label = A.scope_series(_state_panel(), _county_panel(), _county_equity(), scope="United States")
    assert label == "United States"
    assert out["visits_pc"][0] == pytest.approx(out["visits"][0] / out["popu_lsa"][0])
    assert out["year"].is_sorted()


def test_scope_series_state():
    out, label = A.scope_series(_state_panel(), _county_panel(), _county_equity(), scope="One state", state="NY")
    assert label == "NY"
    assert out.height == 6


def test_scope_series_county_found_and_missing():
    ce = _county_equity()
    out, label = A.scope_series(_state_panel(), _county_panel(), ce, scope="One county", county_name="County 0, State")
    assert label == "County 0"
    # missing county name → fips '__none__' → empty frame, label from the name
    out2, label2 = A.scope_series(_state_panel(), _county_panel(), ce, scope="One county", county_name="Nowhere, State")
    assert out2.height == 0 and label2 == "Nowhere"
    # county scope with no county_name falls through to the country branch
    out3, label3 = A.scope_series(_state_panel(), _county_panel(), ce, scope="One county")
    assert label3 == "United States"


# ── index_to_100 ─────────────────────────────────────────────────────────────
def test_index_to_100_plain_and_grouped():
    df = pl.DataFrame({"year": [2000, 2001, 2002], "visits_pc": [10.0, 12.0, 8.0]})
    out = A.index_to_100(df, ["visits_pc"])
    assert out.filter(pl.col("year") == 2000)["idx"][0] == 100.0
    assert set(out["measure"]) == {"visits"}
    grouped = pl.DataFrame({"cohort": ["a", "a", "b", "b"], "year": [1, 2, 1, 2], "x_pc": [10.0, 20.0, 5.0, 5.0]})
    g = A.index_to_100(grouped, ["x_pc"], by="cohort")
    assert g.filter((pl.col("cohort") == "a") & (pl.col("year") == 2))["idx"][0] == 200.0


def test_index_to_100_empty():
    df = pl.DataFrame({"year": [2000, 2001], "visits_pc": [0.0, 0.0]})  # no positive baseline
    out = A.index_to_100(df, ["visits_pc"])
    assert out.height == 0
    assert out.columns == ["year", "measure", "idx"]


# ── leaderboards ─────────────────────────────────────────────────────────────
def test_state_leaderboard_regions_and_id():
    lb = A.state_leaderboard(_county_equity())
    assert {"st", "id", "region", "cost_per_visit"} <= set(lb.columns)
    co = lb.filter(pl.col("st") == "CO")
    assert co["id"][0] == 8 and co["region"][0] == "West"


def test_state_leaderboard_drops_unmapped_state():
    ce = _county_equity().vstack(_county_equity().head(1).with_columns(pl.lit("60001").alias("fips")))
    lb = A.state_leaderboard(ce)
    assert "60" not in [str(i)[:2] for i in lb["id"].to_list()]  # FIPS 60 (AS) unmapped → dropped


def test_state_growth():
    g = A.state_growth(_state_panel(), years=5)
    assert set(g["st"]) == {"CO", "NY", "IL"}
    assert g["chg"].is_not_null().all()


# ── funding cuts ─────────────────────────────────────────────────────────────
def test_funding_deciles_order_and_dtype():
    frame, r = A.funding_deciles(_county_equity())
    assert frame["decile"].dtype == pl.Int32
    assert frame["decile"].to_list() == sorted(frame["decile"].to_list())
    assert -1.0 <= r <= 1.0


def test_quantile_bins_custom():
    frame, _ = A.quantile_bins(_county_equity(), value="funding_pc", measures=["visits_pc"], bins=5)
    max_bin = frame["bin"].max()
    assert isinstance(max_bin, int)
    assert max_bin <= 5


def test_funding_tiers():
    t = A.funding_tiers(_county_equity())
    assert set(t["tier"]) == {"low spend", "med spend", "high spend"}
    assert t["funding_pc"].is_sorted()


def test_cohort_compare():
    """cohort_compare() returns tidy population-weighted per-capita rates, cohorts in dict order."""
    ce = _county_equity()
    groups = {
        "first two": pl.col("fips").is_in(["08001", "08003"]),
        "dense": pl.col("density") > 8.0,
        "all": pl.lit(True),
    }
    out = A.cohort_compare(ce, groups, ["visits", "wifi"], pop="popu_lsa")
    assert {"cohort", "metric", "per_capita", "n", "pop"} == set(out.columns)
    # cohort order is preserved (so the chart x-axis order is whatever the caller wrote)
    assert out["cohort"].unique(maintain_order=True).to_list() == ["first two", "dense", "all"]
    # population-weighted: "all"/visits == sum(visits)/sum(pop), NOT the mean of per-county rates
    allrow = out.filter((pl.col("cohort") == "all") & (pl.col("metric") == "visits"))
    assert abs(allrow["per_capita"][0] - float(ce["visits"].sum()) / float(ce["popu_lsa"].sum())) < 1e-9
    assert out.filter(pl.col("cohort") == "first two")["n"][0] == 2
    # empty cohort -> clean null + zero population, no divide error
    empty = A.cohort_compare(ce, {"none": pl.col("fips") == "99999"}, ["visits"])
    assert empty["per_capita"][0] is None and empty["pop"][0] == 0.0


def test_funding_response_longitudinal_and_singleton():
    grp, sc = A.funding_response_longitudinal(_county_panel(), year_a=2018, year_b=2019)
    assert "fund_group" in grp.columns and not math.isnan(sc["r"])
    # singleton panel → correlation is nan (height <= 1 branch)
    one = _county_panel().filter(pl.col("fips") == "08001")
    grp1, sc1 = A.funding_response_longitudinal(one, year_a=2018, year_b=2019)
    assert math.isnan(sc1["r"])


# ── homelessness ─────────────────────────────────────────────────────────────
def test_homeless_scatter_fit():
    pts, fit = A.homeless_scatter(_county_equity(), _county_homeless(), year=2019)
    assert pts.height > 0 and -1.0 <= fit["r"] <= 1.0 and fit["year"] == 2019


def test_homeless_scatter_zero_variance_slope():
    ce = _county_equity()
    # all homeless_per10k identical → var(x)=0 → slope falls back to 0.0
    ch = pl.DataFrame(
        {
            "fips": ce["fips"],
            "year": [2019] * ce.height,
            "homeless_per10k": [5.0] * ce.height,
            "spend_pc": [1.0] * ce.height,
        }
    )
    _, fit = A.homeless_scatter(ce, ch, year=2019)
    assert fit["slope"] == 0.0


def test_homeless_cohorts_demeaned():
    coh = A.homeless_cohorts(_county_panel(), _county_homeless(), year_a=2018, year_b=2023)
    assert "visits_chg_demeaned" in coh.columns
    # the demeaned column is centred: the n-weighted mean is ~0
    weighted_sum = float((coh["visits_chg_demeaned"] * coh["n"]).sum())
    total_weight = float(coh["n"].sum())
    w = weighted_sum / total_weight
    assert abs(w) < 1e-6


# ── event study ──────────────────────────────────────────────────────────────
def test_event_cohorts_fourth_on_and_off():
    on = A.event_cohorts(_county_panel(), baseline=2009, horizon=5, unit="fips", fourth_cohort=True)
    off = A.event_cohorts(_county_panel(), baseline=2009, horizon=5, unit="fips", fourth_cohort=False)
    assert {"fips", "cohort"} == set(on.columns)
    assert "cut then recovered" not in set(off["cohort"])


def test_event_study_and_cost_curve():
    es, counts = A.event_study(_county_panel(), baseline=2009, horizon=5, unit="fips")
    assert {"cohort", "t", "measure", "idx"} <= set(es.columns)
    assert sum(counts.values()) > 0
    labels = A.event_cohorts(_county_panel(), baseline=2009, horizon=5, unit="fips")
    cc = A.cohort_cost_curve(_county_panel(), labels, baseline=2009, horizon=5)
    assert cc.filter(pl.col("t") == 0)["idx"].to_list()[0] == 100.0


# ── regions, explorer, ML prep ───────────────────────────────────────────────
def test_region_trends_with_and_without_metros():
    plain = A.region_trends(_state_panel(), _metro_panel())
    assert set(plain["kind"]) == {"region"}
    overlaid = A.region_trends(_state_panel(), _metro_panel(), metros=["Denver-Aurora-Lakewood"])
    assert "DEN" in set(overlaid["series"]) and "metro" in set(overlaid["kind"])


def test_explorer_series():
    out = A.explorer_series(_facts(), level="state", metric="visits_pc", geos=["Colorado"], year_lo=2018, year_hi=2019)
    assert set(out["geo_name"]) == {"Colorado"} and out.height == 2
    empty = A.explorer_series(_facts(), level="state", metric="visits_pc", geos=["Nowhere"], year_lo=2018, year_hi=2019)
    assert empty.height == 0


def test_model_matrix():
    X, y, feats, ids = A.model_matrix(_county_equity(), _county_homeless(), year=2019)
    assert feats == A.MODEL_FEATURES
    assert X.shape[1] == len(feats) and y.len() == X.height
    assert set(ids.columns) == {"fips", "NAME", "state"}


def test_model_matrix_multi_target():
    """target= picks the outcome (and its >0 row guard); keep_targets= attaches aligned outcome columns."""
    ce, ch = _county_equity(), _county_homeless()
    # checkouts target → y is checkouts_pc, and keep_targets carries the other two aligned to the same rows
    X, y, feats, ids = A.model_matrix(ce, ch, target="checkouts_pc", keep_targets=["visits_pc", "wifi_pc"])
    assert feats == A.MODEL_FEATURES and y.len() == X.height
    assert {"fips", "NAME", "state", "visits_pc", "wifi_pc"} == set(ids.columns)
    # checkouts_pc == 0.8 * visits_pc in the fixture, so y matches the aligned visits column scaled
    assert y[0] == pytest.approx(ids["visits_pc"][0] * 0.8)


def test_did_panel():
    dp = A.did_panel(_metro_panel(), baseline=2009, post=2014, unit="metro")
    assert {"unit", "year", "visits_pc", "treated", "post"} == set(dp.columns)
    assert set(dp["post"].unique().to_list()) == {0, 1}


def test_national_series():
    ns = A.national_series(_state_panel(), metric="visits_pc", through=2018)
    assert ns["year"].max() == 2018 and "value" in ns.columns


def test_cluster_labels():
    """cluster_labels() maps tertile profiles to the documented personas — asserted BY VALUE so a broken
    label expression can't pass (the prior test only checked labels were non-empty)."""
    profile = pl.DataFrame(
        {
            "cluster": [0, 1, 2],
            "funding_pc": [10.0, 50.0, 100.0],
            "visits_pc": [20.0, 100.0, 200.0],
            "poverty": [0.05, 0.15, 0.30],
            "density": [100.0, 500.0, 2000.0],
        }
    )
    labels = A.cluster_labels(profile)
    assert set(labels.columns) == {"cluster", "label"}
    _m = {r["cluster"]: r["label"] for r in labels.iter_rows(named=True)}
    # cluster 2: hi funding + hi visits → "well-funded, high-use"; cluster 0: lo funding → "underfunded"
    assert _m[2] == "well-funded, high-use"
    assert _m[0] == "underfunded"


def test_cluster_labels_named_personas():
    """A second profile hits other documented branches (high-need urban, well-funded-under-used)."""
    profile = pl.DataFrame(
        {
            "cluster": [0, 1, 2],
            "funding_pc": [100.0, 50.0, 10.0],  # 0 hi-fund, 2 lo-fund
            "visits_pc": [20.0, 100.0, 200.0],  # 0 lo-visits → well-funded under-used
            "poverty": [0.05, 0.15, 0.40],  # 2 hi-poverty + lo-funding → underfunded, high-need
            "density": [100.0, 500.0, 2000.0],  # 2 hi-density + hi-poverty → dense, high-need urban
        }
    )
    _m = {r["cluster"]: r["label"] for r in A.cluster_labels(profile).iter_rows(named=True)}
    assert _m[0] == "well-funded, under-used"  # hi funding, lo visits
    assert _m[2] == "underfunded, high-need"  # hi poverty + lo funding (most-specific rule wins)


def test_persona_glossary_describes_every_label():
    """PERSONA_GLOSSARY gives a plain-English description for *every* persona cluster_labels can emit —
    so the notebook can show a lookup table and nobody takes a label name on faith. Keys must match the
    label set exactly (a new label rule without a glossary entry, or a stale entry, fails here)."""
    expected = {
        "well-funded, high-use",
        "well-funded, under-used",
        "underfunded, high-need",
        "dense, high-need urban",
        "dense urban core",
        "efficient: high-use, modest funds",
        "underfunded",
        "higher-need, mid-funded",
        "lean rural",
        "quiet / low-use",
        "mid-market / suburban",
    }
    assert set(A.PERSONA_GLOSSARY) == expected
    assert all(isinstance(v, str) and v for v in A.PERSONA_GLOSSARY.values())
    # and the labels the function actually emits on the documented profiles are all described
    profile = pl.DataFrame(
        {
            "cluster": [0, 1, 2],
            "funding_pc": [100.0, 50.0, 10.0],
            "visits_pc": [20.0, 100.0, 200.0],
            "poverty": [0.05, 0.15, 0.40],
            "density": [100.0, 500.0, 2000.0],
        }
    )
    emitted = {r["label"] for r in A.cluster_labels(profile).iter_rows(named=True)}
    assert emitted <= set(A.PERSONA_GLOSSARY)


def test_did_means():
    """did_means() computes 2×2 group means for diff-in-diff chart."""
    means = A.did_means(_metro_panel(), baseline=2009, post=2014, unit="metro")
    assert {"group", "year", "period", "visits_pc"} == set(means.columns)
    assert set(means["group"].unique().to_list()) == {"treated", "control"}


def test_qa_report_counts_zero_null_and_skips_missing_frames_and_columns():
    """qa_report() counts bad values and ignores absent frames/columns."""
    frames = {
        "county_equity": pl.DataFrame(
            {
                "visits": [10.0, 0.0],
                "funding": [100.0, None],
                "hours": [5.0, 2.0],
                "popu_lsa": [1000.0, 0.0],
            }
        ),
        # omit state_panel entirely to cover the missing-frame branch
        "county_panel": pl.DataFrame({"visits": [1.0], "funding": [1.0]}),
    }

    report = A.qa_report(frames)

    assert {"frame", "column", "zero_or_null", "total", "pct"} == set(report.columns)
    visits = report.filter((pl.col("frame") == "county_equity") & (pl.col("column") == "visits"))
    funding = report.filter((pl.col("frame") == "county_equity") & (pl.col("column") == "funding"))
    assert visits["zero_or_null"][0] == 1 and visits["pct"][0] == 50.0
    assert funding["zero_or_null"][0] == 1 and funding["total"][0] == 2
    assert "state_panel" not in set(report["frame"])
    assert report.filter((pl.col("frame") == "county_panel") & (pl.col("column") == "popu_lsa")).height == 0


def test_did_trends_returns_treated_and_control_means():
    """did_trends() returns treated/control means for the baseline and post years."""
    trends = A.did_trends(_county_panel(), baseline=2009, post=2014, unit="fips")

    assert {"group", "year", "visits_pc"} == set(trends.columns)
    assert set(trends["group"].unique().to_list()) == {"treated", "control"}
    assert set(trends["year"].unique().to_list()) == {2009, 2014}
    assert trends.height == 4
    assert trends["visits_pc"].is_not_null().all()


def test_allocation_frame():
    """allocation_frame() bumps funding and creates allocation metadata."""
    # Create minimal county_equity, ids, and X matrices for allocation
    county_equity = _county_equity()
    X, y, feats, ids = A.model_matrix(county_equity, _county_homeless(), year=2019)
    X_bumped, meta = A.allocation_frame(county_equity, ids, X, delta=5.0, cap_q=0.90)
    # Should return bumped X and metadata frame
    assert X_bumped.shape == X.shape
    assert {"fips", "NAME", "state", "funding_pc", "delta_applied", "pop"} <= set(meta.columns)


def test_greedy_allocation():
    """greedy_allocation() selects counties by descending value per dollar."""
    marginals = pl.DataFrame(
        {
            "fips": ["01001", "01003", "01005"],
            "NAME": ["County A", "County B", "County C"],
            "state": ["01", "01", "01"],
            "dollars_needed": [1000.0, 2000.0, 500.0],
            "extra_visits": [500.0, 1000.0, 250.0],
        }
    )
    chosen, summary = A.greedy_allocation(marginals, budget=2500.0)
    assert "vpd" in chosen.columns
    assert "cum_cost" in chosen.columns
    assert summary["n_counties"] > 0
    assert summary["spent"] <= 2500.0


def test_optimal_allocation_waterfilling():
    """optimal_allocation() takes the highest visits-per-dollar steps first, stops at budget,
    sums variable per-county amounts, and skips zero/negative steps."""
    increments = pl.DataFrame(
        {
            "fips": ["01001", "01001", "01003", "01005", "01007"],
            "NAME": ["County A", "County A", "County B", "County C", "County D"],
            "state": ["01", "01", "01", "01", "01"],
            # step 2 of County A is cheaper-but-less (lower vpd than its step 1, diminishing returns)
            "dollars": [100.0, 100.0, 100.0, 100.0, 0.0],
            "extra_visits": [500.0, 200.0, 400.0, 50.0, 0.0],  # vpd: 5, 2, 4, 0.5, (filtered)
        }
    )
    plan, summary = A.optimal_allocation(increments, budget=300.0)
    # $300 buys A-step1 (vpd 5), B-step1 (vpd 4), A-step2 (vpd 2); C-step1 (0.5) is too low to reach
    assert {"fips", "NAME", "state", "dollars", "extra_visits", "steps"} == set(plan.columns)
    a = plan.filter(pl.col("fips") == "01001")
    assert a["steps"][0] == 2 and a["dollars"][0] == 200.0  # County A gets a *variable* (2-step) amount
    assert "01005" not in plan["fips"].to_list()  # low-vpd county not funded
    assert "01007" not in plan["fips"].to_list()  # zero-dollar step filtered out
    assert summary["spent"] <= 300.0 and summary["n_counties"] == 2
    # an exhausted/zero budget yields an empty plan and zeroed summary
    empty, esum = A.optimal_allocation(increments, budget=0.0)
    assert empty.height == 0 and esum["spent"] == 0.0 and esum["extra_visits"] == 0.0


def test_coverage_report():
    """coverage_report() reports the county + population denominator behind the model's cross-section."""
    rep = A.coverage_report(_county_equity(), us_counties=100, us_pop=100_000)
    assert rep["counties"] == 12 and rep["us_counties"] == 100
    assert rep["pct_counties"] == 12.0
    assert rep["pop_covered"] > 0 and rep["pct_pop"] == round(rep["pop_covered"] / 100_000 * 100, 1)
    assert rep["year"] == 2019  # default label


def test_quantile_bins_reports_spread():
    """quantile_bins() carries within-bin p25/p75 of the first measure alongside the mean + n."""
    frame, _ = A.quantile_bins(_county_equity(), value="funding_pc", measures=["visits_pc"], bins=4)
    assert {"p25", "p75", "n"} <= set(frame.columns)
    assert (frame["p75"] >= frame["p25"]).all()


# ── event_panel (deeper DiD) ────────────────────────────────────────────────
def test_event_panel_event_time_and_treatment():
    """event_panel() stamps evt = year - post around the treatment year and assigns 0/1 treatment.
    leads widen the pre-period window; default leads=() restricts to [baseline..post]."""
    cp = _county_panel()
    ep = A.event_panel(cp, baseline=2009, post=2014, unit="fips", measure="visits_pc", leads=[2011])
    assert {"unit", "year", "visits_pc", "treated", "evt"} == set(ep.columns)
    # evt is exactly year - post; treatment is binary
    assert all(r["evt"] == r["year"] - 2014 for r in ep.iter_rows(named=True))
    assert set(ep["treated"].unique().to_list()) == {0, 1}
    # leads must WIDEN the window: 2005 is pre-baseline, so it appears ONLY when passed as a lead
    ep_lead = A.event_panel(cp, baseline=2009, post=2014, unit="fips", measure="visits_pc", leads=[2005])
    assert 2005 in ep_lead["year"].to_list()
    assert ep_lead.filter(pl.col("year") == 2005)["evt"][0] == 2005 - 2014  # evt = -9
    # default (no leads) restricts to the baseline..post range — 2005 absent
    ep0 = A.event_panel(cp, baseline=2009, post=2014, unit="fips")
    assert 2005 not in ep0["year"].to_list()
    year_min = cast(int, ep0["year"].min())
    year_max = cast(int, ep0["year"].max())
    assert year_min >= 2009 and year_max <= 2014


# ── acf_pacf (Box-Jenkins identification) ───────────────────────────────────
def test_acf_pacf_ramp_and_durbin_levinson():
    """ACF/PACF of a monotone ramp, pinned to reference values so a wrong-but-non-crashing Durbin-Levinson
    recursion (e.g. a sign flip in the inner phi-update) is caught, not just its absence."""
    ap = A.acf_pacf([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nlags=4)
    assert ap["lag"].to_list() == [1, 2, 3, 4]
    _acf = dict(zip(ap["lag"].to_list(), ap["acf"].to_list(), strict=True))
    _pacf = dict(zip(ap["lag"].to_list(), ap["pacf"].to_list(), strict=True))
    # biased ACF of the ramp (reference values)
    assert _acf[1] == pytest.approx(0.7) and _acf[2] == pytest.approx(0.41212, abs=1e-4)
    # PACF(1) == ACF(1) (the identity), and the higher lags come from the inner recursion — pin them so a
    # sign-flipped update can't survive
    assert _pacf[1] == pytest.approx(0.7)
    assert _pacf[2] == pytest.approx(-0.15270, abs=1e-4)
    assert _pacf[3] == pytest.approx(-0.15491, abs=1e-4)


def test_acf_pacf_constant_short_and_nonfinite():
    """Edge cases: a constant series (zero variance) → all-zero (no /0); a one-element series → empty typed
    frame (k_max == 0); None/NaN elements are dropped, NOT silently treated as a flat series."""
    flat = A.acf_pacf([5.0, 5.0, 5.0, 5.0], nlags=3)
    assert (flat["acf"] == 0.0).all() and (flat["pacf"] == 0.0).all()
    empty = A.acf_pacf([7.0], nlags=5)
    assert empty.height == 0 and empty.schema["acf"] == pl.Float64
    # a None or NaN observation is stripped up front → identical to the clean 5-point series (not all-zero)
    clean = A.acf_pacf([1.0, 2.0, 3.0, 4.0, 5.0], nlags=2)
    for _bad in ([1.0, None, 2.0, 3.0, 4.0, 5.0], [1.0, float("nan"), 2.0, 3.0, 4.0, 5.0]):
        got = A.acf_pacf(_bad, nlags=2)
        assert got["acf"].to_list() == pytest.approx(clean["acf"].to_list())
        assert got["acf"][0] != 0.0  # NOT silently zeroed like a constant series would be


# ── aic_table (Box-Jenkins order selection) ─────────────────────────────────
def test_aic_table_sorts_and_flags_best():
    grid = A.aic_table(
        [
            {"p": 1, "d": 1, "q": 1, "aic": 120.0},
            {"p": 0, "d": 1, "q": 1, "aic": 118.0},
            {"p": 2, "d": 1, "q": 0, "aic": 125.0},
        ]
    )
    assert grid["aic"].to_list() == [118.0, 120.0, 125.0]  # ascending
    assert grid["delta_aic"][0] == 0.0 and grid["best"][0] is True
    assert grid["best"].sum() == 1  # exactly one best
    assert A.aic_table([]).height == 0  # empty branch → typed empty frame


# ── fitting_score (service-level traffic light) ─────────────────────────────
def _ids3() -> pl.DataFrame:
    return pl.DataFrame({"fips": ["1", "2", "3"], "NAME": ["A", "B", "C"], "state": ["CO", "NY", "IL"]})


def test_fitting_score_ratings_and_cost():
    """green/yellow/red thresholds + linear cost-to-target; clamp when actual exceeds potential."""
    out = A.fitting_score(
        _ids3(),
        actual=[5.0, 8.0, 12.0],  # A 0.5 red, B 0.8 yellow, C 1.2→clamp 1.0 green
        pred0=[5.0, 8.0, 12.0],
        potential=[10.0, 10.0, 10.0],
        dollars_to_cap=[100.0, 100.0, 0.0],
        target=0.80,
    )
    assert out["rating"].to_list() == ["red", "yellow", "green"]
    assert out["fitting_pct"].to_list() == [0.5, 0.8, 1.0]
    # A: lift 5→8 of a 5→10 span = 60% of the way → 60% of $100 = $60
    assert out.filter(pl.col("fips") == "1")["cost_to_target"][0] == pytest.approx(60.0)
    # B already at target, C clamped/no headroom → both cost 0
    assert out.filter(pl.col("fips") == "2")["cost_to_target"][0] == 0.0
    assert out.filter(pl.col("fips") == "3")["cost_to_target"][0] == 0.0
    assert {"actual", "potential", "fitting_pct", "rating", "cost_to_target"} <= set(out.columns)


def test_fitting_score_zero_potential_branch():
    """potential <= 0 → treated as fully fit (no /0); head <= 0 → cost 0 (the otherwise branches)."""
    out = A.fitting_score(
        _ids3(),
        actual=[0.0, 4.0, 9.0],
        pred0=[0.0, 12.0, 9.0],  # B: pred0 > potential → head < 0 → frac branch otherwise
        potential=[0.0, 10.0, 10.0],  # A: potential 0 → fitting 1.0
        dollars_to_cap=[50.0, 50.0, 50.0],
        target=0.80,
    )
    assert out.filter(pl.col("fips") == "1")["fitting_pct"][0] == 1.0  # zero-potential → fully fit
    assert out.filter(pl.col("fips") == "2")["cost_to_target"][0] == 0.0  # negative headroom → no cost


def test_fitting_score_nonpositive_and_nan_potential_are_consistent():
    """A non-positive or NaN potential must be 'fully fit, zero cost' on BOTH branches (Polars treats
    NaN > 0 as True, so an unguarded version would mis-rate it green with a NaN/positive cost)."""
    out = A.fitting_score(
        _ids3(),
        actual=[1.0, 1.0, 1.0],
        pred0=[-10.0, 5.0, 1.0],
        potential=[-2.0, float("nan"), 10.0],  # A negative, B NaN, C normal
        dollars_to_cap=[100.0, 100.0, 100.0],
        target=0.80,
    )
    a = out.filter(pl.col("fips") == "1")
    b = out.filter(pl.col("fips") == "2")
    # negative potential → fully fit AND zero cost (the two branches agree)
    assert a["fitting_pct"][0] == 1.0 and a["cost_to_target"][0] == 0.0 and a["rating"][0] == "green"
    # NaN potential → routed to the same "no real headroom" branch, not silently rated on a NaN
    assert b["fitting_pct"][0] == 1.0 and b["cost_to_target"][0] == 0.0


def test_model_matrix_keep_targets_overlap_is_deduped():
    """keep_targets overlapping the target / a feature / an id column must NOT raise DuplicateError — the
    overlap is silently de-duped, order-preserving."""
    ce, ch = _county_equity(), _county_homeless()
    # active target + a feature + an id col all repeated → all dropped from `extra`
    X, y, feats, ids = A.model_matrix(ce, ch, target="visits_pc", keep_targets=["visits_pc", "funding_pc", "fips", "wifi_pc"])
    assert feats == A.MODEL_FEATURES and y.len() == X.height
    # only wifi_pc survives as an extra column (the rest were reserved)
    assert ids.columns == ["fips", "NAME", "state", "wifi_pc"]


# ── whatif_impact (global what-if predictor) ────────────────────────────────
def test_whatif_impact_aggregates_and_pct():
    imp = A.whatif_impact(
        {"visits_pc": [1.0, 2.0], "wifi_pc": [0.0, 0.0]},
        {"visits_pc": [1.5, 2.5], "wifi_pc": [0.0, 0.0]},
        [100.0, 200.0],
        measures=["visits_pc", "wifi_pc"],
    )
    v = imp.filter(pl.col("measure") == "visits")
    assert v["base"][0] == pytest.approx(500.0) and v["scenario"][0] == pytest.approx(650.0)
    assert v["delta"][0] == pytest.approx(150.0) and v["pct"][0] == pytest.approx(30.0)
    # base == 0 → pct guard returns 0.0 rather than dividing by zero
    w = imp.filter(pl.col("measure") == "wifi")
    assert w["base"][0] == 0.0 and w["pct"][0] == 0.0


def test_did_panel_treats_on_funding_increase():
    """did_panel/event_panel generalize the treatment: instead of 'cut hours', flag counties that RAISED
    funding (treat_col='funding_pc', rising=True) — the 'did ADDING funding move the outcome?' lens. Counties
    i>=8 raise funding >5% at 2019 in the fixture; the default (hours cut) path must stay unchanged."""
    cp = _county_panel()
    # funding change at 2019 is (i-6)*5%; a 7% threshold cleanly selects i>=8 (10/15/20/25%) and avoids the
    # county sitting exactly on a 5% boundary (float-fuzzy).
    dp = A.did_panel(cp, baseline=2018, post=2019, unit="fips", treat_col="funding_pc", rising=True, treat_thresh=7.0)
    treated = set(dp.filter(pl.col("treated") == 1)["unit"].unique().to_list())
    assert treated == {_FIPS[i] for i in range(8, 12)}
    # event_panel honours the same generalization
    ep = A.event_panel(cp, baseline=2018, post=2019, unit="fips", measure="visits_pc", treat_col="funding_pc", rising=True, treat_thresh=7.0)
    assert set(ep.filter(pl.col("treated") == 1)["unit"].unique().to_list()) == {_FIPS[i] for i in range(8, 12)}
    # DEFAULT behaviour unchanged: hours-cut treatment still flags the even-index cutters
    dp0 = A.did_panel(cp, baseline=2009, post=2014, unit="fips")
    assert dp0.filter(pl.col("treated") == 1).height > 0
