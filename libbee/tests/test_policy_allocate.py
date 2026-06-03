"""Unit tests for libbee.analysis.policy_allocate — the Act-5 policy engine.

The engine turns a fitted model (passed as a pure ``predict(funding) -> per-capita`` callback) plus per-county
arrays into a funding plan, for three families of strategy: INVEST, CUT, and REALLOCATE. Every branch is
exercised against tiny synthetic inputs (no model, no disk) so the module stays at 100% coverage.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
import pytest

from libbee import analysis as A


def _meta(n):
    return pl.DataFrame(
        {
            "fips": [f"0100{i}" for i in range(n)],
            "NAME": [f"County {i}" for i in range(n)],
            "state": ["01"] * n,
        }
    )


def test_optimize_invests_budget_for_positive_effect():
    """optimize spends within budget and every funded county shows a positive predicted effect."""
    funding = np.array([10.0, 20.0, 30.0])
    pop = np.array([100.0, 100.0, 100.0])
    eff, summary = A.policy_allocate(
        lambda f: np.sqrt(f),  # diminishing-returns model: visits/cap = sqrt(funding)
        funding,
        pop,
        _meta(3),
        strategy="optimize",
        budget=10_000.0,
    )
    assert {"fips", "NAME", "state", "party", "dollars", "effect"} <= set(eff.columns)
    assert eff.height > 0
    assert (eff["dollars"] > 0).all() and (eff["effect"] > 0).all()
    assert summary["moved"] <= 10_000.0 + 1e-6
    assert summary["net"] > 0
    assert summary["touched"] == eff.height


def test_optimize_is_cumulative_via_returned_funding():
    """The plan is CUMULATIVE: feed the returned post-action funding back in and a second round compounds on
    top — funding only rises, never re-spends the same headroom, and never exceeds the fixed cap."""
    funding = np.array([5.0, 10.0, 15.0])
    pop = np.full(3, 100.0)

    def predict(f):
        return np.sqrt(f)

    eff1, s1 = A.policy_allocate(predict, funding, pop, _meta(3), strategy="optimize", budget=600.0, cap=20.0)
    eff2, s2 = A.policy_allocate(predict, s1["funding"], pop, _meta(3), strategy="optimize", budget=600.0, cap=20.0)
    assert np.all(s1["funding"] >= funding - 1e-9)  # round 1 raised funding
    assert np.all(s2["funding"] >= s1["funding"] - 1e-9)  # round 2 compounds on top of round 1
    assert s2["funding"].sum() > s1["funding"].sum()  # the second round actually did something
    assert np.all(s2["funding"] <= 20.0 + 1e-6)  # cumulative total never exceeds the fixed cap


def test_invest_respects_per_county_ceiling_and_floor():
    """ceiling caps any single county's slice; floor drops grants too small to bother funding."""
    pop = np.full(2, 100.0)

    def predict(f):
        return f  # linear → each $5/resident step costs $500

    # ceiling $600 lets through one $500 step per county, blocks the second (cum $1000) — so each caps at $500
    eff, _ = A.policy_allocate(predict, np.array([5.0, 5.0]), pop, _meta(2), strategy="optimize", budget=100_000.0, cap=20.0, ceiling=600.0)
    assert eff.height == 2 and (eff["dollars"] <= 600.0 + 1e-6).all()
    assert eff["dollars"].sum() == pytest.approx(1000.0)  # 2 counties × one $500 step
    # budget only funds one $500 step; a $600 floor then drops it → nothing funded, funding unchanged
    eff2, s2 = A.policy_allocate(predict, np.array([5.0, 5.0]), pop, _meta(2), strategy="optimize", budget=500.0, cap=20.0, floor=600.0)
    assert eff2.height == 0 and s2["touched"] == 0 and s2["moved"] == 0.0
    assert np.allclose(s2["funding"], [5.0, 5.0])


def test_poverty_strategy_funds_only_high_poverty_counties():
    """poverty restricts investment to counties at/above the `pctl` poverty percentile."""
    funding = np.full(4, 5.0)
    pop = np.full(4, 100.0)
    poverty = np.array([0.1, 0.2, 0.5, 0.9])  # counties 2 & 3 are the needy ones
    eff, _ = A.policy_allocate(lambda f: np.sqrt(f), funding, pop, _meta(4), strategy="poverty", budget=100_000.0, cap=20.0, poverty=poverty, pctl=0.5)
    funded = set(eff["fips"].to_list())
    assert funded and funded <= {"01002", "01003"}  # only at/above the median poverty
    assert "01000" not in funded and "01001" not in funded


def test_homeless_strategy_funds_only_high_homelessness_counties():
    """homeless restricts investment to counties at/above the `pctl` homelessness percentile."""
    homeless = np.array([1.0, 2.0, 80.0, 90.0])  # counties 2 & 3 high-homelessness
    eff, _ = A.policy_allocate(lambda f: np.sqrt(f), np.full(4, 5.0), np.full(4, 100.0), _meta(4), strategy="homeless", budget=100_000.0, cap=20.0, homeless=homeless, pctl=0.5)
    funded = set(eff["fips"].to_list())
    assert funded and funded <= {"01002", "01003"}


def test_cut_worst_removes_budget_and_reports_losses():
    """cut_worst pulls the budget out (across-the-board) and reports negative dollars + negative effect; the
    cumulative funding drops."""
    funding = np.array([10.0, 10.0])
    pop = np.full(2, 100.0)
    eff, s = A.policy_allocate(lambda f: f, funding, pop, _meta(2), strategy="cut_worst", budget=1000.0)
    assert eff.height > 0
    assert (eff["dollars"] < 0).all() and (eff["effect"] < 0).all()
    assert s["net"] < 0 and s["moved"] > 0
    assert np.all(s["funding"] <= funding + 1e-9)  # disinvestment lowers funding


def test_reallocate_cuts_one_lean_and_funds_the_other():
    """blue_to_red CUTS blue-leaning counties (negative) and INVESTS in red ones (positive); red_to_blue is
    the mirror image. The transfer is the political angle."""
    funding = np.array([12.0, 12.0, 6.0, 6.0])
    pop = np.full(4, 100.0)
    party = np.array(["blue", "blue", "red", "red"])

    def predict(f):
        return np.sqrt(f)

    b2r, _ = A.policy_allocate(predict, funding, pop, _meta(4), strategy="blue_to_red", budget=1000.0, cap=20.0, party=party)
    blue_max = cast(float, b2r.filter(pl.col("party") == "blue")["dollars"].max())
    red_min = cast(float, b2r.filter(pl.col("party") == "red")["dollars"].min())
    assert blue_max < 0  # blue cut
    assert red_min > 0  # red funded

    r2b, _ = A.policy_allocate(predict, funding, pop, _meta(4), strategy="red_to_blue", budget=1000.0, cap=20.0, party=party)
    red_max = cast(float, r2b.filter(pl.col("party") == "red")["dollars"].max())
    blue_min = cast(float, r2b.filter(pl.col("party") == "blue")["dollars"].min())
    assert red_max < 0  # red cut
    assert blue_min > 0  # blue funded


def test_degenerate_and_defensive_cases():
    """Every defensive branch: unknown strategy, zero budget, no headroom, zero-funding cut, and reallocation
    where one side has no counties (so an invest- or a cut-side comes back empty)."""
    pop = np.full(2, 100.0)

    def predict(f):
        return f

    meta = _meta(2)

    # unknown strategy raises
    with pytest.raises(ValueError):
        A.policy_allocate(predict, np.array([5.0, 5.0]), pop, meta, strategy="bogus", budget=100.0)

    # zero budget → optimal_allocation returns nothing → empty plan, funding untouched
    eff, s = A.policy_allocate(predict, np.array([5.0, 5.0]), pop, meta, strategy="optimize", budget=0.0, cap=20.0)
    assert eff.height == 0 and s["net"] == 0.0 and s["moved"] == 0.0 and s["touched"] == 0
    assert np.allclose(s["funding"], [5.0, 5.0])

    # everything already at the cap → no marginal steps exist → empty
    eff2, _ = A.policy_allocate(predict, np.array([20.0, 20.0]), pop, meta, strategy="optimize", budget=1000.0, cap=20.0)
    assert eff2.height == 0

    # cut where there is no funding to remove → no losses possible
    eff3, s3 = A.policy_allocate(predict, np.array([0.0, 0.0]), pop, meta, strategy="cut_worst", budget=1000.0)
    assert eff3.height == 0 and s3["net"] == 0.0

    party = np.array(["red", "red"])
    # blue_to_red on all-red data: the CUT side (blue) has no counties → cut returns empty; invest red gains
    b2r, _ = A.policy_allocate(predict, np.array([10.0, 10.0]), pop, meta, strategy="blue_to_red", budget=1000.0, cap=20.0, party=party)
    assert b2r.height > 0 and (b2r["party"] == "red").all() and (b2r["dollars"] > 0).all()
    # red_to_blue on all-red data: the INVEST side (blue) has no counties → invest returns empty; red is cut
    r2b, _ = A.policy_allocate(predict, np.array([10.0, 10.0]), pop, meta, strategy="red_to_blue", budget=1000.0, cap=20.0, party=party)
    assert r2b.height > 0 and (r2b["party"] == "red").all() and (r2b["dollars"] < 0).all()


def test_shap_values_explains_a_tree_model():
    """shap_values returns a (samples × features) SHAP array for a tree model, accepting either a Polars
    frame (taken through .to_pandas()) or a raw numpy array (the else branch).

    Skipped if shap is not installed (optional, install with pip install libbee[analysis]).
    """
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestRegressor

    Xpl = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    model = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=0).fit(Xpl.to_pandas(), y)

    sv, exp = A.shap_values(model, Xpl, feature_names=["a", "b"])  # Polars → .to_pandas() branch
    assert sv.shape == (6, 2) and exp is not None
    sv2, _ = A.shap_values(model, Xpl.to_numpy())  # numpy → else branch
    assert sv2.shape == (6, 2)
