"""ML prep & allocators (Part II) — the model matrix, personas, and the budget allocators.

This is the modelling layer's *data* half: it turns the wide, holey county frame into a clean design matrix
(`model_matrix`), names KMeans clusters as readable personas (`cluster_labels`), and splits a fixed budget
across counties by **visits-per-dollar** — greedily for whole-county grants (`greedy_allocation`) or by
water-filling diminishing marginal steps (`optimal_allocation`, `policy_allocate`). The allocators rank by
$\\text{vpd} = \\Delta\\text{visits} / \\Delta\\text{dollars}$ and take the steepest steps first, which is
why money spreads to counties that *keep* responding rather than just the cheapest ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..geo import FIPS2ST

# The seven candidate drivers of library use the model is allowed to choose from — three "need" signals
# (poverty, income, broadband access), one geography signal (density), the two "supply" levers a town
# actually controls (funding_pc, hours_pc), and the one the culture-war argument is about (homeless_per10k).
# The whole point of Act 3 is to let a model weigh these freely and watch homelessness land near zero.
MODEL_FEATURES = [
    "poverty",  # % of residents below the poverty line (Census ACS)
    "median_income",  # median household income $ (Census ACS)
    "no_broadband",  # % of households without a broadband subscription (Census ACS)
    "density",  # residents per square mile (geography / urbanicity proxy)
    "funding_pc",  # public operating revenue per resident $ (the money lever)
    "hours_pc",  # annual open hours per resident (the access lever)
    "homeless_per10k",  # people experiencing homelessness per 10k residents (HUD)
]


def model_matrix(county_equity, county_homeless, *, year=2019, target="visits_pc", keep_targets=None) -> tuple[pl.DataFrame, pl.Series, list[str], pl.DataFrame]:
    """Assemble the model-ready `(X, y, feature_names, ids)` tuple every Act-3 ML cell starts from.

    scikit-learn wants a rectangular, all-numeric matrix with **no missing values**, but the raw county
    frame is wide, mixed-type, and full of holes. This is the single *tested* place that turns "the data"
    into "a design matrix," so the notebook never babysits NaNs or column lists — and it guarantees the
    RandomForest, KMeans, and the budget allocator all train on **exactly the same rows in the same feature
    order**. Any row with a null feature or target is dropped, so $X$ is dense.

    Args:
        county_equity: The wide per-county frame (one row per county) carrying every `MODEL_FEATURES`
            column, the `target`, `fips`, and `NAME`.
        county_homeless: The county homelessness panel; the `year` slice supplies `homeless_per10k`, joined
            on `fips` (inner).
        year: Cross-section to model. Default **2019**, the last clean pre-COVID year.
        target: Outcome to predict — `"visits_pc"` | `"checkouts_pc"` | `"wifi_pc"`. The `> 0` row guard
            applies to whichever target you pass.
        keep_targets: Extra outcome columns to carry onto `ids` (de-duplicated against the reserved/feature
            columns), so a multi-target notebook can train several forests on the **same** rows.

    Returns:
        A 4-tuple `(X, y, feature_names, ids)`: `X` is one row per county with columns `MODEL_FEATURES`
        (every value present); `y` is the `target` column, row-aligned to `X`; `feature_names` is `X`'s
        column order, so `rf.feature_importances_` lines up one-to-one; `ids` is
        `[fips, NAME, state, *keep_targets]`, row-aligned to `X` (counties whose `state` can't be mapped
        from the FIPS prefix are dropped).

    Example:
        ```python
        from sklearn.ensemble import RandomForestRegressor
        county = libbee.load("county_equity"); ch = libbee.load("county_homeless")
        X, yv, feats, ids = libbee.analysis.model_matrix(
            county, ch, year=2019, target="visits_pc", keep_targets=["checkouts_pc", "wifi_pc"])
        rf = RandomForestRegressor(max_depth=5, random_state=0).fit(X.to_pandas(), yv.to_numpy())
        dict(zip(feats, rf.feature_importances_.round(3)))   # -> funding_pc dominates, homeless ~0
        ```
    """
    # de-dup keep_targets against the columns we already select (the target, the features, the id cols),
    # order-preserving — so a caller who passes the active target (or a feature) in keep_targets gets a
    # clean result instead of a Polars DuplicateError.
    _reserved = {"fips", "NAME", "state", target, *MODEL_FEATURES}
    extra = [c for c in dict.fromkeys(keep_targets or []) if c not in _reserved]
    ch = county_homeless.filter(pl.col("year") == year).select(["fips", "homeless_per10k"])
    d = (
        county_equity.join(ch, on="fips", how="inner")
        .filter(pl.col(target) > 0)
        .with_columns(pl.col("fips").str.slice(0, 2).replace_strict(FIPS2ST, default=None).alias("state"))
        .select(["fips", "NAME", "state", *MODEL_FEATURES, target, *extra])
        .drop_nulls()
    )
    ids = d.select(["fips", "NAME", "state", *extra])
    return d.select(MODEL_FEATURES), d[target], list(MODEL_FEATURES), ids


# Plain-English description for every persona ``cluster_labels`` can emit. Tertiles are RELATIVE — ranked
# across the k clusters in a given run — so "underfunded" means low-funded *vs the other archetypes here*,
# not below an absolute dollar line. Keep keys in sync with the label rules below (a test enforces it).
PERSONA_GLOSSARY: dict[str, str] = {
    "well-funded, high-use": "Top-tertile funding/resident AND top-tertile visits/resident — money in, use out.",
    "well-funded, under-used": "Top-tertile funding but bottom-tertile visits — spends a lot for little traffic.",
    "underfunded, high-need": "Top-tertile poverty AND bottom-tertile funding — high need, least money.",
    "dense, high-need urban": "Top-tertile density AND top-tertile poverty — crowded, high-need metros.",
    "dense urban core": "Top-tertile density (without the high-poverty flag) — big-city library systems.",
    "efficient: high-use, modest funds": "Top-tertile visits without top-tertile funding — high use per dollar.",
    "underfunded": "Bottom-tertile funding/resident (need not flagged high) — simply low-funded.",
    "higher-need, mid-funded": "Top-tertile poverty with middling funding — needier than its budget suggests.",
    "lean rural": "Bottom-tertile density — sparse, rural service areas.",
    "quiet / low-use": "Bottom-tertile visits/resident — little foot traffic, need not extreme.",
    "mid-market / suburban": "No tertile extreme on any axis — the middle-of-the-road catch-all.",
}


def cluster_labels(profile, *, funding="funding_pc", visits="visits_pc", poverty="poverty", density="density") -> pl.DataFrame:
    """Turn a per-cluster *means* frame into human **persona labels** — so a cluster reads as a name.

    KMeans gives you anonymous groups; an audience wants names. For each of funding / visits / poverty /
    density this ranks the clusters into **tertiles** (`lo` / `mid` / `hi`) — a cluster is `lo` when its
    rank fraction $r = \\text{rank} / n \\le \\tfrac13$, `hi` when $r > \\tfrac23$ — then composes a short
    English label from those ranks (low funding + high poverty → "underfunded, high-need"; high funding +
    high visits → "well-funded, high-use"). Rules run most-specific first; see `PERSONA_GLOSSARY` for the
    full vocabulary. Tertiles are **relative** to the clusters in this run, not absolute dollar lines.

    Args:
        profile: One row per cluster with the **mean** of each feature (e.g. from a `group_by("cluster")`).
            Must carry a `cluster` column plus the four feature columns named by the kwargs below.
        funding: Column name holding mean funding-per-resident. Default `"funding_pc"`.
        visits: Column name holding mean visits-per-resident. Default `"visits_pc"`.
        poverty: Column name holding mean poverty rate. Default `"poverty"`.
        density: Column name holding mean population density. Default `"density"`.

    Returns:
        A Polars frame `[cluster, label]` — one readable persona string per cluster.

    Example:
        ```python
        profile = personas.group_by("cluster").agg(
            pl.col("funding_pc").mean(), pl.col("visits_pc").mean(),
            pl.col("poverty").mean(), pl.col("density").mean())
        labels = libbee.analysis.cluster_labels(profile)   # -> [cluster, label]
        personas.join(labels, on="cluster")               # attach the human name to every county
        ```
    """

    def _t(col):
        _r = pl.col(col).rank() / pl.len()
        return pl.when(_r <= 1 / 3).then(pl.lit("lo")).when(_r <= 2 / 3).then(pl.lit("mid")).otherwise(pl.lit("hi"))

    p = profile.with_columns(_t(funding).alias("_f"), _t(visits).alias("_v"), _t(poverty).alias("_p"), _t(density).alias("_d"))
    # most-specific → least: the extra mid-band rules break the old "mixed / suburban" catch-all into
    # readable personas (well-funded-under-used, efficient, underfunded, higher-need, quiet) so few clusters
    # land in the generic bucket.
    label = (
        pl.when((pl.col("_f") == "hi") & (pl.col("_v") == "hi"))
        .then(pl.lit("well-funded, high-use"))
        .when((pl.col("_f") == "hi") & (pl.col("_v") == "lo"))
        .then(pl.lit("well-funded, under-used"))
        .when((pl.col("_p") == "hi") & (pl.col("_f") == "lo"))
        .then(pl.lit("underfunded, high-need"))
        .when((pl.col("_d") == "hi") & (pl.col("_p") == "hi"))
        .then(pl.lit("dense, high-need urban"))
        .when(pl.col("_d") == "hi")
        .then(pl.lit("dense urban core"))
        .when((pl.col("_v") == "hi") & (pl.col("_f") != "hi"))
        .then(pl.lit("efficient: high-use, modest funds"))
        .when(pl.col("_f") == "lo")
        .then(pl.lit("underfunded"))
        .when(pl.col("_p") == "hi")
        .then(pl.lit("higher-need, mid-funded"))
        .when(pl.col("_d") == "lo")
        .then(pl.lit("lean rural"))
        .when(pl.col("_v") == "lo")
        .then(pl.lit("quiet / low-use"))
        .otherwise(pl.lit("mid-market / suburban"))
        .alias("label")
    )
    return p.with_columns(label).select(["cluster", "label"])


def allocation_frame(county_equity, ids, X, *, delta=5.0, cap_q=0.90) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the bumped feature matrix + metadata for the §"where to spend" marginal-impact allocator.
    Bumps ``funding_pc`` by ``delta`` per resident, **clamped at the ``cap_q`` quantile** so we never ask
    the (non-extrapolating) RandomForest about a funding level it never saw. Returns
    ``(X_bumped [same columns/order as X], meta[fips, NAME, state, funding_pc, delta_applied, pop])``.

    Example::

        X, y, feats, ids = libbee.analysis.model_matrix(county, ch, year=2019)
        rf = RandomForestRegressor().fit(X.to_pandas(), y.to_numpy())
        X_bumped, meta = libbee.analysis.allocation_frame(county, ids, X, delta=5.0)   # +$5/resident
        extra = (rf.predict(X_bumped.to_pandas()) - rf.predict(X.to_pandas())) * meta["pop"]  # extra visits
    """
    cap = float(X["funding_pc"].quantile(cap_q))
    # bump up to the cap, but never below the county's current funding (counties already above the
    # cap get delta_applied = 0 and are excluded downstream) — so delta_applied is always >= 0.
    bumped = pl.max_horizontal(pl.min_horizontal(pl.col("funding_pc") + delta, pl.lit(cap)), pl.col("funding_pc"))
    X_bumped = X.with_columns(bumped.alias("funding_pc")).select(X.columns)
    pop = county_equity.select(["fips", pl.col("popu_lsa").alias("pop")])
    meta = ids.with_columns(X["funding_pc"].alias("funding_pc"), (X_bumped["funding_pc"] - X["funding_pc"]).alias("delta_applied")).join(pop, on="fips", how="left")
    return X_bumped, meta


def greedy_allocation(marginals, *, budget) -> tuple[pl.DataFrame, dict]:
    """Greedily allocate a fixed `budget` to **whole counties** by descending predicted visits-per-dollar.

    Each county is scored by its efficiency $\\text{vpd} = \\text{extra\\_visits} / \\text{dollars\\_needed}$,
    sorted high → low, and funded in that order until the running cost would exceed `budget`. Counties with
    non-positive cost or predicted gain are dropped first. This is the whole-grant ("fund a county or
    don't") allocator; for diminishing per-dollar steps use `optimal_allocation`.

    Args:
        budget: Total dollars available; counties are taken while cumulative `dollars_needed` stays
            $\\le$ `budget`.
        marginals: One row per candidate county `[fips, NAME, state, dollars_needed, extra_visits, ...]`,
            where `dollars_needed` is the grant size and `extra_visits` the model's predicted gain.

    Returns:
        A 2-tuple `(chosen, summary)`. `chosen` is the funded subset with added `vpd` and `cum_cost`
        columns; `summary` is `{"n_counties", "spent", "extra_visits"}` with `spent` the dollars used and
        `extra_visits` the total predicted gain.
    """
    m = (
        marginals.filter((pl.col("dollars_needed") > 0) & (pl.col("extra_visits") > 0))
        .with_columns((pl.col("extra_visits") / pl.col("dollars_needed")).alias("vpd"))
        .sort("vpd", descending=True)
        .with_columns(pl.col("dollars_needed").cum_sum().alias("cum_cost"))
    )
    chosen = m.filter(pl.col("cum_cost") <= budget)
    summary = {
        "n_counties": chosen.height,
        "spent": float(chosen["dollars_needed"].sum()) if chosen.height else 0.0,
        "extra_visits": float(chosen["extra_visits"].sum()) if chosen.height else 0.0,
    }
    return chosen, summary


def optimal_allocation(increments, *, budget) -> tuple[pl.DataFrame, dict]:
    """Water-filling budget split. ``increments`` is a long frame of marginal funding **steps** — each
    row one ``$Δ`` step for one county (``fips, NAME, state, dollars, extra_visits``, with later steps
    carrying smaller ``extra_visits`` as the model saturates). Greedily take the highest
    visits-per-dollar steps until the budget is spent: because the response diminishes, money naturally
    spreads to counties that **keep responding** and stops once a county plateaus — the optimal split
    for a fixed budget over separable increments, and it targets *responsiveness*, not just low funding.
    Returns ``(plan[fips, NAME, state, dollars, extra_visits, steps], summary)``.

    Example::

        # `increments` = one row per ($Δ step, county): [fips, NAME, state, dollars, extra_visits]
        plan, summary = libbee.analysis.optimal_allocation(increments, budget=10_000_000)
        summary            # e.g. {'n_counties': ~92, 'spent': ~5.6e6, 'extra_visits': ~3.2e6}
        plan.head()        # the funded counties, each with its $ and predicted extra visits
    """
    inc = (
        increments.filter((pl.col("dollars") > 0) & (pl.col("extra_visits") > 0))
        .with_columns((pl.col("extra_visits") / pl.col("dollars")).alias("vpd"))
        .sort("vpd", descending=True)
        .with_columns(pl.col("dollars").cum_sum().alias("cum"))
    )
    taken = inc.filter(pl.col("cum") <= budget)
    plan = (
        taken.group_by("fips", "NAME", "state")
        .agg(pl.col("dollars").sum().alias("dollars"), pl.col("extra_visits").sum().alias("extra_visits"), pl.len().alias("steps"))
        .sort("extra_visits", descending=True)
    )
    summary = {
        "n_counties": plan.height,
        "spent": float(taken["dollars"].sum()) if taken.height else 0.0,
        "extra_visits": float(taken["extra_visits"].sum()) if taken.height else 0.0,
    }
    return plan, summary


def policy_allocate(
    predict,
    funding,
    pop,
    meta,
    *,
    strategy,
    budget,
    party=None,
    poverty=None,
    homeless=None,
    pctl=0.75,
    floor=0.0,
    ceiling=None,
    cap=None,
    cap_quantile=0.90,
    step=5.0,
    max_steps=48,
):
    """The Act-5 policy engine: turn a fitted model into a per-county funding plan.

    The *only* coupling to the model is ``predict`` — a pure callback ``predict(funding_array) -> per-capita
    prediction array``. Everything else is plain arrays aligned to ``meta`` (a frame with ``[fips, NAME,
    state]``). Returns ``(effect, summary)`` where ``effect`` is one row per affected county
    ``[fips, NAME, state, party, dollars(±), effect(±)]`` and ``summary`` carries the headline numbers plus
    the **post-action funding vector** so calls *chain* (invest, then cut, then …) — that's what makes the
    tool cumulative.

    Currently implements ``strategy="optimize"`` (invest the budget for the biggest predicted gain by
    water-filling the highest visits-per-dollar marginal steps).
    """
    import numpy as np  # lazy: a module-level numpy import trips a coverage/C-extension double-load

    funding = np.asarray(funding, dtype=float)
    pop = np.asarray(pop, dtype=float)
    n = funding.shape[0]
    base = np.asarray(predict(funding), dtype=float)
    _fips, _name, _state = meta["fips"], meta["NAME"], meta["state"]
    _party = np.asarray(party) if party is not None else np.array(["?"] * n)

    _INC = {"fips": pl.Utf8, "NAME": pl.Utf8, "state": pl.Utf8, "dollars": pl.Float64, "extra_visits": pl.Float64, "step": pl.Int64}
    _EFF = {"fips": pl.Utf8, "NAME": pl.Utf8, "state": pl.Utf8, "party": pl.Utf8, "dollars": pl.Float64, "effect": pl.Float64}

    def _up(mask):
        # marginal +``step``/resident funding increments for masked counties, up to the cap; each later step
        # carries less ``extra_visits`` as the model saturates. Capping at observed support keeps a tree model
        # from extrapolating into fantasy.
        if not mask.any():
            return pl.DataFrame(schema=_INC)
        # a fixed `cap` keeps the per-county ceiling stable across cumulative rounds; otherwise infer it from
        # the current funding's `cap_quantile`.
        _cap = float(cap) if cap is not None else float(np.quantile(funding, cap_quantile))
        prev_f, prev, frames, s = funding.copy(), base, [], 0
        while np.any((prev_f < _cap - 1e-9) & mask) and s < max_steps:
            nf = prev_f.copy()
            nf[mask] = np.minimum(prev_f[mask] + step, _cap)
            npred = np.asarray(predict(nf), dtype=float)
            frames.append(
                pl.DataFrame(
                    {
                        "fips": _fips,
                        "NAME": _name,
                        "state": _state,
                        "dollars": (nf - prev_f) * pop,
                        "extra_visits": np.clip(npred - prev, 0.0, None) * pop,
                        "step": np.full(n, s + 1, dtype=np.int64),
                    },
                    schema=_INC,
                ).filter((pl.col("dollars") > 0) & (pl.col("extra_visits") > 0))
            )
            prev_f, prev, s = nf, npred, s + 1
        return pl.concat(frames) if frames else pl.DataFrame(schema=_INC)

    def _invest(mask, b):
        inc = _up(mask)
        if inc.height and ceiling is not None:  # ceiling: stop one county from eating the budget
            inc = inc.sort(["fips", "step"]).with_columns(pl.col("dollars").cum_sum().over("fips").alias("_cum")).filter(pl.col("_cum") <= ceiling)
        if not inc.height:
            return pl.DataFrame(schema=_EFF)
        plan, _ = optimal_allocation(inc.select(["fips", "NAME", "state", "dollars", "extra_visits"]), budget=b)
        if plan.height and floor > 0:  # floor: drop grants too small to bother with
            plan = plan.filter(pl.col("dollars") >= floor)
        if not plan.height:
            return pl.DataFrame(schema=_EFF)
        return plan.join(pl.DataFrame({"fips": _fips, "party": _party}), on="fips", how="left").select(
            "fips", "NAME", "state", "party", "dollars", pl.col("extra_visits").alias("effect")
        )

    def _cut(mask, b):
        # across-the-board fractional cut to raise ``b`` from the masked counties; effect = visits LOST.
        if not mask.any():
            return pl.DataFrame(schema=_EFF)
        total = float((funding * pop)[mask].sum())
        if total <= 0:
            return pl.DataFrame(schema=_EFF)
        frac = min(1.0, b / total)
        nf = funding.copy()
        nf[mask] = funding[mask] * (1.0 - frac)
        npred = np.asarray(predict(nf), dtype=float)
        return pl.DataFrame(
            {
                "fips": _fips,
                "NAME": _name,
                "state": _state,
                "party": _party,
                "dollars": -(funding - nf) * pop,  # negative = money removed
                "effect": -np.clip(base - npred, 0.0, None) * pop,  # negative = visits lost
            },
            schema=_EFF,
        ).filter(pl.col("dollars") < -1.0)

    def _need_mask(arr):
        need = np.asarray(arr, dtype=float)
        return need >= np.quantile(need, pctl)

    _allm = np.ones(n, dtype=bool)
    if strategy == "optimize":
        eff = _invest(_allm, budget)
    elif strategy == "poverty":
        eff = _invest(_need_mask(poverty), budget)
    elif strategy == "homeless":
        eff = _invest(_need_mask(homeless), budget)
    elif strategy == "cut_worst":
        eff = _cut(_allm, budget)
    elif strategy == "blue_to_red":  # pull budget OUT of blue, INTO red — net the loss against the gain
        eff = pl.concat([_cut(_party == "blue", budget), _invest(_party == "red", budget)], how="vertical_relaxed")
    elif strategy == "red_to_blue":
        eff = pl.concat([_cut(_party == "red", budget), _invest(_party == "blue", budget)], how="vertical_relaxed")
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    # cumulative: the funding vector AFTER applying this plan (signed dollars / pop), so the caller can feed
    # it straight back in for the next action.
    delta = np.zeros(n)
    if eff.height:
        _by = {r["fips"]: r["dollars"] for r in eff.select(["fips", "dollars"]).iter_rows(named=True)}
        delta = np.array([_by.get(f, 0.0) for f in _fips.to_list()]) / np.where(pop > 0, pop, 1.0)
    summary = {
        "touched": eff.height,
        "moved": float(eff["dollars"].abs().sum()) if eff.height else 0.0,
        "net": float(eff["effect"].sum()) if eff.height else 0.0,
        "funding": funding + delta,
    }
    return eff, summary


def fitting_score(ids, *, actual, pred0, potential, dollars_to_cap, target=0.80, green=0.90, yellow=0.70) -> pl.DataFrame:
    """Score each unit's **service level ("fitting")**, rate it green/yellow/red, and price the funding gap.

    A library's *potential* is what an honest model predicts it would do at **adequate funding** (need and
    density held fixed), so a unit's fitting fraction is

    $\\text{fitting\\_pct} = \\operatorname{clip}(\\text{actual} / \\text{potential},\\,0,\\,1)$

    and the shortfall is a *funding* gap you can cost. This is the pure, tested core — the notebook owns the
    model and supplies three per-unit prediction arrays; this turns them into a rated, costed frame. A unit
    with non-positive or non-finite `potential` is treated as fully fit (no headroom to claim) instead of
    dividing by zero.

    Args:
        ids: Identity frame `[fips, NAME, state, …]`; every array below is row-aligned to it.
        actual: Observed outcome per unit (e.g. `visits_pc`) — the numerator.
        pred0: The model's prediction at the unit's **current** funding.
        potential: The model's prediction at the **in-support funding cap** (the adequately-funded ceiling).
        dollars_to_cap: Dollars to raise the unit to that cap, $(\\text{cap} - \\text{funding\\_pc})\\cdot
            \\text{pop}$.
        target: Fitting fraction every unit should clear. Default `0.80` ("tolerable, not perfect").
        green: Rating threshold — `fitting_pct` $\\ge$ `green` (default 0.90) is green.
        yellow: Rating threshold — `fitting_pct` $\\ge$ `yellow` (default 0.70) is yellow, else red.

    Returns:
        A Polars frame `[fips, NAME, state, …, actual, potential, fitting_pct, rating, cost_to_target]`.
        `cost_to_target` is the share of `dollars_to_cap` needed to lift the prediction to
        $\\text{target}\\cdot\\text{potential}$ (linear between `pred0` and `potential`; 0 if already at/above
        target or if there's no headroom).

    Example:
        ```python
        sl = libbee.analysis.fitting_score(ids, actual=y.to_numpy(), pred0=pred_now,
                                           potential=pred_cap, dollars_to_cap=gap_dollars, target=0.80)
        sl.group_by("rating").len()                              # how many green / yellow / red
        sl.filter(pl.col("fitting_pct") < 0.80)["cost_to_target"].sum()   # $ to make everyone tolerable
        ```
    """
    df = ids.with_columns(
        pl.Series("actual", actual, dtype=pl.Float64),
        pl.Series("pred0", pred0, dtype=pl.Float64),
        pl.Series("potential", potential, dtype=pl.Float64),
        pl.Series("dollars_to_cap", dollars_to_cap, dtype=pl.Float64),
    )
    _head = pl.col("potential") - pl.col("pred0")  # prediction headroom from current funding to the cap
    # a real positive potential is the precondition for BOTH "fitting %" and "cost to target" — note Polars
    # treats NaN > 0 as True, so we add an explicit is_finite() or a NaN/zero/negative potential would slip
    # through and be rated 'green' with a NaN score. Both branches share the guard so they can't disagree.
    _ok = (pl.col("potential") > 0) & pl.col("potential").is_finite()
    df = df.with_columns(
        pl.when(_ok)
        .then((pl.col("actual") / pl.col("potential")).clip(0.0, 1.0))
        .otherwise(1.0)  # no real headroom to judge against → treat as fully fit (no headroom claim)
        .alias("fitting_pct"),
        # fraction of the way to the cap needed to reach target·potential predicted outcome
        pl.when(_ok & (_head > 0)).then(((target * pl.col("potential") - pl.col("pred0")) / _head).clip(0.0, 1.0)).otherwise(0.0).alias("_frac"),
    )
    return df.with_columns(
        pl.when(pl.col("fitting_pct") >= green).then(pl.lit("green")).when(pl.col("fitting_pct") >= yellow).then(pl.lit("yellow")).otherwise(pl.lit("red")).alias("rating"),
        (pl.col("_frac") * pl.col("dollars_to_cap")).alias("cost_to_target"),
    ).drop(["pred0", "dollars_to_cap", "_frac"])


def whatif_impact(base_preds, new_preds, pop, *, measures) -> pl.DataFrame:
    """Aggregate a **what-if** funding shock across the whole dataset, per outcome. ``base_preds`` and
    ``new_preds`` are dicts ``{measure: per-unit prediction array}`` (the notebook produces them with one
    ``rf.predict`` per measure, before/after perturbing the feature matrix); ``pop`` is the per-unit
    population. Returns ``[measure, base, scenario, delta, pct]`` — national totals (``pred·pop`` summed)
    before and after, with the change. Pure: no model here, so it tests on plain lists.

    Example::

        impact = libbee.analysis.whatif_impact(base, new, pop,
                                               measures=["visits_pc", "checkouts_pc", "wifi_pc"])
        impact  # one row per measure: predicted national visits/checkouts/wifi at the new funding & hours
    """
    p = pl.Series("pop", pop, dtype=pl.Float64)
    rows = []
    for m in measures:
        base = float((pl.Series(base_preds[m], dtype=pl.Float64) * p).sum())
        scen = float((pl.Series(new_preds[m], dtype=pl.Float64) * p).sum())
        rows.append({"measure": m.replace("_pc", ""), "base": base, "scenario": scen, "delta": scen - base, "pct": (scen - base) / base * 100 if base else 0.0})
    return pl.DataFrame(rows, schema=["measure", "base", "scenario", "delta", "pct"])


def shap_values(model, X, *, feature_names=None) -> tuple[np.ndarray, Any]:
    """Pure helper for SHAP explainability. Returns (shap_values, explainer_object).

    Why this exists: SHAP explainer boilerplate and feature name alignment should be consistent.
    """
    import numpy as np

    try:
        import shap
    except ModuleNotFoundError:  # pragma: no cover
        raise ImportError("shap is required for SHAP explainability. Install it with: pip install libbee[analysis]") from None

    # RandomForests are tree-based; use TreeExplainer for speed
    explainer = shap.TreeExplainer(model)
    # X can be Polars, SHAP wants NumPy or Pandas with names
    X_input = X.to_pandas() if hasattr(X, "to_pandas") else X
    sv = explainer.shap_values(X_input)

    # Defensive shim for shap builds that return a per-output LIST instead of an array (the pinned shap
    # returns an ndarray, so this guard is version-compat only).
    if isinstance(sv, list):  # pragma: no cover
        sv = np.array(sv)

    return sv, explainer
