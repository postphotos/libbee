"""Time-series & scope cuts — index/scope a metric, region trends, the explorer feed."""

from __future__ import annotations

import polars as pl

from ..geo import CENSUS_REGION, metro_code


def scope_series(state_panel, county_panel, county_equity, *, scope, state=None, county_name=None) -> tuple[pl.DataFrame, str]:
    """Resolve a geo scope to ONE yearly per-capita frame + a label.

    scope is "United States" | "One state" | "One county". Returns ``(frame, label)`` where frame has
    ``year, visits, checkouts, wifi, popu_lsa, visits_pc, checkouts_pc, wifi_pc`` sorted by year.

    Example::

        sp, cp, ce = (libbee.load(n) for n in ("state_panel", "county_panel", "county_equity"))
        nat, label = libbee.analysis.scope_series(sp, cp, ce, scope="United States")
        co, label = libbee.analysis.scope_series(sp, cp, ce, scope="One state", state="CA")
        # label is the human caption; chart `nat` with x="year", y="visits_pc"
    """
    cols = ["year", "visits", "checkouts", "wifi", "popu_lsa"]
    if scope == "One county" and county_name:
        hits = county_equity.filter(pl.col("NAME") == county_name)["fips"].to_list()
        fips = hits[0] if hits else "__none__"
        label = county_name.split(",")[0]
        src = county_panel.filter(pl.col("fips") == fips).select(cols)
    elif scope == "One state":
        label = state or "—"
        src = state_panel.filter(pl.col("st") == state).select(cols)
    else:
        label = "United States"
        src = state_panel.group_by("year").agg(pl.col("visits").sum(), pl.col("checkouts").sum(), pl.col("wifi").sum(), pl.col("popu_lsa").sum())
    out = (
        src.filter(pl.col("popu_lsa") > 0)
        .sort("year")
        .with_columns(
            (pl.col("visits") / pl.col("popu_lsa")).alias("visits_pc"),
            (pl.col("checkouts") / pl.col("popu_lsa")).alias("checkouts_pc"),
            (pl.col("wifi") / pl.col("popu_lsa")).alias("wifi_pc"),
        )
    )
    return out, label


def index_to_100(df, measures, *, by=None) -> pl.DataFrame:
    """Re-express each measure as an **index** where its first positive year = 100 — so series of wildly
    different scale can share one axis and be compared by **shape**, not magnitude.

    This is the move that makes the "physical down, WiFi up" chart honest: visits-per-resident (~4) and
    WiFi-sessions-per-resident (tiny at first) can't share a y-axis as raw numbers, but indexed to 100 they
    tell the truth about *direction*. Pass ``by`` to index **within groups** (e.g. one baseline per
    event-study cohort). Returns long/tidy ``[*by, year, measure, idx]`` with the ``_pc`` suffix stripped
    from measure names for nicer legends; a series with no positive baseline to divide by is dropped rather
    than dividing by zero.

    Example::

        # df has one row per year with the columns you name in `measures`
        idx = libbee.analysis.index_to_100(df, ["visits_pc", "checkouts_pc", "wifi_pc"])
        idx.filter(pl.col("measure") == "wifi")          # -> [year, measure, idx] starting at 100
        # within groups (a baseline per cohort): index_to_100(es, ["visits_pc"], by="cohort")
    """
    cols = ([by] if by else []) + ["year", "measure", "idx"]
    groups = df.partition_by(by) if by else [df]
    rows: list[dict] = []
    for g in groups:
        key = g[by][0] if by else None
        for m in measures:
            # NaN-safe: Polars NaN > 0 is True, so without is_finite() a NaN baseline would be picked and
            # poison the whole indexed series with NaN instead of being dropped like a zero/negative one.
            s = g.filter((pl.col(m) > 0) & pl.col(m).is_finite()).sort("year")
            if s.height == 0:
                continue
            base = s[m][0]
            label = m.replace("_pc", "")
            for r in s.iter_rows(named=True):
                row = {"year": r["year"], "measure": label, "idx": r[m] / base * 100}
                if by:
                    row[by] = key
                rows.append(row)
    if not rows:
        return pl.DataFrame(schema={c: (pl.Utf8 if c in (by, "measure") else pl.Float64) for c in cols})
    return pl.DataFrame(rows).select(cols)


def region_trends(state_panel, metro_panel, *, metros=None) -> pl.DataFrame:
    """§2 — the four **Census regions** as visits-per-resident time series, with optional **metro overlays**.

    A single national line hides that the Northeast, Midwest, South, and West sit on different levels and
    move at different speeds; this rolls each region up per year so those gaps are visible. Pass ``metros``
    (a list of metro names) to overlay specific metros labelled by airport-style code, so a city in the
    room can find itself. Returns long ``[series, year, visits_pc, kind]`` where ``kind`` is ``'region'``
    or ``'metro'`` — chart it with colour = ``series`` and use ``kind`` to draw regions bold and metros
    dashed.

    Example::

        sp, mp = libbee.load("state_panel"), libbee.load("metro_panel")
        trend = libbee.analysis.region_trends(sp, mp, metros=["Seattle-Tacoma-Bellevue"])  # SEA overlay
        trend.filter(pl.col("kind") == "region")["series"].unique()   # Northeast, Midwest, South, West
    """
    reg = (
        state_panel.with_columns(pl.col("st").replace_strict(CENSUS_REGION, default=None).alias("series"))
        .filter(pl.col("series").is_not_null())
        .group_by("series", "year")
        .agg((pl.col("visits").sum() / pl.col("popu_lsa").sum()).alias("visits_pc"))
        .with_columns(pl.lit("region").alias("kind"))
        .select(["series", "year", "visits_pc", "kind"])
    )
    if not metros:
        return reg.sort("kind", "series", "year")
    codes = {m: metro_code(m) for m in metros}
    met = (
        metro_panel.filter(pl.col("metro").is_in(metros) & (pl.col("visits_pc") > 0))
        .with_columns(pl.col("metro").replace_strict(codes, default=None).alias("series"), pl.lit("metro").alias("kind"))
        .select(["series", "year", "visits_pc", "kind"])
    )
    return pl.concat([reg, met], how="vertical_relaxed").sort("kind", "series", "year")


def explorer_series(facts, *, level, metric, geos, year_lo, year_hi) -> pl.DataFrame:
    """🔭 — pull a tidy ``[geo_name, year, value]`` slice from the unified facts table for the picker."""
    return (
        facts.filter((pl.col("geo_level") == level) & (pl.col("metric") == metric) & pl.col("geo_name").is_in(geos) & pl.col("year").is_between(year_lo, year_hi))
        .select(["geo_name", "year", "value"])
        .sort("geo_name", "year")
    )


def national_series(state_panel, *, metric="visits_pc", through=2019, since=None) -> pl.DataFrame:
    """National per-capita yearly series (visit-weighted) — the training input for the §M4 COVID
    counterfactual. Returns ``[year, value]`` for ``since <= year <= through`` (``since`` lets the
    forecast fit the recent post-peak regime so the counterfactual doesn't project a long-run rebound)."""
    raw = metric.replace("_pc", "")
    out = state_panel.group_by("year").agg((pl.col(raw).sum() / pl.col("popu_lsa").sum()).alias("value")).filter(pl.col("year") <= through).sort("year")
    return out.filter(pl.col("year") >= since) if since is not None else out


def acf_pacf(series, *, nlags=12) -> pl.DataFrame:
    """The **Box–Jenkins identification** view: sample autocorrelation (ACF) and partial autocorrelation
    (PACF) of a 1-D numeric ``series`` for lags ``1..nlags``. This is the "which ARIMA order?" diagnostic —
    a slowly-decaying ACF with a sharp PACF cutoff at lag *p* says AR(*p*); the mirror image says MA(*q*).

    ACF(k) = Σ (xₜ−μ)(xₜ₋ₖ−μ) / Σ (xₜ−μ)²  (the biased estimator); PACF via the Durbin–Levinson recursion
    on the ACF. Pure Python arithmetic (no numpy/statsmodels), so it stays in the tested ``analysis`` layer.
    Returns ``[lag, acf, pacf]``; the notebook draws the ±1.96/√n significance band. A constant series (zero
    variance) returns all-zero ACF/PACF rather than dividing by zero.

    Example::

        ap = libbee.analysis.acf_pacf(national["value"].to_list(), nlags=10)
        ap.filter(pl.col("lag") == 1)            # lag-1 autocorrelation; near 1 ⇒ needs differencing (d≥1)
    """
    # drop None / NaN up front — a missing observation must not silently zero out the whole diagnostic
    # (Polars NaN > 0 is True, so an unguarded NaN would masquerade as a flat, white-noise series).
    x = [float(v) for v in series if v is not None and float(v) == float(v)]
    n = len(x)
    m = sum(x) / n if n else 0.0
    dev = [v - m for v in x]
    c0 = sum(d * d for d in dev)
    k_max = min(nlags, n - 1) if n > 1 else 0
    # sample ACF (biased): rho[0]=1, rho[k]=gamma_k/gamma_0
    rho = [1.0]
    for k in range(1, k_max + 1):
        gk = sum(dev[t] * dev[t - k] for t in range(k, n))
        rho.append(gk / c0 if c0 > 0 else 0.0)
    # Durbin–Levinson recursion → PACF phi[k][k]
    phi = {}
    pacf = [0.0] * (k_max + 1)
    if k_max >= 1:
        phi[(1, 1)] = rho[1]
        pacf[1] = rho[1]
        for k in range(2, k_max + 1):
            num = rho[k] - sum(phi[(k - 1, j)] * rho[k - j] for j in range(1, k))
            den = 1.0 - sum(phi[(k - 1, j)] * rho[j] for j in range(1, k))
            pkk = num / den if den != 0 else 0.0
            phi[(k, k)] = pkk
            pacf[k] = pkk
            for j in range(1, k):
                phi[(k, j)] = phi[(k - 1, j)] - pkk * phi[(k - 1, k - j)]
    rows = [{"lag": k, "acf": rho[k], "pacf": pacf[k]} for k in range(1, k_max + 1)]
    if not rows:
        return pl.DataFrame(schema={"lag": pl.Int64, "acf": pl.Float64, "pacf": pl.Float64})
    return pl.DataFrame(rows).select(["lag", "acf", "pacf"])


def aic_table(results) -> pl.DataFrame:
    """Format an **AIC order-selection** grid for Box–Jenkins estimation. ``results`` is a sequence of
    dicts ``{p, d, q, aic}`` (the notebook fits each candidate ARIMA(p,d,q) — statsmodels lives in the
    notebook, not here — and collects their AICs). Sorts ascending by AIC, adds ``delta_aic`` (= aic − min,
    the penalty for not picking the best) and a ``best`` flag on the minimum. Returns
    ``[p, d, q, aic, delta_aic, best]``; an empty input returns the empty typed frame.

    Example::

        cand = [{"p": p, "d": 1, "q": q, "aic": ARIMA(y, order=(p, 1, q)).fit().aic}
                for p in range(3) for q in range(3)]
        grid = libbee.analysis.aic_table(cand)
        best = grid.row(0, named=True)        # lowest-AIC order → the recommended (p,d,q)
    """
    cols = ["p", "d", "q", "aic", "delta_aic", "best"]
    rows = list(results)
    if not rows:
        return pl.DataFrame(schema={"p": pl.Int64, "d": pl.Int64, "q": pl.Int64, "aic": pl.Float64, "delta_aic": pl.Float64, "best": pl.Boolean})
    df = pl.DataFrame(rows).sort("aic")
    amin = df["aic"].min()
    return df.with_columns(
        (pl.col("aic") - amin).alias("delta_aic"),
        (pl.col("aic") == amin).alias("best"),
    ).select(cols)
