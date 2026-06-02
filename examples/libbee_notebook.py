import marimo

__generated_with = "0.23.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def imports():
    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, mo, pl


@app.cell
def libbee_imports():
    import libbee

    return (libbee,)


@app.cell
def intro(mo):
    mo.md("""
    # 🐝 libbee · the facts explorer
    A small reactive dashboard over **`libbee.facts()`** — the unified long table of US public-library
    metrics (IMLS + HUD + Census), 1992–2023, at library / county / metro / state grain. Pick a
    geography level, one or more metrics, a year range, and (optionally) a state; the trend chart,
    summary, preview, and exports all react.
    """)
    return


@app.cell
def load_facts(libbee, mo):
    """Load the unified facts table (one tidy row per geo × year × metric)."""
    with mo.status.spinner(title="Loading the unified facts table..."):
        facts = libbee.facts()

    facts
    return (facts,)


@app.cell
def apply_filters(facts, geo_dd, metric_ms, pl, state_input, year_slider):
    """Filter the facts table to the chosen slice.

    Drops non-finite values (e.g. ``cost_per_visit`` is ``inf`` when a geography reports zero visits) so
    the summary stats and chart can't be poisoned by an infinity.
    """
    result = facts.filter(
        (pl.col("geo_level") == geo_dd.value)
        & (pl.col("metric").is_in(metric_ms.value))
        & (pl.col("year") >= year_slider.value[0])
        & (pl.col("year") <= year_slider.value[1])
        & pl.col("value").is_finite()
    )

    if state_input.value.strip():
        result = result.filter(pl.col("state") == state_input.value.upper().strip())
    return (result,)


@app.cell
def data_preview(mo, result):
    """Collapsible preview of the filtered rows (no-code Polars viewer)."""
    if result.is_empty():
        preview_view = mo.callout(mo.md("No data to preview."), kind="info")
    else:
        preview_view = mo.accordion({f"📋 Data preview (first 100 of {result.height:,} rows)": mo.ui.dataframe(result.head(100))})
    preview_view
    return


@app.cell(column=1)
def trend_chart(alt, mo, pl, result, summary):
    """Multi-metric trend: mean value per year per metric, one line each (chart left, summary right)."""
    if result.is_empty():
        trend_view = mo.callout(mo.md("No data to chart."), kind="warn")
    else:
        agg = result.group_by("year", "metric").agg(pl.col("value").mean().alias("avg_value")).sort("year", "metric")
        base = (
            alt.Chart(agg.to_pandas())
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("avg_value:Q", title=None),
                color=alt.Color("metric:N", legend=None),
                tooltip=["year:O", "metric:N", alt.Tooltip("avg_value:Q", format=".2f")],
            )
            .properties(width=500, height=160)
        )
        chart = (
            base.facet(row=alt.Row("metric:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")))
            .resolve_scale(y="independent")
            .properties(title="Trend over time (Independent Scales)")
        )
        # NB: metrics live on different scales (e.g. checkouts/resident vs cost/visit); they now have
        # independent faceted axes, so you can clearly see their individual shapes without scale-crushing.
        trend_view = mo.hstack([chart, summary], justify="start", gap=2, align="start")
    trend_view
    return


@app.cell
def _():
    return


@app.cell(column=2)
def filter_controls(geo_levels, metrics, mo, year_range):
    """Build the filter UI. Metric is a *multiselect* so the trend chart can overlay several series."""
    geo_dd = mo.ui.dropdown(
        options={gl: gl for gl in geo_levels},
        value="state" if "state" in geo_levels else (geo_levels[0] if geo_levels else None),
        label="📍 Geography level",
        full_width=True,
    )

    metric_ms = mo.ui.multiselect(
        options={m: m for m in metrics},
        value=metrics[:3] if len(metrics) >= 3 else metrics,
        label="📊 Metrics (pick one or more)",
        full_width=True,
    )

    year_slider = mo.ui.range_slider(
        start=year_range[0],
        stop=year_range[1],
        step=1,
        value=[year_range[0], year_range[1]],
        label="📅 Year range",
        full_width=True,
        show_value=True,
    )

    state_input = mo.ui.text(
        placeholder="e.g. 'CA' (blank = all)",
        label="🗺️ State (optional filter)",
        full_width=True,
    )

    mo.vstack(
        [
            mo.md("### Filters"),
            geo_dd,
            metric_ms,
            year_slider,
            state_input,
        ]
    )
    return geo_dd, metric_ms, state_input, year_slider


@app.cell
def export_section(mo, result):
    """Download the *filtered* dataset. Each button serialises lazily, only when clicked."""
    base = "libbee_empty" if result.is_empty() else f"libbee_{int(result['year'].min())}_{int(result['year'].max())}"

    def _parquet_bytes():
        import io

        buf = io.BytesIO()
        result.write_parquet(buf)
        return buf.getvalue()

    downloads = mo.hstack(
        [
            mo.download(data=lambda: result.write_csv(), filename=f"{base}.csv", mimetype="text/csv", label="📥 CSV"),
            mo.download(data=lambda: result.write_json(), filename=f"{base}.json", mimetype="application/json", label="📥 JSON"),
            mo.download(data=_parquet_bytes, filename=f"{base}.parquet", mimetype="application/octet-stream", label="📥 Parquet"),
        ],
        justify="start",
        gap=2,
    )

    mo.vstack([mo.md("### Export filtered data"), downloads])
    return


@app.cell
def _():
    return


@app.cell(column=3)
def summary_stats(mo, result):
    """Summarise the filtered data (always defines ``summary`` so the chart cell can place it)."""
    if result.is_empty():
        summary = mo.callout(
            mo.md("**No data matches your filters.** Try widening the year range, clearing the state, or selecting at least one metric."),
            kind="warn",
        )
    else:
        summary = mo.vstack(
            [
                mo.md("### Filtered data"),
                mo.md(f"Showing **{result.height:,}** points across **{result['geo_id'].n_unique():,}** geographies and **{result['metric'].n_unique()}** metric(s)."),
                mo.hstack(
                    [
                        mo.stat(value=f"{result.height:,}", label="Rows"),
                        mo.stat(value=f"{result['geo_id'].n_unique():,}", label="Geographies"),
                        mo.stat(value=f"{result['value'].mean():.2f}", label="Mean value"),
                    ],
                    justify="start",
                    gap=1,
                ),
                mo.hstack(
                    [
                        mo.stat(value=f"{result['value'].min():.2f}", label="Min", bordered=True),
                        mo.stat(value=f"{result['value'].max():.2f}", label="Max", bordered=True),
                    ],
                    justify="start",
                    gap=1,
                ),
            ]
        )
    summary
    return (summary,)


@app.cell
def table_info(facts, mo):
    """Display basic shape + the available dimensions to filter on."""
    n_rows, n_cols = facts.shape
    geo_levels = sorted(facts["geo_level"].unique().to_list())
    metrics = sorted(facts["metric"].unique().to_list())
    year_range = (int(facts["year"].min()), int(facts["year"].max()))

    mo.vstack(
        [
            mo.md(f"**Rows:** {n_rows:,} | **Columns:** {n_cols}"),
            mo.md(f"**Years:** {year_range[0]} → {year_range[1]}"),
            mo.md(f"**Geo levels:** {', '.join(geo_levels)}"),
            mo.md(f"**Metrics ({len(metrics)}):** {', '.join(metrics)}"),
        ]
    )
    return geo_levels, metrics, year_range


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
