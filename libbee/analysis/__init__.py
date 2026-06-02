"""The analysis layer — the **cuts** the notebook tells stories with.

Every function is **pure**: it takes already-loaded Polars frames (from `libbee.frames.load()`) plus scalar
parameters and returns a Polars frame (and, where the caller needs them, a small dict/tuple of scalars). No
disk IO, no `frames.load()` inside — so the notebook cells stay a one-liner call and the functions are
trivially unit-testable against tiny synthetic frames.

This is the deliberate separation of concerns: `libbee` does the data engineering and the repeatable
aggregations; the notebook does the storytelling and the live modelling. The per-capita convention is
consistent throughout — a rate is $\\text{sum(count)} / \\text{sum(pop)}$, summed *before* dividing, so a
small county never counts the same as a large one.

The cuts are grouped by theme into submodules; this package re-exports them, so the public surface is a
flat `libbee.analysis.<function>` regardless of which file a cut lives in:

    series       scope/index a metric, region trends, the explorer feed
    leaderboards state rankings, growth, funding deciles/tiers, the dose-response
    homeless     the §6 scatter + hours×spend cohort heatmap
    events       the §7 event study and difference-in-differences
    ml           the model matrix, personas, and the budget allocators
    quality      the zero/missing scan and the coverage denominator

Example:
    ```python
    county = libbee.load("county_equity")
    board = libbee.analysis.state_leaderboard(county)
    ```
"""

from __future__ import annotations

from .events import (
    cohort_cost_curve,
    did_means,
    did_panel,
    did_trends,
    event_cohorts,
    event_panel,
    event_study,
)
from .homeless import homeless_cohorts, homeless_scatter
from .leaderboards import (
    cohort_compare,
    funding_deciles,
    funding_response_longitudinal,
    funding_tiers,
    quantile_bins,
    state_growth,
    state_leaderboard,
)
from .ml import (
    MODEL_FEATURES,
    PERSONA_GLOSSARY,
    allocation_frame,
    cluster_labels,
    fitting_score,
    greedy_allocation,
    model_matrix,
    optimal_allocation,
    policy_allocate,
    shap_values,
    whatif_impact,
)
from .quality import coverage_report, qa_report
from .series import (
    acf_pacf,
    aic_table,
    explorer_series,
    index_to_100,
    national_series,
    region_trends,
    scope_series,
)

__all__ = [
    # series
    "scope_series",
    "index_to_100",
    "region_trends",
    "explorer_series",
    "national_series",
    "acf_pacf",
    "aic_table",
    # leaderboards
    "state_leaderboard",
    "state_growth",
    "quantile_bins",
    "funding_deciles",
    "funding_tiers",
    "funding_response_longitudinal",
    "cohort_compare",
    # homeless
    "homeless_scatter",
    "homeless_cohorts",
    # events / causal
    "event_cohorts",
    "event_study",
    "event_panel",
    "cohort_cost_curve",
    "did_panel",
    "did_means",
    "did_trends",
    # ml + allocators
    "MODEL_FEATURES",
    "PERSONA_GLOSSARY",
    "model_matrix",
    "cluster_labels",
    "allocation_frame",
    "greedy_allocation",
    "optimal_allocation",
    "policy_allocate",
    "fitting_score",
    "shap_values",
    "whatif_impact",
    # data quality
    "qa_report",
    "coverage_report",
]
