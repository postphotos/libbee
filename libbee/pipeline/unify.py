"""Unify — concatenate every adapter's facts contribution into one long table (+ geo_dim).

Fully generic: it asks each adapter for `to_facts()` and stacks them. It contains no per-source
logic, so a new dataset's rows appear here for free once its adapter declares `to_facts()`.
"""

from __future__ import annotations

import polars as pl

from ..adapters import REGISTRY, order_adapters
from ..io.schemas import FactsSchema, validate_polars_schema
from ..io.store import write_table


def build() -> None:
    parts = [a.to_facts() for a in order_adapters(REGISTRY, attr="facts_after")]
    parts = [p for p in parts if p is not None and p.height > 0]
    facts = pl.concat(parts, how="vertical_relaxed").unique(subset=["geo_level", "geo_id", "year", "metric", "source"]).sort("geo_level", "geo_id", "year", "metric")
    validate_polars_schema(facts, FactsSchema)  # metadata-only contract rail — catches upstream drift
    geo_dim = facts.select("geo_level", "geo_id", "geo_name", "state").unique().sort("geo_level", "geo_id")
    write_table(facts, "facts")
    write_table(geo_dim, "geo_dim")
    print(f"  → facts: {facts.height:,} rows · {facts['geo_level'].n_unique()} levels · {facts['source'].n_unique()} sources · geo_dim {geo_dim.height:,}")
