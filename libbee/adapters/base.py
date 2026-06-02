"""The generic adapter contract.

A dataset is an Adapter subclass that owns two things:
  • build()    — fetch (cached) → conform → write its source table(s)
  • to_facts() — (optional) its contribution to the unified long `facts` table

`unify` just concatenates everyone's to_facts(), so it knows nothing about IMLS/HUD/CA — adding a
new dataset is a new module, no edits to the core. `melt()` is the shared long-format helper.
"""

from __future__ import annotations

import polars as pl

from ..io.store import db_has
from ..metrics import CONFORMED


def order_adapters(adapters: list[Adapter], *, attr: str) -> list[Adapter]:
    """Return adapters in stable topological order for the named dependency attribute."""
    by_name = {adapter.name: adapter for adapter in adapters}
    pending = list(adapters)
    ordered: list[Adapter] = []
    resolved: set[str] = set()
    while pending:
        progress = False
        for adapter in list(pending):
            deps = tuple(getattr(adapter, attr, ()) or ())
            unknown = [dep for dep in deps if dep not in by_name]
            if unknown:
                joined = ", ".join(unknown)
                raise ValueError(f"{adapter.name}: unknown {attr} dependency {joined}")
            if all(dep in resolved for dep in deps):
                ordered.append(adapter)
                resolved.add(adapter.name)
                pending.remove(adapter)
                progress = True
        if not progress:
            cycle = ", ".join(adapter.name for adapter in pending)
            raise ValueError(f"cyclic {attr} dependencies: {cycle}")
    return ordered


def melt(
    df: pl.DataFrame,
    geo_level: str,
    geo_id: str,
    geo_name: str,
    source: str,
    *,
    year,
    state: str | None = None,
    metrics: list[str] | None = None,
) -> pl.DataFrame:
    """Unpivot a frame's conformed metric columns into facts rows. `year` is a column name
    (time series) or a fixed int (cross-section); `state` is a column or None."""
    metrics = metrics or [c for c in CONFORMED if c in df.columns]
    keep = list(dict.fromkeys([geo_id, geo_name] + ([year] if isinstance(year, str) else []) + ([state] if state else [])))
    return (
        df.select(keep + metrics)
        .unpivot(index=keep, on=metrics, variable_name="metric", value_name="value")
        .drop_nulls("value")
        .with_columns(
            pl.lit(geo_level).alias("geo_level"),
            pl.col(geo_id).cast(pl.Utf8).alias("geo_id"),
            pl.col(geo_name).cast(pl.Utf8).alias("geo_name"),
            (pl.col(year) if isinstance(year, str) else pl.lit(year)).cast(pl.Int64).alias("year"),
            pl.lit(source).alias("source"),
            (pl.col(state) if state else pl.lit(None)).alias("state"),
        )
        .select("geo_level", "geo_id", "geo_name", "state", "year", "metric", "value", "source")
    )


class Adapter:
    name: str = ""
    source: str = ""
    tables: tuple[str, ...] = ()
    provenance: str = ""
    build_after: tuple[str, ...] = ()
    facts_after: tuple[str, ...] = ()

    def build(self) -> None:
        """Fetch (cached) + conform + write this adapter's source table(s) to SQLite."""
        raise NotImplementedError

    def to_facts(self) -> pl.DataFrame | None:
        """Optional: rows for the unified `facts` table
        (geo_level, geo_id, geo_name, state, year, metric, value, source)."""
        return None

    def present(self) -> bool:
        return all(db_has(t) for t in self.tables)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} · {self.source} → {', '.join(self.tables)}>"
