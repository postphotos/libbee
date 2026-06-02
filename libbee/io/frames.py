"""The boring layer, as one import — so the notebook never re-implements data loading.

`load()` returns every demo-ready frame in a dict, building anything missing first (self-contained,
cached → no drift). This is what `data_prep.py` used to do inline; the notebook now just calls
`libbee.frames.load()`.
"""

from __future__ import annotations

import glob

import polars as pl

from . import store
from .paths import DUCKDB, FLAT

# Every flat frame the notebooks read. Keep in sync with the adapters' outputs + unify.
TABLES = [
    "ca_annual",
    "ca_county_recent",
    "ms_broadband",
    "noaa_daily",
    "imls_admin",
    "metro_panel",
    "metro_libraries",
    "metro_lookup",
    "county_equity",
    "county_panel",
    "coc_homeless",
    "metro_homeless",
    "coc_spend",
    "metro_spend",
    "county_homeless",
    "state_panel",
    "ca_libraries",
    "facts",
    "geo_dim",
]


def _selected_tables(names: list[str] | None = None) -> list[str]:
    return list(names) if names is not None else list(TABLES)


def prepare(names: list[str] | None = None) -> None:
    """Ensure every flat Parquet exists; build only what's missing (instant once on disk)."""
    target_names = _selected_tables(names)
    FLAT.mkdir(parents=True, exist_ok=True)
    have = {p.rsplit("/", 1)[-1][:-8] for p in glob.glob(f"{FLAT}/*.parquet")}
    if [t for t in target_names if t not in have]:
        import libbee

        libbee.build()  # fetch (cached) → conform → flatten → unify facts → MANIFEST


def to_duckdb() -> str:
    """Materialise the Parquet into a DuckDB database (columnar SQL), once. Returns its path."""
    import duckdb

    if not DUCKDB.exists():
        con = duckdb.connect(str(DUCKDB))
        for f in sorted(glob.glob(f"{FLAT}/*.parquet")):
            name = f.rsplit("/", 1)[-1][:-8]
            if not name.startswith("_"):
                con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{f}')")
        con.close()
    return str(DUCKDB)


def load(names: list[str] | None = None) -> dict[str, pl.DataFrame]:
    """Build-if-missing, then return {table_name: DataFrame} for each requested flat frame."""
    target_names = _selected_tables(names)
    prepare(target_names)
    return {t: store.load(t) for t in target_names}


def scan(names: list[str] | None = None) -> dict[str, pl.LazyFrame]:
    """Build-if-missing, then return {table_name: LazyFrame} for each requested flat frame."""
    target_names = _selected_tables(names)
    prepare(target_names)
    return {t: store.scan(t) for t in target_names}
