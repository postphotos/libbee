"""Read libbee's finished frames, write source tables, and flatten to Parquet + DuckDB.

The **libbee data product** publishes a set of conformed, demo-ready frames as Parquet files under
`data/flat/`. This module is the public access layer: name a frame and get it back as a
[polars](https://pola.rs) `DataFrame` (eager) or `LazyFrame` (deferred), list what's available, or
export frames to portable CSV / JSON.

Frames are addressed by a short **name** — the Parquet file stem (for example `'facts'` or
`'county_equity'`), without the `.parquet` extension.

Behind the published frames sit a SQLite source DB and a columnar DuckDB mirror; `write_table` and
`flatten` build those, while `load` / `scan` / `tables` / `export_*` read the published surface.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal

import polars as pl

from .paths import DB, DUCKDB, FLAT

ExportFormat = Literal["csv", "json"]


def load(name: str) -> pl.DataFrame:
    """Read one finished frame **eagerly** into memory by name.

    Loads the Parquet file `<name>.parquet` from the published `flat/` directory and returns it as a
    fully materialized polars `DataFrame`. Use this when you want the rows in memory right now; use
    `scan` when you'd rather build a lazy query and only read what you need.

    Args:
        name: The frame name — the Parquet file stem, e.g. `'facts'` or `'county_equity'`, with no
            `.parquet` extension. Must name an existing published frame (see `tables`).

    Returns:
        A polars `DataFrame` holding every row and column of the named frame.

    Raises:
        FileNotFoundError: when no `<name>.parquet` frame exists in the `flat/` directory.

    Example:
        ```python
        df = load("facts")
        ```
    """
    path = FLAT / f"{name}.parquet"
    try:
        return pl.read_parquet(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Conformed table '{name}' not found at {path}.\n"
            f"The local cache might be empty or unbuilt.\n"
            f"Please run 'libbee.build()' in Python, or run the CLI command 'libbee build' in your shell to build it."
        ) from e


def scan(name: str) -> pl.LazyFrame:
    """Open one finished frame **lazily** by name.

    Returns a polars `LazyFrame` over the Parquet file `<name>.parquet` in the published `flat/`
    directory. Nothing is read until you call `.collect()`; filters and projections push down into the
    Parquet scan, so a query like `scan(name).select(...).filter(...).collect()` reads only the columns
    and rows it needs. Prefer this over `load` for large frames or selective queries.

    Args:
        name: The frame name — the Parquet file stem, e.g. `'facts'` or `'county_equity'`, with no
            `.parquet` extension. Must name an existing published frame (see `tables`).

    Returns:
        A polars `LazyFrame` bound to the named frame's Parquet file, ready to compose and `.collect()`.

    Example:
        ```python
        lf = scan("county_equity")
        df = lf.collect()
        ```
    """
    path = FLAT / f"{name}.parquet"
    try:
        return pl.scan_parquet(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Conformed table '{name}' not found at {path}.\n"
            f"The local cache might be empty or unbuilt.\n"
            f"Please run 'libbee.build()' in Python, or run the CLI command 'libbee build' in your shell to build it."
        ) from e


def tables() -> list[str]:
    """List the names of every published frame.

    Globs the `flat/` directory for `*.parquet` files and returns their **stems** — the names you pass
    to `load`, `scan`, and `export_table`. The list is sorted alphabetically. If no frames have been
    published yet, this returns an empty list.

    Returns:
        A `list[str]` of frame names (Parquet stems, no extension), sorted alphabetically.

    Example:
        ```python
        names = tables()  # e.g. ["county_equity", "facts", ...]
        ```
    """
    return sorted(p.stem for p in FLAT.glob("*.parquet"))


def _export_path(name: str, *, file_format: ExportFormat, destination: str | Path | None) -> Path:
    if file_format not in {"csv", "json"}:
        raise ValueError(f"Unsupported export format: {file_format}")
    base = Path(destination) if destination is not None else FLAT.parent / "exports"
    path = base if base.suffix else base / f"{name}.{file_format}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def export_table(
    name: str,
    *,
    file_format: ExportFormat = "csv",
    destination: str | Path | None = None,
) -> Path:
    """Export one published frame to a portable **CSV** or **row-oriented JSON** file.

    Loads the named frame, then writes it to disk. For `"csv"` the frame is written with polars'
    `write_csv`; for `"json"` it is serialized as a JSON **array of row objects** (one dict per row),
    pretty-printed with a 2-space indent and a trailing newline, with non-JSON-native values rendered
    via `str`. The output location is resolved from `destination`:

    - `None` → `data/exports/<name>.<file_format>`.
    - a path **with a suffix** (e.g. `out.csv`) → used verbatim as the output file.
    - a path **without a suffix** (a directory) → `<destination>/<name>.<file_format>`.

    Parent directories are created as needed.

    Args:
        name: The frame name to export — a Parquet stem (see `tables`).
        file_format: Either `"csv"` or `"json"`. Defaults to `"csv"`.
        destination: Output file or directory; see the resolution rules above. Defaults to `None`.

    Returns:
        The `pathlib.Path` of the file that was written.

    Raises:
        ValueError: when `file_format` is not `"csv"` or `"json"`.
        FileNotFoundError: when the named frame does not exist in `flat/`.

    Example:
        ```python
        path = export_table("facts", file_format="json")
        ```
    """
    path = _export_path(name, file_format=file_format, destination=destination)
    df = load(name)
    if file_format == "csv":
        df.write_csv(path)
    else:
        path.write_text(json.dumps(df.to_dicts(), indent=2, default=str) + "\n")
    return path


def export_all(
    names: list[str] | None = None,
    *,
    file_format: ExportFormat = "csv",
    destination: str | Path | None = None,
) -> list[Path]:
    """Export many published frames to CSV or JSON in one call.

    Calls `export_table` once per frame and collects the resulting paths. When `names` is omitted (or
    empty), it exports **every** published frame returned by `tables`. The `file_format` and
    `destination` arguments are forwarded unchanged to each `export_table` call, so when `destination`
    is a directory the files land side by side as `<name>.<file_format>`.

    Args:
        names: The frames to export, as a `list[str]` of Parquet stems. If `None` or empty, all frames
            from `tables` are exported. Defaults to `None`.
        file_format: Either `"csv"` or `"json"`, applied to every frame. Defaults to `"csv"`.
        destination: Output file or directory, forwarded to each `export_table` call. Defaults to `None`.

    Returns:
        A `list[pathlib.Path]`, one written file per exported frame, in the order processed.

    Raises:
        ValueError: when `file_format` is not `"csv"` or `"json"`.
        FileNotFoundError: when a requested frame does not exist in `flat/`.

    Example:
        ```python
        paths = export_all(["facts", "county_equity"], file_format="csv")
        ```
    """
    target_names = names or tables()
    return [export_table(name, file_format=file_format, destination=destination) for name in target_names]


def db_uri() -> str:
    DB.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DB}"


def write_table(df: pl.DataFrame, name: str) -> None:
    """Write a polars frame into the SQLite **source** database, replacing any prior table.

    Source adapters call this to land a conformed frame as a SQLite table via polars' `write_database`
    (ADBC engine). Any existing table of the same name is **replaced**. This populates the source DB
    that `flatten` later reads from to publish the `flat/` Parquet frames; it does not write to `flat/`
    directly.

    Args:
        df: The polars `DataFrame` to persist.
        name: The destination table name in the SQLite source DB.

    Returns:
        `None`. The table is written as a side effect.

    Example:
        ```python
        write_table(df, "facts")
        ```
    """
    df.write_database(name, db_uri(), engine="adbc", if_table_exists="replace")


def db_has(name: str) -> bool:
    try:
        c = sqlite3.connect(DB)
        c.execute(f"SELECT 1 FROM {name} LIMIT 1")
        c.close()
        return True
    except Exception:
        return False


def flatten(names: list[str]) -> None:
    r"""Publish SQLite source tables to Parquet, then mirror them into DuckDB.

    For each table name, this reads the full table from the SQLite source DB (`SELECT * FROM <name>`,
    inferring the schema across all rows), writes it to `flat/<name>.parquet` — the **demo-ready
    snapshot** that `load` / `scan` / `tables` serve — and then creates or replaces a matching DuckDB
    table that reads from that Parquet file. So the pipeline is:

    $$\text{SQLite} \;\to\; \text{Parquet (flat/)} \;\to\; \text{DuckDB}$$

    DuckDB is imported lazily inside the function. The `flat/` directory is created if missing.

    Args:
        names: The source-table names to publish, as a `list[str]`. Each must exist in the SQLite
            source DB. One Parquet file and one DuckDB table are produced per name.

    Returns:
        `None`. Parquet files and DuckDB tables are written as a side effect.

    Example:
        ```python
        flatten(["facts", "county_equity"])
        ```
    """
    FLAT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    for n in names:
        _df = pl.read_database(f"SELECT * FROM {n}", conn, infer_schema_length=None)
        _df.write_parquet(FLAT / f"{n}.parquet")
    conn.close()
    import duckdb

    con = duckdb.connect(str(DUCKDB))
    for n in names:
        con.execute(f"CREATE OR REPLACE TABLE {n} AS SELECT * FROM read_parquet('{FLAT / f'{n}.parquet'}')")
    con.close()
