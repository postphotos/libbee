"""Human-facing introspection — *is the data built* (`status`) and *what's in it* (`list`).

These answer the two questions you ask **before** the first query: "is the cache ready?" and
"what can I load, and what does it contain?" Both read the published `flat/` surface and the
committed `MANIFEST.json`; neither fetches nor builds. The canonical guard is:

    >>> import libbee
    >>> if not libbee.status():
    ...     libbee.build()
    >>> libbee.list()                       # ['county_homeless', 'county_panel', 'facts', ...]
    >>> libbee.list("county_panel").cols()  # ['fips', 'year', 'visits', 'funding', ...]
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import polars as pl

from .paths import FLAT, MANIFEST
from .store import load, scan, tables


@dataclass(frozen=True)
class BuildStatus:
    """Whether the local data product is built and ready to `load`.

    Truthy when at least one frame is published and nothing the MANIFEST expects is missing — so
    `if not libbee.status(): libbee.build()` does the right thing on a fresh checkout.

    Attributes:
        built: The overall verdict (also this object's truth value).
        present: Frame names currently published under `flat/`.
        expected: Frame names the committed MANIFEST lists (empty if no MANIFEST yet).
        missing: `expected` frames that are not `present`.
        has_manifest: Whether a `MANIFEST.json` exists at all.
    """

    built: bool
    present: list[str]
    expected: list[str]
    missing: list[str]
    has_manifest: bool

    def __bool__(self) -> bool:
        return self.built

    def summary(self) -> str:
        if self.built:
            return f"✅ libbee data is built — {len(self.present)} tables ready"
        if not self.present:
            return "✗ libbee data is not built — run libbee.build()"
        return f"⚠ libbee data is partial — missing {len(self.missing)}: {', '.join(self.missing)}"

    def __repr__(self) -> str:
        return f"BuildStatus(built={self.built}, present={len(self.present)}, missing={self.missing})"

    def _repr_markdown_(self) -> str:  # renders nicely inside marimo / Jupyter
        bullets = "\n".join(f"- `{t}`" for t in self.present) or "_(none yet)_"
        return f"**{self.summary()}**\n\n{bullets}"


def status() -> BuildStatus:
    """Report whether the conformed data is built and ready to load — without building it.

    Globs the published `flat/` directory and, if a `MANIFEST.json` is present, checks that every
    table it lists is published. Returns a `BuildStatus` whose truth value is `True` when the cache
    is ready, so the canonical guard is:

        >>> if not libbee.status():
        ...     libbee.build()

    Returns:
        A `BuildStatus`. Truthy when at least one frame exists and none the MANIFEST expects is missing.
    """
    present = tables()
    expected: list[str] = []
    has_manifest = MANIFEST.exists()
    if has_manifest:
        try:
            expected = sorted(json.loads(MANIFEST.read_text()).get("tables", {}))
        except (json.JSONDecodeError, OSError):
            expected = []
    missing = [t for t in expected if t not in present]
    built = bool(present) and not missing
    return BuildStatus(built=built, present=present, expected=expected, missing=missing, has_manifest=has_manifest)


@dataclass(frozen=True)
class TableHandle:
    """A lightweight handle to one published frame — inspect it without loading every row.

    Returned by `list(name)`. Reads the Parquet schema and metadata lazily, so `.cols()`, `.schema()`
    and `.shape` stay cheap even on the largest frames; `.load()` / `.scan()` hand you the data.
    """

    name: str

    def exists(self) -> bool:
        return (FLAT / f"{self.name}.parquet").exists()

    def cols(self) -> list[str]:
        """The column names, read from the Parquet schema (no full read)."""
        return scan(self.name).collect_schema().names()

    def schema(self) -> dict[str, str]:
        """`{column: dtype}` as strings, read from the Parquet schema (no full read)."""
        return {name: str(dtype) for name, dtype in scan(self.name).collect_schema().items()}

    @property
    def shape(self) -> tuple[int, int]:
        """`(rows, cols)`, read from Parquet metadata (no full read)."""
        rows = scan(self.name).select(pl.len()).collect().item()
        return (int(rows), len(self.cols()))

    def head(self, n: int = 5) -> pl.DataFrame:
        """The first `n` rows."""
        return scan(self.name).head(n).collect()

    def load(self) -> pl.DataFrame:
        """The whole frame, eagerly (delegates to `libbee.load`)."""
        return load(self.name)

    def scan(self) -> pl.LazyFrame:
        """A lazy frame over this table (delegates to `libbee.scan`)."""
        return scan(self.name)

    def __repr__(self) -> str:
        if not self.exists():
            return f"TableHandle('{self.name}', missing)"
        rows, cols = self.shape
        return f"TableHandle('{self.name}', {rows:,} rows × {cols} cols)"

    def _repr_markdown_(self) -> str:
        if not self.exists():
            return f"**`{self.name}`** — not found. Run `libbee.build()`, or see `libbee.list()`."
        rows, cols = self.shape
        cols_md = ", ".join(f"`{name}`" for name in self.cols())
        return f"**`{self.name}`** — {rows:,} rows × {cols} cols\n\n{cols_md}"


def list(name: str | None = None):  # noqa: A001 — deliberately exposed as libbee.list
    """List the published tables, or inspect one by name.

    - `libbee.list()` → `list[str]` of every published frame name (the same set as `tables()`).
    - `libbee.list("county_panel")` → a `TableHandle`; call `.cols()`, `.schema()`, `.head()`,
      `.shape`, `.load()`, or `.scan()` on it.

    Args:
        name: A frame name to inspect, or `None` to list every published frame.

    Returns:
        `list[str]` when `name is None`, otherwise a `TableHandle` for that frame.
    """
    if name is None:
        return tables()
    return TableHandle(name)
