"""Metadata-level data contracts (Pydantic) for libbee.

Two cheap, **row-free** safety rails:

1. :func:`validate_polars_schema` — compare a Polars frame's *schema* (column names + dtypes) against a
   Pydantic contract in microseconds. It never materialises a row, so it keeps Polars' Arrow speed while
   catching upstream drift (a column that vanished or changed type).
2. :class:`ManifestSchema` / :func:`validate_manifest` — validate the structure of ``data/MANIFEST.json``
   (the reproducibility fingerprint) before trusting it.

Row-by-row validation is deliberately *not* offered — converting frames to dicts to run per-row validators
would defeat the point of a columnar engine.
"""

from __future__ import annotations

import types
import typing
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel


# ── 1 · Polars schema contracts ──────────────────────────────────────────────
class FactsSchema(BaseModel):
    """The unified long table's contract — ``libbee.facts()`` columns and their types."""

    geo_level: Literal["state", "metro", "county", "library"]
    geo_id: str
    geo_name: str
    state: str | None  # null for national rollups
    year: int
    metric: str
    value: float
    source: str


# Python annotation → the Polars dtype *families* that satisfy it.
_TYPE_FAMILIES: dict[type, tuple[type, ...]] = {
    str: (pl.String, pl.Categorical, pl.Enum),
    int: (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64),
    float: (pl.Float32, pl.Float64),
    bool: (pl.Boolean,),
}


def _base_type(annotation: object) -> type:
    """Reduce an annotation to the base Python type used for dtype mapping: unwrap ``Optional[X]`` /
    ``X | None`` and collapse ``Literal[...]`` (string-valued here) to ``str``."""
    origin = typing.get_origin(annotation)
    if origin is Literal:
        return str
    if origin in (typing.Union, types.UnionType):  # Optional[X] or X | None
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        return non_none[0] if non_none else str
    return annotation if isinstance(annotation, type) else str


def validate_polars_schema(df: pl.DataFrame | pl.LazyFrame, contract: type[BaseModel]) -> None:
    """Assert ``df`` has every column in ``contract`` with a compatible dtype. Metadata-only — O(columns),
    not O(rows). Raises ``TypeError`` on a missing column or a dtype mismatch."""
    schema = df.schema if isinstance(df, pl.DataFrame) else df.collect_schema()
    for name, field in contract.model_fields.items():
        if name not in schema:
            raise TypeError(f"{contract.__name__}: missing required column {name!r}")
        base = _base_type(field.annotation)
        allowed = _TYPE_FAMILIES.get(base)
        if allowed is not None and not isinstance(schema[name], allowed):
            raise TypeError(f"{contract.__name__}: column {name!r} is {schema[name]}, expected {base.__name__}")


# ── 2 · MANIFEST.json contract ───────────────────────────────────────────────
class RawInputFingerprint(BaseModel):
    """A cached raw input's fingerprint: ``{bytes, sha256}``."""

    bytes: int
    sha256: str


class TableFingerprint(BaseModel):
    """A built frame's fingerprint: ``{cols, rows, sha256}``."""

    cols: int
    rows: int
    sha256: str


class ManifestSchema(BaseModel):
    """The structure of ``data/MANIFEST.json`` — every raw input and every built table, fingerprinted."""

    raw_inputs: dict[str, RawInputFingerprint]
    tables: dict[str, TableFingerprint]


def validate_manifest(data: dict | str | Path) -> ManifestSchema:
    """Validate a manifest given as a dict, a JSON string, or a path to ``MANIFEST.json``. Raises
    ``pydantic.ValidationError`` if the structure is corrupt."""
    if isinstance(data, dict):
        return ManifestSchema.model_validate(data)
    path = Path(data)
    if path.exists():
        return ManifestSchema.model_validate_json(path.read_text())
    return ManifestSchema.model_validate_json(str(data))  # a raw JSON string
