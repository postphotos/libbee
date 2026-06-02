"""Unit tests for libbee.io.schemas — metadata-level data contracts (no row validation)."""

from __future__ import annotations

import typing

import polars as pl
import pytest
from pydantic import BaseModel, ValidationError

from libbee.io import schemas
from libbee.io.schemas import (
    FactsSchema,
    ManifestSchema,
    validate_manifest,
    validate_polars_schema,
)


def _good_facts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "geo_level": ["state"],
            "geo_id": ["08"],
            "geo_name": ["Colorado"],
            "state": ["CO"],
            "year": [2019],
            "metric": ["visits_pc"],
            "value": [3.1],
            "source": ["IMLS"],
        }
    )


# ── validate_polars_schema ───────────────────────────────────────────────────
def test_facts_schema_passes_on_good_frame():
    validate_polars_schema(_good_facts(), FactsSchema)  # no raise


def test_lazyframe_is_accepted():
    validate_polars_schema(_good_facts().lazy(), FactsSchema)  # collect_schema branch


def test_missing_column_raises():
    with pytest.raises(TypeError, match="missing required column 'value'"):
        validate_polars_schema(_good_facts().drop("value"), FactsSchema)


def test_dtype_mismatch_raises():
    bad = _good_facts().with_columns(pl.col("year").cast(pl.String))  # year should be int
    with pytest.raises(TypeError, match="column 'year' is String, expected int"):
        validate_polars_schema(bad, FactsSchema)


def test_optional_and_literal_columns():
    """state is Optional[str] and geo_level is a Literal — both reduce to str and pass."""
    df = _good_facts().with_columns(pl.lit(None, dtype=pl.String).alias("state"))
    validate_polars_schema(df, FactsSchema)  # null state is fine


def test_unmapped_field_type_is_skipped():
    """A field whose base type isn't a known dtype family (bytes) is presence-checked, not type-checked."""

    class _Blob(BaseModel):
        payload: bytes

    df = pl.DataFrame({"payload": [b"x"]}, schema={"payload": pl.Binary})
    validate_polars_schema(df, _Blob)  # no raise — bytes has no dtype family, only presence is enforced


def test_generic_annotation_reduces_to_str():
    """A non-primitive annotation (Any) reduces to str (the isinstance-False branch)."""

    class _AnyField(BaseModel):
        tag: typing.Any

    validate_polars_schema(pl.DataFrame({"tag": ["hi"]}), _AnyField)  # tag:String satisfies str


# ── validate_manifest / ManifestSchema ───────────────────────────────────────
_VALID_MANIFEST = {
    "raw_inputs": {"imls/x.zip": {"bytes": 10, "sha256": "abc"}},
    "tables": {"facts": {"cols": 8, "rows": 100, "sha256": "def"}},
}


def test_validate_manifest_from_dict():
    m = validate_manifest(_VALID_MANIFEST)
    assert isinstance(m, ManifestSchema)
    assert m.tables["facts"].rows == 100
    assert m.raw_inputs["imls/x.zip"].bytes == 10


def test_validate_manifest_from_json_string():
    import json

    m = validate_manifest(json.dumps(_VALID_MANIFEST))
    assert m.tables["facts"].cols == 8


def test_validate_manifest_from_file(tmp_path):
    import json

    p = tmp_path / "MANIFEST.json"
    p.write_text(json.dumps(_VALID_MANIFEST))
    m = validate_manifest(p)
    assert m.raw_inputs["imls/x.zip"].sha256 == "abc"


def test_validate_manifest_rejects_corrupt():
    with pytest.raises(ValidationError):
        validate_manifest({"raw_inputs": {}, "tables": {"facts": {"rows": "not-an-int"}}})


def test_real_manifest_matches_contract():
    """The committed data/MANIFEST.json validates against the contract (when present)."""
    from libbee.io.paths import MANIFEST

    if MANIFEST.exists():
        assert isinstance(validate_manifest(MANIFEST), schemas.ManifestSchema)
