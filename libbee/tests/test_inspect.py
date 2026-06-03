"""Unit tests for libbee.io.inspect — status() and list()/TableHandle introspection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import polars as pl
import pytest

from libbee.io import inspect


@pytest.fixture
def flat(tmp_path: Path):
    """A temp flat/ dir wired into both inspect and store, with a MANIFEST path."""
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    manifest = tmp_path / "MANIFEST.json"
    with (
        mock.patch("libbee.io.inspect.FLAT", flat_dir),
        mock.patch("libbee.io.inspect.MANIFEST", manifest),
        mock.patch("libbee.io.store.FLAT", flat_dir),
    ):
        yield flat_dir, manifest


def _write(flat_dir: Path, name: str, df: pl.DataFrame) -> None:
    df.write_parquet(flat_dir / f"{name}.parquet")


class TestStatus:
    def test_not_built_when_empty(self, flat):
        """No parquet files and no MANIFEST → not built, falsy."""
        s = inspect.status()
        assert s.built is False
        assert not s
        assert s.present == []
        assert "not built" in s.summary()

    def test_built_when_manifest_satisfied(self, flat):
        """Every table the MANIFEST expects is present → built, truthy."""
        flat_dir, manifest = flat
        _write(flat_dir, "county_panel", pl.DataFrame({"fips": ["00001"], "year": [2019]}))
        manifest.write_text(json.dumps({"tables": {"county_panel": {"sha256": "x"}}}))
        s = inspect.status()
        assert s.built is True
        assert bool(s) is True
        assert s.present == ["county_panel"]
        assert s.missing == []

    def test_partial_when_table_missing(self, flat):
        """MANIFEST expects a table that isn't published → partial, falsy."""
        flat_dir, manifest = flat
        _write(flat_dir, "county_panel", pl.DataFrame({"fips": ["00001"]}))
        manifest.write_text(json.dumps({"tables": {"county_panel": {}, "facts": {}}}))
        s = inspect.status()
        assert s.built is False
        assert s.missing == ["facts"]

    def test_built_with_tables_but_no_manifest(self, flat):
        """Tables present, no MANIFEST → treated as built (data is loadable)."""
        flat_dir, _ = flat
        _write(flat_dir, "facts", pl.DataFrame({"a": [1]}))
        s = inspect.status()
        assert s.built is True
        assert s.has_manifest is False


class TestList:
    def test_list_no_arg_returns_names(self, flat):
        flat_dir, _ = flat
        _write(flat_dir, "facts", pl.DataFrame({"a": [1]}))
        _write(flat_dir, "county_panel", pl.DataFrame({"b": [2]}))
        assert inspect.list() == ["county_panel", "facts"]

    def test_list_name_returns_handle(self, flat):
        handle = inspect.list("county_panel")
        assert isinstance(handle, inspect.TableHandle)
        assert handle.name == "county_panel"

    def test_handle_cols_and_shape(self, flat):
        flat_dir, _ = flat
        _write(flat_dir, "county_panel", pl.DataFrame({"fips": ["1", "2"], "visits": [10, 20]}))
        handle = inspect.list("county_panel")
        assert handle.exists() is True
        assert handle.cols() == ["fips", "visits"]
        assert handle.shape == (2, 2)
        assert set(handle.schema()) == {"fips", "visits"}

    def test_handle_missing(self, flat):
        handle = inspect.list("nope")
        assert handle.exists() is False
        assert "missing" in repr(handle)

    def test_handle_head_load_scan(self, flat):
        flat_dir, _ = flat
        _write(flat_dir, "t", pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        handle = inspect.list("t")
        assert handle.head(2).height == 2
        assert handle.load().height == 3
        assert handle.scan().collect().height == 3

    def test_handle_repr_and_markdown_present(self, flat):
        flat_dir, _ = flat
        _write(flat_dir, "t", pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
        handle = inspect.list("t")
        assert "2 rows × 2 cols" in repr(handle)
        md = handle._repr_markdown_()
        assert "`t`" in md and "`a`" in md and "`b`" in md

    def test_handle_markdown_missing(self, flat):
        md = inspect.list("nope")._repr_markdown_()
        assert "not found" in md


class TestBuildStatusRendering:
    def test_summary_built_and_partial(self, flat):
        flat_dir, manifest = flat
        _write(flat_dir, "county_panel", pl.DataFrame({"a": [1]}))
        manifest.write_text(json.dumps({"tables": {"county_panel": {}}}))
        assert "ready" in inspect.status().summary()  # built branch
        manifest.write_text(json.dumps({"tables": {"county_panel": {}, "facts": {}}}))
        assert "partial" in inspect.status().summary()  # partial branch

    def test_repr_and_markdown(self, flat):
        flat_dir, manifest = flat
        _write(flat_dir, "county_panel", pl.DataFrame({"a": [1]}))
        manifest.write_text(json.dumps({"tables": {"county_panel": {}}}))
        s = inspect.status()
        assert "BuildStatus(built=True" in repr(s)
        assert "`county_panel`" in s._repr_markdown_()
        # empty present → the "(none yet)" markdown branch
        manifest.unlink()
        for p in flat_dir.glob("*.parquet"):
            p.unlink()
        assert "none yet" in inspect.status()._repr_markdown_()

    def test_malformed_manifest_is_tolerated(self, flat):
        flat_dir, manifest = flat
        _write(flat_dir, "county_panel", pl.DataFrame({"a": [1]}))
        manifest.write_text("{not valid json")  # JSONDecodeError → expected=[]
        s = inspect.status()
        assert s.built is True
        assert s.expected == []
