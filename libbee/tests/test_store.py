"""Unit tests for libbee.io.store — I/O operations (SQLite, Parquet, DuckDB)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest import mock

import polars as pl
import pytest

from libbee.io import store


class TestLoad:
    """Test store.load() reading frames."""

    def test_load_reads_parquet(self):
        """load() reads a Parquet file."""
        with mock.patch("libbee.io.store.pl.read_parquet") as mock_read:
            mock_df = pl.DataFrame({"col": [1, 2, 3]})
            mock_read.return_value = mock_df
            with mock.patch("libbee.io.store.FLAT", Path("/data/flat")):
                result = store.load("test_table")
                assert isinstance(result, pl.DataFrame)

    def test_load_constructs_path(self):
        """load() constructs the correct file path."""
        with mock.patch("polars.read_parquet") as mock_read:
            mock_read.return_value = pl.DataFrame()
            with mock.patch("libbee.io.store.FLAT", Path("/data/flat")):
                store.load("mytable")
                mock_read.assert_called_once()
                call_path = mock_read.call_args[0][0]
                assert "mytable.parquet" in str(call_path)


class TestScan:
    """Test store.scan() lazy parquet access."""

    def test_scan_reads_parquet_lazily(self):
        """scan() returns a LazyFrame from scan_parquet."""
        mock_lf = mock.MagicMock(spec=pl.LazyFrame)
        with mock.patch("libbee.io.store.pl.scan_parquet", return_value=mock_lf) as mock_scan:
            with mock.patch("libbee.io.store.FLAT", Path("/data/flat")):
                result = store.scan("facts")
                assert result is mock_lf
                mock_scan.assert_called_once()

    def test_scan_constructs_path(self):
        """scan() constructs the correct parquet path."""
        with mock.patch("libbee.io.store.pl.scan_parquet") as mock_scan:
            with mock.patch("libbee.io.store.FLAT", Path("/data/flat")):
                store.scan("geo_dim")
                call_path = mock_scan.call_args[0][0]
                assert "geo_dim.parquet" in str(call_path)


class TestTables:
    """Test store.tables() listing."""

    def test_tables_lists_parquet_files(self):
        """tables() returns sorted list of parquet file names."""
        with mock.patch("libbee.io.store.FLAT") as mock_flat:
            mock_flat.glob.return_value = [
                Path("/data/flat/state_panel.parquet"),
                Path("/data/flat/county_equity.parquet"),
                Path("/data/flat/facts.parquet"),
            ]
            result = store.tables()
            assert result == ["county_equity", "facts", "state_panel"]

    def test_tables_empty(self):
        """tables() returns empty list when no Parquet files."""
        with mock.patch("libbee.io.store.FLAT") as mock_flat:
            mock_flat.glob.return_value = []
            result = store.tables()
            assert result == []


class TestDbUri:
    """Test store.db_uri() connection string."""

    def test_db_uri_returns_sqlite_string(self):
        """db_uri() returns a SQLite connection string."""
        mock_db = mock.MagicMock(spec=Path)
        mock_db.__truediv__ = lambda self, x: f"sqlite:///{x}"
        mock_db.parent = mock.MagicMock()
        mock_db.parent.mkdir = mock.MagicMock()
        with mock.patch("libbee.io.store.DB", mock_db):
            store.db_uri()
            mock_db.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_db_uri_creates_parent_dir(self):
        """db_uri() ensures parent directory exists."""
        mock_db = mock.MagicMock(spec=Path)
        mock_parent = mock.MagicMock(spec=Path)
        mock_db.parent = mock_parent
        with mock.patch("libbee.io.store.DB", mock_db):
            store.db_uri()
            mock_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestWriteTable:
    """Test store.write_table() writing to SQLite."""

    def test_write_table_calls_polars_write(self):
        """write_table() writes to SQLite via Polars."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        with mock.patch.object(df, "write_database") as mock_write:
            with mock.patch("libbee.io.store.db_uri", return_value="sqlite:///test.db"):
                store.write_table(df, "test_table")
                mock_write.assert_called_once()

    def test_write_table_replaces_existing(self):
        """write_table() replaces existing table."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        with mock.patch.object(df, "write_database") as mock_write:
            with mock.patch("libbee.io.store.db_uri", return_value="sqlite:///test.db"):
                store.write_table(df, "test_table")
                call_kwargs = mock_write.call_args[1]
                assert call_kwargs["if_table_exists"] == "replace"


class TestDbHas:
    """Test store.db_has() table existence checking."""

    def test_db_has_true_when_table_exists(self):
        """db_has() returns True when table exists."""
        with mock.patch("sqlite3.connect") as mock_connect:
            mock_conn = mock.MagicMock()
            mock_connect.return_value = mock_conn
            mock_cursor = mock.MagicMock()
            mock_conn.execute.return_value = mock_cursor
            with mock.patch("libbee.io.store.DB", Path("/tmp/test.db")):
                result = store.db_has("test_table")
                assert result is True

    def test_db_has_false_on_error(self):
        """db_has() returns False when table doesn't exist."""
        with mock.patch("sqlite3.connect") as mock_connect:
            mock_conn = mock.MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.execute.side_effect = Exception("table not found")
            with mock.patch("libbee.io.store.DB", Path("/tmp/test.db")):
                result = store.db_has("nonexistent")
                assert result is False


class TestFlatten:
    """Test store.flatten() converting to Parquet/DuckDB."""

    def test_flatten_creates_flat_dir(self):
        """flatten() creates data/flat directory."""
        mock_df = pl.DataFrame({"col": [1]})
        with mock.patch("libbee.io.store.FLAT") as mock_flat:
            mock_flat.mkdir = mock.MagicMock()
            with mock.patch("sqlite3.connect"):
                with mock.patch("duckdb.connect"):
                    with mock.patch("libbee.io.store.pl.read_database", return_value=mock_df):
                        with mock.patch.object(mock_df, "write_parquet"):
                            store.flatten(["test"])
                            mock_flat.mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_flatten_reads_from_sqlite(self):
        """flatten() reads from SQLite."""
        mock_df = pl.DataFrame({"col": [1]})
        mock_flat = mock.MagicMock()
        with mock.patch("libbee.io.store.FLAT", mock_flat):
            with mock.patch("sqlite3.connect") as mock_connect:
                mock_conn = mock.MagicMock()
                mock_connect.return_value = mock_conn
                with mock.patch("duckdb.connect"):
                    with mock.patch("libbee.io.store.pl.read_database", return_value=mock_df) as mock_read:
                        with mock.patch.object(mock_df, "write_parquet"):
                            store.flatten(["test"])
                            mock_read.assert_called()

    def test_flatten_writes_parquet(self):
        """flatten() writes to Parquet."""
        mock_df = pl.DataFrame({"col": [1]})
        mock_flat = mock.MagicMock()
        mock_flat.__truediv__ = mock.MagicMock(return_value="test.parquet")
        with mock.patch("libbee.io.store.FLAT", mock_flat):
            with mock.patch("sqlite3.connect"):
                with mock.patch("duckdb.connect"):
                    with mock.patch("libbee.io.store.pl.read_database", return_value=mock_df):
                        with mock.patch.object(mock_df, "write_parquet") as mock_write:
                            store.flatten(["test"])
                            mock_write.assert_called()


class TestExport:
    """Test store export helpers for alternate formats."""

    def test_export_table_writes_csv(self, tmp_path):
        """export_table() writes a CSV file for one table."""
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        with mock.patch("libbee.io.store.load", return_value=df):
            path = store.export_table("facts", destination=tmp_path)
        assert path == tmp_path / "facts.csv"
        assert path.read_text().splitlines()[0] == "a,b"

    def test_export_table_writes_json(self, tmp_path):
        """export_table() writes row-oriented JSON for one table."""
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        with mock.patch("libbee.io.store.load", return_value=df):
            path = store.export_table("facts", file_format="json", destination=tmp_path)
        assert path == tmp_path / "facts.json"
        assert json.loads(path.read_text()) == [{"a": 1, "b": "x"}]

    def test_export_table_rejects_unknown_format(self, tmp_path):
        """export_table() rejects unsupported output formats."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            store.export_table("facts", file_format=cast(Any, "yaml"), destination=tmp_path)

    def test_export_all_uses_tables_when_names_omitted(self, tmp_path):
        """export_all() exports every available table by default."""
        with mock.patch("libbee.io.store.tables", return_value=["facts", "geo_dim"]):
            with mock.patch("libbee.io.store.export_table") as mock_export:
                mock_export.side_effect = [tmp_path / "facts.csv", tmp_path / "geo_dim.csv"]
                paths = store.export_all(destination=tmp_path)
        assert paths == [tmp_path / "facts.csv", tmp_path / "geo_dim.csv"]
        assert mock_export.call_count == 2
