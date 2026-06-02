"""Unit tests for libbee.frames — frame loading and preparation."""

from __future__ import annotations

from unittest import mock

import polars as pl

from libbee import frames


class TestTablesList:
    """Test the TABLES constant."""

    def test_tables_list_exists(self):
        """TABLES is defined."""
        assert hasattr(frames, "TABLES")
        assert isinstance(frames.TABLES, list)

    def test_tables_non_empty(self):
        """TABLES contains expected table names."""
        assert len(frames.TABLES) > 0
        assert "facts" in frames.TABLES
        assert "geo_dim" in frames.TABLES

    def test_tables_all_strings(self):
        """All items in TABLES are strings."""
        assert all(isinstance(t, str) for t in frames.TABLES)

    def test_tables_no_duplicates(self):
        """TABLES has no duplicates."""
        assert len(frames.TABLES) == len(set(frames.TABLES))


class TestPrepare:
    """Test frames.prepare() ensuring data exists."""

    def test_prepare_creates_flat_dir(self):
        """prepare() creates the FLAT directory."""
        with mock.patch("libbee.io.frames.FLAT") as mock_flat:
            mock_flat.mkdir = mock.MagicMock()
            with mock.patch("libbee.io.frames.glob.glob", return_value=[]):
                with mock.patch("libbee.build"):
                    frames.prepare()
                    mock_flat.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_prepare_checks_existing_tables(self):
        """prepare() checks which tables already exist."""
        with mock.patch("libbee.io.frames.FLAT") as mock_flat:
            mock_flat.mkdir = mock.MagicMock()
            with mock.patch("libbee.io.frames.glob.glob") as mock_glob:
                mock_glob.return_value = [
                    "/data/flat/facts.parquet",
                    "/data/flat/geo_dim.parquet",
                ]
                with mock.patch("libbee.build"):
                    frames.prepare()
                    mock_glob.assert_called()

    def test_prepare_calls_build_if_missing(self):
        """prepare() calls build() if tables are missing."""
        with mock.patch("libbee.io.frames.FLAT") as mock_flat:
            mock_flat.mkdir = mock.MagicMock()
            with mock.patch("libbee.io.frames.glob.glob", return_value=[]):
                with mock.patch("libbee.build") as mock_build:
                    frames.prepare()
                    mock_build.assert_called_once()

    def test_prepare_skips_build_if_all_present(self):
        """prepare() skips build() if all tables exist."""
        all_paths = [f"/data/flat/{t}.parquet" for t in frames.TABLES]
        with mock.patch("libbee.io.frames.FLAT") as mock_flat:
            mock_flat.mkdir = mock.MagicMock()
            with mock.patch("libbee.io.frames.glob.glob", return_value=all_paths):
                with mock.patch("libbee.build") as mock_build:
                    frames.prepare()
                    mock_build.assert_not_called()

    def test_prepare_checks_only_requested_tables(self):
        """prepare(names=...) only checks the requested tables."""
        with mock.patch("libbee.io.frames.FLAT") as mock_flat:
            mock_flat.mkdir = mock.MagicMock()
            with mock.patch("libbee.io.frames.glob.glob", return_value=["/data/flat/facts.parquet"]):
                with mock.patch("libbee.build") as mock_build:
                    frames.prepare(["facts"])
                    mock_build.assert_not_called()


class TestToDuckdb:
    """Test frames.to_duckdb() materializing to DuckDB."""

    def test_to_duckdb_returns_path(self):
        """to_duckdb() returns a string path."""
        with mock.patch("libbee.io.frames.DUCKDB") as mock_duckdb:
            mock_duckdb.exists.return_value = True
            with mock.patch("duckdb.connect"):
                result = frames.to_duckdb()
                assert isinstance(result, str)

    def test_to_duckdb_creates_connection(self):
        """to_duckdb() creates a DuckDB connection."""
        with mock.patch("libbee.io.frames.DUCKDB") as mock_duckdb:
            mock_duckdb.exists.return_value = False
            with mock.patch("libbee.io.frames.glob.glob", return_value=[]):
                with mock.patch("duckdb.connect") as mock_connect:
                    mock_conn = mock.MagicMock()
                    mock_connect.return_value = mock_conn
                    frames.to_duckdb()
                    mock_connect.assert_called()

    def test_to_duckdb_loads_parquet_files(self):
        """to_duckdb() loads all Parquet files into DuckDB."""
        with mock.patch("libbee.io.frames.DUCKDB") as mock_duckdb:
            mock_duckdb.exists.return_value = False
            test_files = ["/data/flat/facts.parquet", "/data/flat/geo_dim.parquet"]
            with mock.patch("libbee.io.frames.glob.glob", return_value=test_files):
                with mock.patch("duckdb.connect") as mock_connect:
                    mock_conn = mock.MagicMock()
                    mock_connect.return_value = mock_conn
                    frames.to_duckdb()
                    assert mock_conn.execute.call_count >= 2

    def test_to_duckdb_skips_private_tables(self):
        """to_duckdb() skips files starting with _ (private tables)."""
        with mock.patch("libbee.io.frames.DUCKDB") as mock_duckdb:
            mock_duckdb.exists.return_value = False
            test_files = ["/data/flat/facts.parquet", "/data/flat/_temp.parquet"]
            with mock.patch("libbee.io.frames.glob.glob", return_value=test_files):
                with mock.patch("duckdb.connect") as mock_connect:
                    mock_conn = mock.MagicMock()
                    mock_connect.return_value = mock_conn
                    frames.to_duckdb()
                    calls_str = " ".join(str(c) for c in mock_conn.execute.call_args_list)
                    assert "facts" in calls_str or "CREATE" in calls_str


class TestLoad:
    """Test frames.load() loading frames."""

    def test_load_returns_dict(self):
        """load() returns a dictionary."""
        with mock.patch("libbee.io.frames.prepare"):
            with mock.patch("libbee.io.frames.store.load") as mock_load:
                mock_load.return_value = pl.DataFrame({"col": [1]})
                result = frames.load()
        assert isinstance(result, dict)
        assert len(result) == len(frames.TABLES)

    def test_load_calls_prepare(self):
        """load() calls prepare() first."""
        with mock.patch("libbee.io.frames.prepare") as mock_prepare:
            with mock.patch("libbee.io.frames.store.load", return_value=pl.DataFrame()):
                frames.load()
        mock_prepare.assert_called_once()

    def test_load_reads_all_requested_tables(self):
        """load() reads each requested table."""
        with mock.patch("libbee.io.frames.prepare"):
            with mock.patch("libbee.io.frames.store.load") as mock_load:
                mock_df = pl.DataFrame({"col": [1]})
                mock_load.return_value = mock_df
                result = frames.load()
        assert mock_load.call_count == len(frames.TABLES)
        assert all(isinstance(v, pl.DataFrame) for v in result.values())

    def test_load_keys_match_requested_names(self):
        """load() keys match requested table names."""
        with mock.patch("libbee.io.frames.prepare"):
            with mock.patch("libbee.io.frames.store.load", return_value=pl.DataFrame({"col": [1]})):
                result = frames.load(["facts", "geo_dim"])
        assert set(result.keys()) == {"facts", "geo_dim"}


class TestScan:
    """Test frames.scan() loading selected frames lazily."""

    def test_scan_calls_prepare(self):
        """scan() ensures the requested tables are prepared first."""
        mock_lazy = mock.MagicMock(spec=pl.LazyFrame)
        with mock.patch("libbee.io.frames.prepare") as mock_prepare:
            with mock.patch("libbee.io.frames.store.scan", return_value=mock_lazy):
                frames.scan(["facts"])
        mock_prepare.assert_called_once_with(["facts"])

    def test_scan_returns_requested_subset(self):
        """scan(names=...) returns only the requested lazy frames."""
        mock_lazy = mock.MagicMock(spec=pl.LazyFrame)
        with mock.patch("libbee.io.frames.prepare"):
            with mock.patch("libbee.io.frames.store.scan", return_value=mock_lazy) as mock_scan:
                result = frames.scan(["facts", "geo_dim"])
        assert set(result.keys()) == {"facts", "geo_dim"}
        assert mock_scan.call_count == 2
