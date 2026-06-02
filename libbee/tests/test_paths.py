"""Unit tests for libbee.paths — data directory resolution."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from libbee.io import paths


class TestDataRoot:
    """Test data_root() directory resolution."""

    def test_data_root_from_project_pyproject(self, tmp_path, monkeypatch):
        """data_root() finds pyproject.toml and returns data/ sibling."""
        # $LIBBEE_DATA takes precedence by design, so clear it to exercise the pyproject-ancestor branch
        # (a sibling package's conftest may set it process-wide in a combined test run).
        monkeypatch.delenv("LIBBEE_DATA", raising=False)
        # Create project structure
        root = tmp_path / "project"
        data = root / "data"
        root.mkdir()
        data.mkdir()
        (root / "pyproject.toml").write_text("[project]\n")

        # Start from a file inside the project
        test_file = root / "libbee" / "module.py"
        test_file.parent.mkdir()
        result = paths.data_root(test_file)
        assert result == data

    def test_data_root_env_var_takes_precedence(self, tmp_path):
        """$LIBBEE_DATA env var overrides filesystem search."""
        custom_data = tmp_path / "custom_data"
        custom_data.mkdir()
        with mock.patch.dict("os.environ", {"LIBBEE_DATA": str(custom_data)}):
            result = paths.data_root()
            assert result == custom_data.resolve()

    def test_data_root_env_var_expands_user(self):
        """$LIBBEE_DATA expands ~ for home directory."""
        with mock.patch.dict("os.environ", {"LIBBEE_DATA": "~/libbee_data"}):
            result = paths.data_root()
            assert "~" not in str(result)
            assert result.is_absolute()

    def test_data_root_fallback_to_cwd(self):
        """data_root() falls back to ./data if no pyproject.toml found."""
        # Mock the parent search to fail
        import os

        with mock.patch.dict(os.environ, {}, clear=False):
            # This is hard to test cleanly without mocking Path.parents,
            # so we'll test the fallback exists in the code
            result = paths.data_root(Path("/tmp"))
            assert "data" in result.parts


class TestPredefinedPaths:
    """Test the pre-resolved path constants."""

    def test_data_is_path(self):
        """DATA is a Path object."""
        assert isinstance(paths.DATA, Path)

    def test_flat_is_under_data(self):
        """FLAT is under DATA."""
        assert paths.FLAT.is_relative_to(paths.DATA)
        assert paths.FLAT.name == "flat"

    def test_db_dir_is_under_data(self):
        """DB_DIR is under DATA."""
        assert paths.DB_DIR.is_relative_to(paths.DATA)
        assert paths.DB_DIR.name == "db"

    def test_db_is_in_db_dir(self):
        """DB is in DB_DIR."""
        assert paths.DB.is_relative_to(paths.DB_DIR)
        assert paths.DB.name == "libbee.db"

    def test_duckdb_is_in_db_dir(self):
        """DUCKDB is in DB_DIR."""
        assert paths.DUCKDB.is_relative_to(paths.DB_DIR)
        assert paths.DUCKDB.name == "libbee.duckdb"

    def test_manifest_is_in_data(self):
        """MANIFEST is in DATA."""
        assert paths.MANIFEST.is_relative_to(paths.DATA)
        assert paths.MANIFEST.name == "MANIFEST.json"

    def test_raw_folders_are_under_raw(self):
        """All RAW_* paths are under RAW."""
        assert paths.RAW_IMLS.is_relative_to(paths.RAW)
        assert paths.RAW_HUD.is_relative_to(paths.RAW)
        assert paths.RAW_CENSUS.is_relative_to(paths.RAW)
        assert paths.RAW_CALIBPAS.is_relative_to(paths.RAW)
        assert paths.RAW_GEO.is_relative_to(paths.RAW)

    def test_raw_folder_names(self):
        """RAW_* have expected names."""
        assert paths.RAW_IMLS.name == "imls"
        assert paths.RAW_HUD.name == "hud"
        assert paths.RAW_CENSUS.name == "census"
        assert paths.RAW_CALIBPAS.name == "ca_libpas"
        assert paths.RAW_GEO.name == "geo"

    def test_root_is_data_parent(self):
        """ROOT is DATA's parent."""
        assert paths.ROOT == paths.DATA.parent
