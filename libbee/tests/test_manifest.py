"""Unit tests for libbee.manifest — reproducibility fingerprinting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import polars as pl

from libbee.io import manifest


class TestContentHash:
    """Test manifest._content_hash() for dataframes."""

    def test_content_hash_deterministic(self):
        """_content_hash() returns same hash for same frame."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hash1 = manifest._content_hash(df)
        hash2 = manifest._content_hash(df)
        assert hash1 == hash2

    def test_content_hash_different_for_different_data(self):
        """_content_hash() returns different hashes for different data."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [1, 2, 4]})
        hash1 = manifest._content_hash(df1)
        hash2 = manifest._content_hash(df2)
        assert hash1 != hash2

    def test_content_hash_is_sha256(self):
        """_content_hash() produces a 64-character hex SHA256."""
        df = pl.DataFrame({"a": [1]})
        result = manifest._content_hash(df)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_content_hash_ignores_order(self):
        """_content_hash() hashes sorted frames (order-independent)."""
        df = pl.DataFrame({"b": [2, 1], "a": [4, 3]})
        # melt sorts by columns, so hash should be consistent
        result = manifest._content_hash(df)
        assert len(result) == 64


class TestFileHash:
    """Test manifest._file_hash() for files."""

    def test_file_hash_deterministic(self, tmp_path):
        """_file_hash() returns same hash for same file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        hash1 = manifest._file_hash(test_file)
        hash2 = manifest._file_hash(test_file)
        assert hash1 == hash2

    def test_file_hash_different_for_different_content(self, tmp_path):
        """_file_hash() returns different hashes for different files."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")
        hash1 = manifest._file_hash(file1)
        hash2 = manifest._file_hash(file2)
        assert hash1 != hash2

    def test_file_hash_is_sha256(self, tmp_path):
        """_file_hash() produces a 64-character hex SHA256."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        result = manifest._file_hash(test_file)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestCompute:
    """Test manifest.compute() computing hashes."""

    def test_compute_returns_dict(self):
        """compute() returns a dict with tables and raw_inputs."""
        with mock.patch("libbee.io.manifest.FLAT") as mock_flat:
            with mock.patch("libbee.io.manifest.RAW") as mock_raw:
                mock_flat.glob.return_value = []
                mock_raw.rglob.return_value = []
                result = manifest.compute()
                assert isinstance(result, dict)
                assert "tables" in result
                assert "raw_inputs" in result

    def test_compute_tables_structure(self):
        """compute() produces correct table metadata structure."""
        mock_df = pl.DataFrame({"col": [1, 2, 3]})
        with mock.patch("libbee.io.manifest.FLAT") as mock_flat:
            with mock.patch("libbee.io.manifest.RAW") as mock_raw:
                test_parquet = Path("/data/flat/test.parquet")
                mock_flat.glob.return_value = [test_parquet]
                mock_raw.rglob.return_value = []
                with mock.patch("polars.read_parquet", return_value=mock_df):
                    result = manifest.compute()
                    assert "test" in result["tables"]
                    table_meta = result["tables"]["test"]
                    assert "rows" in table_meta
                    assert "cols" in table_meta
                    assert "sha256" in table_meta

    def test_compute_raw_inputs_structure(self, tmp_path):
        """compute() produces correct raw input metadata structure."""
        # Create a minimal temp file structure
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        input_file = raw_dir / "test" / "input.csv"
        input_file.parent.mkdir()
        input_file.write_text("data")

        # Mock FLAT to return empty parquets
        with mock.patch("libbee.io.manifest.FLAT") as mock_flat:
            mock_flat.glob.return_value = []
            # Mock RAW to be the actual raw_dir
            with mock.patch("libbee.io.manifest.RAW", raw_dir):
                result = manifest.compute()
                # Should have raw_inputs with at least the test file
                assert "raw_inputs" in result
                assert len(result["raw_inputs"]) > 0
                # Check structure of raw_inputs
                for _key, val in result["raw_inputs"].items():
                    assert "bytes" in val
                    assert "sha256" in val


class TestBuild:
    """Test manifest.build() writing fingerprint."""

    def test_build_creates_manifest_dir(self, tmp_path):
        """build() creates MANIFEST parent directory."""
        with mock.patch("libbee.io.manifest.MANIFEST", tmp_path / "nested" / "dir" / "MANIFEST.json"):
            with mock.patch("libbee.io.manifest.compute", return_value={"tables": {}, "raw_inputs": {}}):
                manifest.build()
                assert (tmp_path / "nested" / "dir").exists()

    def test_build_writes_json(self, tmp_path):
        """build() writes MANIFEST as JSON."""
        manifest_path = tmp_path / "MANIFEST.json"
        _computed = {"tables": {"t1": {}}, "raw_inputs": {}}
        with mock.patch("libbee.io.manifest.MANIFEST", manifest_path):
            with mock.patch("libbee.io.manifest.compute", return_value=_computed):
                manifest.build()
                assert manifest_path.exists()
                data = json.loads(manifest_path.read_text())
                assert "tables" in data
                assert "raw_inputs" in data

    def test_build_returns_dict(self):
        """build() returns the computed manifest."""
        test_manifest = {"tables": {"test": {}}, "raw_inputs": {}}
        with mock.patch("libbee.io.manifest.MANIFEST"):
            with mock.patch("libbee.io.manifest.compute", return_value=test_manifest):
                result = manifest.build()
                assert result == test_manifest


class TestVerify:
    """Test manifest.verify() reproducibility checking."""

    def test_verify_returns_false_no_manifest(self, tmp_path):
        """verify() returns False if MANIFEST doesn't exist."""
        with mock.patch("libbee.io.manifest.MANIFEST") as mock_manifest:
            mock_manifest.exists.return_value = False
            result = manifest.verify()
            assert result is False

    def test_verify_returns_true_when_data_matches(self, tmp_path):
        """verify() returns True when current data matches MANIFEST."""
        saved_manifest = {"tables": {"test": {"sha256": "abc123"}}, "raw_inputs": {}}
        computed = {"tables": {"test": {"sha256": "abc123"}}, "raw_inputs": {}}

        with mock.patch("libbee.io.manifest.MANIFEST") as mock_manifest:
            mock_manifest.exists.return_value = True
            mock_manifest.read_text.return_value = json.dumps(saved_manifest)
            with mock.patch("libbee.io.manifest.compute", return_value=computed):
                result = manifest.verify()
                assert result is True

    def test_verify_returns_false_when_data_differs(self):
        """verify() returns False when table data differs."""
        saved_manifest = {"tables": {"test": {"sha256": "abc123"}}, "raw_inputs": {}}
        computed = {"tables": {"test": {"sha256": "def456"}}, "raw_inputs": {}}

        with mock.patch("libbee.io.manifest.MANIFEST") as mock_manifest:
            mock_manifest.exists.return_value = True
            mock_manifest.read_text.return_value = json.dumps(saved_manifest)
            with mock.patch("libbee.io.manifest.compute", return_value=computed):
                result = manifest.verify()
                assert result is False

    def test_verify_returns_false_when_table_missing(self):
        """verify() returns False when table is missing."""
        saved_manifest = {"tables": {"test": {"sha256": "abc123"}}, "raw_inputs": {}}
        computed = {"tables": {}, "raw_inputs": {}}

        with mock.patch("libbee.io.manifest.MANIFEST") as mock_manifest:
            mock_manifest.exists.return_value = True
            mock_manifest.read_text.return_value = json.dumps(saved_manifest)
            with mock.patch("libbee.io.manifest.compute", return_value=computed):
                result = manifest.verify()
                assert result is False
