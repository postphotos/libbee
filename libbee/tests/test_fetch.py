"""Unit tests for libbee.fetch — caching download helper."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from libbee.io import fetch


class _Response(io.BytesIO):
    """Context-manager response stub for urllib."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class TestCached:
    """Test fetch.cached() download and caching."""

    def test_cached_returns_path(self):
        """cached() returns a Path object."""
        with mock.patch("urllib.request.urlopen", return_value=_Response(b"test data")):
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("libbee.io.fetch.RAW", Path(tmpdir)):
                    result = fetch.cached("http://example.com/file.csv", "test.csv")
                    assert isinstance(result, Path)
                    assert result.name == "test.csv"

    def test_cached_creates_parent_directories(self):
        """cached() creates parent directories if they don't exist."""
        with mock.patch("urllib.request.urlopen", return_value=_Response(b"test data")):
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("libbee.io.fetch.RAW", Path(tmpdir)):
                    result = fetch.cached("http://example.com/nested/file.csv", "nested/test.csv")
                    assert result.parent.exists()

    def test_cached_downloads_file(self):
        """cached() downloads file and writes to disk."""
        mock_data = b"test file content"
        with mock.patch("urllib.request.urlopen", return_value=_Response(mock_data)):
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("libbee.io.fetch.RAW", Path(tmpdir)):
                    result = fetch.cached("http://example.com/file.csv", "test.csv")
                    assert result.read_bytes() == mock_data

    def test_cached_skips_redownload_if_exists(self):
        """cached() returns cached file without re-downloading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("libbee.io.fetch.RAW", Path(tmpdir)):
                dest = Path(tmpdir) / "test.csv"
                dest.write_bytes(b"cached data")
                with mock.patch("urllib.request.urlopen") as mock_urlopen:
                    result = fetch.cached("http://example.com/file.csv", "test.csv")
                    mock_urlopen.assert_not_called()
                    assert result.read_bytes() == b"cached data"

    def test_cached_respects_min_size(self):
        """cached() re-downloads if file is smaller than min_size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("libbee.io.fetch.RAW", Path(tmpdir)):
                dest = Path(tmpdir) / "test.csv"
                dest.write_bytes(b"x")
                with mock.patch(
                    "urllib.request.urlopen",
                    return_value=_Response(b"new large content"),
                ) as mock_urlopen:
                    result = fetch.cached("http://example.com/file.csv", "test.csv", min_size=10)
                    mock_urlopen.assert_called_once()
                    assert result.read_bytes() == b"new large content"

    def test_cached_rejects_too_small_download_and_cleans_temp(self):
        """cached() rejects undersized downloads and removes temp files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("libbee.io.fetch.RAW", Path(tmpdir)):
                with mock.patch("urllib.request.urlopen", return_value=_Response(b"x")):
                    with pytest.raises(ValueError, match="Downloaded file too small"):
                        fetch.cached("http://example.com/file.csv", "test.csv", min_size=10)
                assert not (Path(tmpdir) / "test.csv").exists()
                assert not any(Path(tmpdir).glob(".test.csv.*"))

    def test_cached_sets_user_agent_when_browser_ua_true(self):
        """cached() sets User-Agent header when browser_ua=True."""
        with mock.patch("urllib.request.Request") as mock_request:
            with mock.patch("urllib.request.urlopen", return_value=_Response(b"test")):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with mock.patch("libbee.io.fetch.RAW", Path(tmpdir)):
                        fetch.cached("http://example.com/file.csv", "test.csv", browser_ua=True)
                        mock_request.assert_called_once()
                        call_kwargs = mock_request.call_args[1]
                        assert "headers" in call_kwargs
                        assert call_kwargs["headers"]["User-Agent"] == fetch.UA

    def test_cached_no_user_agent_when_browser_ua_false(self):
        """cached() doesn't set User-Agent when browser_ua=False."""
        with mock.patch("urllib.request.Request") as mock_request:
            with mock.patch("urllib.request.urlopen", return_value=_Response(b"test")):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with mock.patch("libbee.io.fetch.RAW", Path(tmpdir)):
                        fetch.cached("http://example.com/file.csv", "test.csv", browser_ua=False)
                        mock_request.assert_called_once()
                        call_kwargs = mock_request.call_args[1]
                        assert call_kwargs["headers"] == {}

    def test_cached_timeout_set(self):
        """cached() sets a timeout for downloads."""
        with mock.patch("urllib.request.urlopen", return_value=_Response(b"test")) as mock_urlopen:
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("libbee.io.fetch.RAW", Path(tmpdir)):
                    fetch.cached("http://example.com/file.csv", "test.csv")
                    assert mock_urlopen.call_args[1]["timeout"] == 180
