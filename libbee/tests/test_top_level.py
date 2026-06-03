"""Unit tests for libbee.__init__ and libbee.__main__ — top-level API and CLI."""

from __future__ import annotations

from unittest import mock

import polars as pl

from libbee import __main__


class TestCliMain:
    """Test the CLI main() function."""

    def test_main_default_build(self):
        """main() defaults to 'build' command."""
        with mock.patch("sys.argv", ["libbee"]):
            with mock.patch("libbee.__main__.build") as mock_build:
                __main__.main()
                mock_build.assert_called_once_with(force=False, verbose=True)

    def test_main_build_command(self):
        """main() accepts 'build' command."""
        with mock.patch("sys.argv", ["libbee", "build"]):
            with mock.patch("libbee.__main__.build") as mock_build:
                __main__.main()
                mock_build.assert_called_once_with(force=False, verbose=True)

    def test_main_build_force_flag(self):
        """main() honors --force flag."""
        with mock.patch("sys.argv", ["libbee", "build", "--force"]):
            with mock.patch("libbee.__main__.build") as mock_build:
                __main__.main()
                mock_build.assert_called_once_with(force=True, verbose=True)

    def test_main_verify_command(self):
        """main('verify') calls verify and returns status code."""
        with mock.patch("sys.argv", ["libbee", "verify"]):
            with mock.patch("libbee.__main__.verify") as mock_verify:
                mock_verify.return_value = True
                exit_code = __main__.main()
                assert exit_code == 0
                mock_verify.assert_called_once()

    def test_main_verify_failure_returns_1(self):
        """main() returns 1 when verify() fails."""
        with mock.patch("sys.argv", ["libbee", "verify"]):
            with mock.patch("libbee.__main__.verify") as mock_verify:
                mock_verify.return_value = False
                exit_code = __main__.main()
                assert exit_code == 1

    def test_main_adapters_command(self):
        """main('adapters') lists adapters."""
        mock_adapter1 = mock.MagicMock()
        mock_adapter1.__str__ = mock.MagicMock(return_value="adapter1")
        mock_adapter1.provenance = "http://example.com"
        mock_adapter2 = mock.MagicMock()
        mock_adapter2.__str__ = mock.MagicMock(return_value="adapter2")
        mock_adapter2.provenance = "http://example2.com"

        with mock.patch("sys.argv", ["libbee", "adapters"]):
            with mock.patch("libbee.__main__.adapters", [mock_adapter1, mock_adapter2]):
                with mock.patch("builtins.print") as mock_print:
                    exit_code = __main__.main()
                    assert exit_code == 0
                    assert mock_print.call_count >= 2

    def test_main_export_table_command(self):
        """main('export <table>') exports one table as CSV by default."""
        with mock.patch("sys.argv", ["libbee", "export", "facts"]):
            with mock.patch("libbee.__main__.export_table") as mock_export:
                exit_code = __main__.main()
                assert exit_code == 0
                mock_export.assert_called_once_with("facts", file_format="csv", destination=None)

    def test_main_export_all_command(self):
        """main('export --all') exports every table in the requested format."""
        with mock.patch("sys.argv", ["libbee", "export", "--all", "--format", "json", "--out", "out"]):
            with mock.patch("libbee.__main__.export_all") as mock_export_all:
                exit_code = __main__.main()
                assert exit_code == 0
                mock_export_all.assert_called_once_with(file_format="json", destination="out")

    def test_main_demo_returns_1_when_marimo_is_unavailable(self):
        """main('demo') exits cleanly with guidance when marimo is not installed."""
        with mock.patch("sys.argv", ["libbee", "demo"]):
            with mock.patch("importlib.util.find_spec", return_value=None):
                with mock.patch("builtins.print") as mock_print:
                    exit_code = __main__.main()

        assert exit_code == 1
        mock_print.assert_called_once_with("Error: marimo is not installed. Please install it with: pip install 'libbee[notebook]'")

    def test_main_returns_0_on_success(self):
        """main() returns 0 on successful build."""
        with mock.patch("sys.argv", ["libbee", "build"]):
            with mock.patch("libbee.__main__.build"):
                exit_code = __main__.main()
                assert exit_code == 0

    def test_main_help(self):
        """main() prints help when --help is passed."""
        with mock.patch("sys.argv", ["libbee", "--help"]):
            with mock.patch("builtins.print") as mock_print:
                exit_code = __main__.main()
                assert exit_code == 0
                mock_print.assert_called_once()

    def test_main_with_sys_exit(self):
        """main() can be called via if __name__ == '__main__' with sys.exit."""
        with mock.patch("sys.argv", ["libbee", "verify"]):
            with mock.patch("libbee.__main__.verify") as mock_verify:
                mock_verify.return_value = True
                exit_code = __main__.main()
                assert exit_code == 0


class TestTopLevelAPI:
    """Test libbee.__init__ top-level API."""

    def test_facts_function_exists(self):
        """libbee.facts() function is accessible."""
        import libbee

        assert hasattr(libbee, "facts")
        assert callable(libbee.facts)

    def test_geo_dim_function_exists(self):
        """libbee.geo_dim() function is accessible."""
        import libbee

        assert hasattr(libbee, "geo_dim")
        assert callable(libbee.geo_dim)

    def test_load_function_exists(self):
        """libbee.load() function is accessible."""
        import libbee

        assert hasattr(libbee, "load")
        assert callable(libbee.load)

    def test_scan_function_exists(self):
        """libbee.scan() function is accessible."""
        import libbee

        assert hasattr(libbee, "scan")
        assert callable(libbee.scan)

    def test_tables_function_exists(self):
        """libbee.tables() function is accessible."""
        import libbee

        assert hasattr(libbee, "tables")
        assert callable(libbee.tables)

    def test_export_functions_exist(self):
        """libbee export helpers are accessible."""
        import libbee

        assert hasattr(libbee, "export_table")
        assert callable(libbee.export_table)
        assert hasattr(libbee, "export_all")
        assert callable(libbee.export_all)

    def test_build_function_exists(self):
        """libbee.build() function is accessible."""
        import libbee

        assert hasattr(libbee, "build")
        assert callable(libbee.build)

    def test_verify_function_exists(self):
        """libbee.verify() function is accessible."""
        import libbee

        assert hasattr(libbee, "verify")
        assert callable(libbee.verify)

    def test_adapters_registry_exists(self):
        """libbee.adapters registry is accessible."""
        import libbee

        assert hasattr(libbee, "adapters")

    def test_all_exports(self):
        """libbee.__all__ is properly defined."""
        import libbee

        assert hasattr(libbee, "__all__")
        expected_exports = [
            # data product
            "adapters",
            "build",
            "verify",
            "load",
            "scan",
            "tables",
            "export_table",
            "export_all",
            "facts",
            "geo_dim",
            # analytical surface (pure, render-free)
            "analysis",
            "frames",
            "geo",
            "inflation",
            "metrics",
            # validated config + metadata-level data contracts (Pydantic)
            "config",
            "settings",
            "schemas",
        ]
        assert set(libbee.__all__) == set(expected_exports)
        # every advertised name resolves
        assert all(hasattr(libbee, name) for name in libbee.__all__)
        # the render layer is opt-in: NOT auto-exported by the data core
        assert "viz" not in libbee.__all__ and "screenshots" not in libbee.__all__

    def test_version_defined(self):
        """libbee.__version__ is defined."""
        import libbee

        assert hasattr(libbee, "__version__")
        assert isinstance(libbee.__version__, str)


class TestFactsFunction:
    """Test the facts() convenience function."""

    def test_facts_calls_load(self):
        """facts() loads the 'facts' table."""
        with mock.patch("libbee.load") as mock_load:
            mock_load.return_value = pl.DataFrame()
            import libbee

            libbee.facts()
            mock_load.assert_called_once_with("facts")

    def test_facts_returns_dataframe(self):
        """facts() returns a DataFrame."""
        mock_df = pl.DataFrame({"col": [1, 2, 3]})
        with mock.patch("libbee.load", return_value=mock_df):
            import libbee

            result = libbee.facts()
            assert isinstance(result, pl.DataFrame)


class TestGeoDimFunction:
    """Test the geo_dim() convenience function."""

    def test_geo_dim_calls_load(self):
        """geo_dim() loads the 'geo_dim' table."""
        with mock.patch("libbee.load") as mock_load:
            mock_load.return_value = pl.DataFrame()
            import libbee

            libbee.geo_dim()
            mock_load.assert_called_once_with("geo_dim")

    def test_geo_dim_returns_dataframe(self):
        """geo_dim() returns a DataFrame."""
        mock_df = pl.DataFrame({"geo_level": ["state"], "geo_id": ["08"]})
        with mock.patch("libbee.load", return_value=mock_df):
            import libbee

            result = libbee.geo_dim()
            assert isinstance(result, pl.DataFrame)
