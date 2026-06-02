"""Unit tests for libbee.pipeline — orchestration."""

from __future__ import annotations

from unittest import mock

import pytest

from libbee import pipeline


class TestBuild:
    """Test pipeline.build() orchestration."""

    def test_build_calls_adapters(self):
        """build() calls build() on each adapter."""
        with mock.patch("libbee.pipeline.REGISTRY") as mock_registry:
            mock_adapter1 = mock.MagicMock()
            mock_adapter2 = mock.MagicMock()
            mock_adapter1.present.return_value = False
            mock_adapter2.present.return_value = True
            mock_registry.__iter__.return_value = [mock_adapter1, mock_adapter2]

            with mock.patch("libbee.pipeline.flatten"):
                with mock.patch("libbee.pipeline.unify.build"):
                    with mock.patch("libbee.pipeline.manifest.build"):
                        pipeline.build(force=False, verbose=False)

            mock_adapter1.build.assert_called_once()
            mock_adapter2.build.assert_not_called()

    def test_build_force_rebuilds_all(self):
        """build(force=True) rebuilds all adapters."""
        with mock.patch("libbee.pipeline.REGISTRY") as mock_registry:
            mock_adapter = mock.MagicMock()
            mock_adapter.present.return_value = True
            mock_registry.__iter__.return_value = [mock_adapter]

            with mock.patch("libbee.pipeline.flatten"):
                with mock.patch("libbee.pipeline.unify.build"):
                    with mock.patch("libbee.pipeline.manifest.build"):
                        pipeline.build(force=True, verbose=False)

            mock_adapter.build.assert_called_once()

    def test_build_flattens_source_tables(self):
        """build() flattens source tables to Parquet."""
        with mock.patch("libbee.pipeline.REGISTRY") as mock_registry:
            mock_adapter = mock.MagicMock()
            mock_adapter.present.return_value = True
            mock_adapter.tables = ("table1", "table2")
            mock_registry.__iter__.return_value = [mock_adapter]

            with mock.patch("libbee.pipeline.flatten") as mock_flatten:
                with mock.patch("libbee.pipeline.unify.build"):
                    with mock.patch("libbee.pipeline.manifest.build"):
                        pipeline.build(force=False, verbose=False)

            mock_flatten.assert_any_call(["table1", "table2"])

    def test_build_builds_facts_and_geo_dim(self):
        """build() flattens facts and geo_dim after unify."""
        with mock.patch("libbee.pipeline.REGISTRY") as mock_registry:
            mock_adapter = mock.MagicMock()
            mock_adapter.present.return_value = True
            mock_adapter.tables = ()
            mock_registry.__iter__.return_value = [mock_adapter]

            with mock.patch("libbee.pipeline.flatten") as mock_flatten:
                with mock.patch("libbee.pipeline.unify.build"):
                    with mock.patch("libbee.pipeline.manifest.build"):
                        pipeline.build(force=False, verbose=False)

            calls = mock_flatten.call_args_list
            assert any("facts" in str(call) for call in calls)
            assert any("geo_dim" in str(call) for call in calls)

    def test_build_calls_manifest(self):
        """build() builds the manifest fingerprint."""
        with mock.patch("libbee.pipeline.REGISTRY") as mock_registry:
            mock_adapter = mock.MagicMock()
            mock_adapter.present.return_value = True
            mock_adapter.tables = ()
            mock_registry.__iter__.return_value = [mock_adapter]

            with mock.patch("libbee.pipeline.flatten"):
                with mock.patch("libbee.pipeline.unify.build"):
                    with mock.patch("libbee.pipeline.manifest.build") as mock_manifest_build:
                        pipeline.build(force=False, verbose=False)

            mock_manifest_build.assert_called_once()

    def test_build_verbose_prints_adapter_info(self):
        """build(verbose=True) prints adapter information."""
        with mock.patch("libbee.pipeline.REGISTRY") as mock_registry:
            mock_adapter = mock.MagicMock()
            mock_adapter.present.return_value = False
            mock_adapter.name = "test_adapter"
            mock_adapter.source = "test_source"
            mock_adapter.provenance = "http://example.com"
            mock_adapter.tables = ()
            mock_registry.__iter__.return_value = [mock_adapter]

            with mock.patch("libbee.pipeline.flatten"):
                with mock.patch("libbee.pipeline.unify.build"):
                    with mock.patch("libbee.pipeline.manifest.build"):
                        with mock.patch("builtins.print") as mock_print:
                            pipeline.build(force=False, verbose=True)
                            mock_print.assert_called()
                            call_str = str(mock_print.call_args_list)
                            assert "test_adapter" in call_str

    def test_build_orders_adapters_by_declared_dependencies(self):
        """build() honors adapter build_after dependencies."""
        built: list[str] = []

        def _adapter(name, deps=()):
            adapter = mock.MagicMock()
            adapter.name = name
            adapter.source = name
            adapter.provenance = name
            adapter.tables = ()
            adapter.build_after = deps
            adapter.present.return_value = False
            adapter.build.side_effect = lambda: built.append(name)
            return adapter

        third = _adapter("third", ("second",))
        first = _adapter("first")
        second = _adapter("second", ("first",))

        with mock.patch("libbee.pipeline.REGISTRY", [third, first, second]):
            with mock.patch("libbee.pipeline.flatten"):
                with mock.patch("libbee.pipeline.unify.build"):
                    with mock.patch("libbee.pipeline.manifest.build"):
                        pipeline.build(force=False, verbose=False)

        assert built == ["first", "second", "third"]

    def test_build_rejects_unknown_dependency(self):
        """build() raises when an adapter depends on an unknown name."""
        adapter = mock.MagicMock()
        adapter.name = "county_homeless"
        adapter.source = "HUD"
        adapter.provenance = "test"
        adapter.tables = ()
        adapter.build_after = ("hud",)
        adapter.present.return_value = False

        with mock.patch("libbee.pipeline.REGISTRY", [adapter]):
            with mock.patch("libbee.pipeline.flatten"):
                with mock.patch("libbee.pipeline.unify.build"):
                    with mock.patch("libbee.pipeline.manifest.build"):
                        with pytest.raises(ValueError, match="unknown build_after dependency"):
                            pipeline.build(force=False, verbose=False)


class TestVerify:
    """Test pipeline.verify() reproducibility check."""

    def test_verify_returns_true_on_success(self):
        """verify() returns True when data matches."""
        with mock.patch("libbee.pipeline.manifest.verify") as mock_verify:
            mock_verify.return_value = True
            result = pipeline.verify()
            assert result is True

    def test_verify_returns_false_on_failure(self):
        """verify() returns False when data differs."""
        with mock.patch("libbee.pipeline.manifest.verify") as mock_verify:
            mock_verify.return_value = False
            result = pipeline.verify()
            assert result is False

    def test_verify_calls_manifest_verify(self):
        """verify() delegates to manifest.verify()."""
        with mock.patch("libbee.pipeline.manifest.verify") as mock_verify:
            mock_verify.return_value = True
            pipeline.verify()
            mock_verify.assert_called_once()
