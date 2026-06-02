"""Unit tests for libbee.pipeline.unify — facts table unification."""

from __future__ import annotations

from unittest import mock

import polars as pl
import pytest

from libbee.pipeline import unify


class TestUnifyBuild:
    """Test unify.build() concatenating facts."""

    def test_build_concatenates_adapter_facts(self):
        """unify.build() concatenates facts from all adapters."""
        facts1 = pl.DataFrame(
            {
                "geo_level": ["state"],
                "geo_id": ["08"],
                "geo_name": ["Colorado"],
                "state": ["CO"],
                "year": [2023],
                "metric": ["visits_pc"],
                "value": [3.0],
                "source": ["IMLS"],
            }
        )
        facts2 = pl.DataFrame(
            {
                "geo_level": ["state"],
                "geo_id": ["36"],
                "geo_name": ["New York"],
                "state": ["NY"],
                "year": [2023],
                "metric": ["visits_pc"],
                "value": [3.5],
                "source": ["IMLS"],
            }
        )

        mock_adapter1 = mock.MagicMock()
        mock_adapter1.to_facts.return_value = facts1
        mock_adapter2 = mock.MagicMock()
        mock_adapter2.to_facts.return_value = facts2

        with mock.patch("libbee.pipeline.unify.REGISTRY", [mock_adapter1, mock_adapter2]):
            with mock.patch("libbee.pipeline.unify.write_table") as mock_write:
                unify.build()
                assert mock_write.call_count == 2

    def test_build_filters_empty_frames(self):
        """unify.build() skips None and empty frames."""
        facts = pl.DataFrame(
            {
                "geo_level": ["state"],
                "geo_id": ["08"],
                "geo_name": ["Colorado"],
                "state": ["CO"],
                "year": [2023],
                "metric": ["visits_pc"],
                "value": [3.0],
                "source": ["IMLS"],
            }
        )

        mock_adapter1 = mock.MagicMock()
        mock_adapter1.to_facts.return_value = facts
        mock_adapter2 = mock.MagicMock()
        mock_adapter2.to_facts.return_value = None
        mock_adapter3 = mock.MagicMock()
        mock_adapter3.to_facts.return_value = pl.DataFrame()

        with mock.patch("libbee.pipeline.unify.REGISTRY", [mock_adapter1, mock_adapter2, mock_adapter3]):
            with mock.patch("libbee.pipeline.unify.write_table") as mock_write:
                unify.build()
                assert mock_write.call_count == 2

    def test_build_deduplicates_facts(self):
        """unify.build() removes duplicate fact rows."""
        duplicate_facts = pl.DataFrame(
            {
                "geo_level": ["state", "state"],
                "geo_id": ["08", "08"],
                "geo_name": ["Colorado", "Colorado"],
                "state": ["CO", "CO"],
                "year": [2023, 2023],
                "metric": ["visits_pc", "visits_pc"],
                "value": [3.0, 3.0],
                "source": ["IMLS", "IMLS"],
            }
        )

        mock_adapter = mock.MagicMock()
        mock_adapter.to_facts.return_value = duplicate_facts

        with mock.patch("libbee.pipeline.unify.REGISTRY", [mock_adapter]):
            with mock.patch("libbee.pipeline.unify.write_table") as mock_write:
                unify.build()
                facts_call = mock_write.call_args_list[0]
                facts_df = facts_call[0][0]
                assert facts_df.height == 1

    def test_build_sorts_facts(self):
        """unify.build() sorts facts by key columns."""
        unsorted_facts = pl.DataFrame(
            {
                "geo_level": ["county", "state", "county"],
                "geo_id": ["08003", "08", "08001"],
                "geo_name": ["County B", "Colorado", "County A"],
                "state": ["CO", "CO", "CO"],
                "year": [2023, 2023, 2023],
                "metric": ["visits_pc", "visits_pc", "visits_pc"],
                "value": [3.0, 2.0, 2.5],
                "source": ["IMLS", "IMLS", "IMLS"],
            }
        )

        mock_adapter = mock.MagicMock()
        mock_adapter.to_facts.return_value = unsorted_facts

        with mock.patch("libbee.pipeline.unify.REGISTRY", [mock_adapter]):
            with mock.patch("libbee.pipeline.unify.write_table") as mock_write:
                unify.build()
                facts_df = mock_write.call_args_list[0][0][0]
                sorted_df = facts_df.sort("geo_level", "geo_id", "year", "metric")
                assert facts_df.equals(sorted_df)

    def test_build_creates_geo_dim(self):
        """unify.build() creates geo_dim from facts."""
        facts = pl.DataFrame(
            {
                "geo_level": ["state", "state"],
                "geo_id": ["08", "36"],
                "geo_name": ["Colorado", "New York"],
                "state": ["CO", "NY"],
                "year": [2023, 2023],
                "metric": ["visits_pc", "visits_pc"],
                "value": [3.0, 3.5],
                "source": ["IMLS", "IMLS"],
            }
        )

        mock_adapter = mock.MagicMock()
        mock_adapter.to_facts.return_value = facts

        with mock.patch("libbee.pipeline.unify.REGISTRY", [mock_adapter]):
            with mock.patch("libbee.pipeline.unify.write_table") as mock_write:
                unify.build()
                geo_dim_call = mock_write.call_args_list[1]
                assert geo_dim_call[0][1] == "geo_dim"
                geo_dim_df = geo_dim_call[0][0]
                assert set(geo_dim_df.columns) == {"geo_level", "geo_id", "geo_name", "state"}

    def test_build_geo_dim_unique(self):
        """unify.build() creates unique geo_dim entries."""
        facts = pl.DataFrame(
            {
                "geo_level": ["state", "state", "state"],
                "geo_id": ["08", "08", "08"],
                "geo_name": ["Colorado", "Colorado", "Colorado"],
                "state": ["CO", "CO", "CO"],
                "year": [2022, 2023, 2024],
                "metric": ["visits_pc", "checkouts_pc", "visits_pc"],
                "value": [3.0, 2.0, 3.1],
                "source": ["IMLS", "IMLS", "IMLS"],
            }
        )

        mock_adapter = mock.MagicMock()
        mock_adapter.to_facts.return_value = facts

        with mock.patch("libbee.pipeline.unify.REGISTRY", [mock_adapter]):
            with mock.patch("libbee.pipeline.unify.write_table") as mock_write:
                unify.build()
                geo_dim_df = mock_write.call_args_list[1][0][0]
                assert geo_dim_df.height == 1

    def test_build_orders_facts_by_declared_dependencies(self):
        """unify.build() honors adapter facts_after dependencies."""
        order: list[str] = []
        facts = pl.DataFrame(
            {
                "geo_level": ["state"],
                "geo_id": ["08"],
                "geo_name": ["Colorado"],
                "state": ["CO"],
                "year": [2023],
                "metric": ["visits_pc"],
                "value": [3.0],
                "source": ["IMLS"],
            }
        )

        def _adapter(name, deps=()):
            adapter = mock.MagicMock()
            adapter.name = name
            adapter.facts_after = deps
            adapter.to_facts.side_effect = lambda: order.append(name) or facts
            return adapter

        county = _adapter("county_panel", ("equity",))
        equity = _adapter("equity")

        with mock.patch("libbee.pipeline.unify.REGISTRY", [county, equity]):
            with mock.patch("libbee.pipeline.unify.write_table"):
                unify.build()

        assert order == ["equity", "county_panel"]

    def test_build_rejects_unknown_facts_dependency(self):
        """unify.build() raises on unknown facts_after dependencies."""
        adapter = mock.MagicMock()
        adapter.name = "county_panel"
        adapter.facts_after = ("equity",)
        adapter.to_facts.return_value = pl.DataFrame()

        with mock.patch("libbee.pipeline.unify.REGISTRY", [adapter]):
            with mock.patch("libbee.pipeline.unify.write_table"):
                with pytest.raises(ValueError, match="unknown facts_after dependency"):
                    unify.build()
