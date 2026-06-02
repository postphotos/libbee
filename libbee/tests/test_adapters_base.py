"""Unit tests for libbee.adapters.base — adapter framework and melt helper."""

from __future__ import annotations

from unittest import mock

import polars as pl
import pytest

from libbee.adapters import base


class TestMelt:
    """Test the melt() unpivot helper."""

    def test_melt_basic_cross_section(self):
        """melt() unpivots conformed metrics into facts."""
        df = pl.DataFrame(
            {
                "county_fips": ["08001", "08003"],
                "county_name": ["County A", "County B"],
                "visits_pc": [3.0, 3.5],
                "checkouts_pc": [2.0, 2.2],
                "funding_pc": [50.0, 60.0],
            }
        )
        result = base.melt(
            df,
            geo_level="county",
            geo_id="county_fips",
            geo_name="county_name",
            source="TEST",
            year=2023,
        )
        assert result.columns == [
            "geo_level",
            "geo_id",
            "geo_name",
            "state",
            "year",
            "metric",
            "value",
            "source",
        ]
        assert result.height == 6  # 2 counties × 3 metrics
        assert set(result["metric"]) == {"visits_pc", "checkouts_pc", "funding_pc"}
        assert all(result["year"] == 2023)
        assert all(result["source"] == "TEST")

    def test_melt_with_time_series(self):
        """melt() handles year as a column name."""
        df = pl.DataFrame(
            {
                "state": ["CO", "CO"],
                "year": [2022, 2023],
                "visits_pc": [3.0, 3.1],
            }
        )
        result = base.melt(
            df,
            geo_level="state",
            geo_id="state",
            geo_name="state",
            source="TEST",
            year="year",
        )
        assert set(result["year"]) == {2022, 2023}

    def test_melt_with_state_column(self):
        """melt() includes state column when provided."""
        df = pl.DataFrame(
            {
                "fips": ["08001"],
                "name": ["Denver County, CO"],
                "state": ["CO"],
                "visits_pc": [3.0],
            }
        )
        result = base.melt(
            df,
            geo_level="county",
            geo_id="fips",
            geo_name="name",
            source="TEST",
            year=2023,
            state="state",
        )
        assert all(result["state"] == "CO")

    def test_melt_drops_nulls(self):
        """melt() drops rows where value is null."""
        df = pl.DataFrame(
            {
                "id": ["a", "b"],
                "name": ["A", "B"],
                "visits_pc": [3.0, None],
            }
        )
        result = base.melt(
            df,
            geo_level="test",
            geo_id="id",
            geo_name="name",
            source="TEST",
            year=2023,
        )
        assert result.height == 1

    def test_melt_custom_metrics(self):
        """melt() respects custom metrics list."""
        df = pl.DataFrame(
            {
                "id": ["a"],
                "name": ["A"],
                "visits_pc": [3.0],
                "checkouts_pc": [2.0],
                "other": [1.0],
            }
        )
        result = base.melt(
            df,
            geo_level="test",
            geo_id="id",
            geo_name="name",
            source="TEST",
            year=2023,
            metrics=["visits_pc"],
        )
        assert set(result["metric"]) == {"visits_pc"}

    def test_melt_empty_frame(self):
        """melt() handles empty frames."""
        df = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Utf8),
                "name": pl.Series([], dtype=pl.Utf8),
                "visits_pc": pl.Series([], dtype=pl.Float64),
            }
        )
        result = base.melt(
            df,
            geo_level="test",
            geo_id="id",
            geo_name="name",
            source="TEST",
            year=2023,
        )
        assert result.height == 0
        assert result.columns == [
            "geo_level",
            "geo_id",
            "geo_name",
            "state",
            "year",
            "metric",
            "value",
            "source",
        ]


class TestAdapter:
    """Test the Adapter base class."""

    def test_adapter_fields(self):
        """Adapter has expected class fields."""
        assert hasattr(base.Adapter, "name")
        assert hasattr(base.Adapter, "source")
        assert hasattr(base.Adapter, "tables")
        assert hasattr(base.Adapter, "provenance")
        assert hasattr(base.Adapter, "build_after")
        assert hasattr(base.Adapter, "facts_after")

    def test_adapter_build_not_implemented(self):
        """Adapter.build() raises NotImplementedError by default."""
        adapter = base.Adapter()
        with pytest.raises(NotImplementedError):
            adapter.build()

    def test_adapter_to_facts_default(self):
        """Adapter.to_facts() returns None by default."""
        adapter = base.Adapter()
        assert adapter.to_facts() is None

    def test_adapter_present_empty(self):
        """Adapter.present() returns True when no tables."""
        adapter = base.Adapter()
        adapter.tables = ()
        assert adapter.present()

    def test_adapter_present_checks_all_tables(self):
        """Adapter.present() checks all tables exist."""
        adapter = base.Adapter()
        adapter.tables = ("table1", "table2")
        with mock.patch("libbee.adapters.base.db_has") as mock_db_has:
            mock_db_has.side_effect = [True, False]
            assert not adapter.present()
            assert mock_db_has.call_count == 2

    def test_adapter_repr(self):
        """Adapter.__repr__() shows type, name, source, tables."""
        adapter = base.Adapter()
        adapter.name = "test"
        adapter.source = "Test Source"
        adapter.tables = ("t1", "t2")
        repr_str = repr(adapter)
        assert "Adapter" in repr_str
        assert "test" in repr_str
        assert "Test Source" in repr_str
        assert "t1" in repr_str

    def test_order_adapters_orders_by_dependency(self):
        """order_adapters() returns a stable topological order."""

        class First(base.Adapter):
            name = "first"

        class Second(base.Adapter):
            name = "second"
            build_after = ("first",)

        class Third(base.Adapter):
            name = "third"
            build_after = ("second",)

        ordered = base.order_adapters([Third(), First(), Second()], attr="build_after")
        assert [adapter.name for adapter in ordered] == ["first", "second", "third"]

    def test_order_adapters_rejects_unknown_dependency(self):
        """order_adapters() raises on unknown dependencies."""

        class Broken(base.Adapter):
            name = "broken"
            build_after = ("missing",)

        with pytest.raises(ValueError, match="unknown build_after dependency"):
            base.order_adapters([Broken()], attr="build_after")

    def test_order_adapters_rejects_cycles(self):
        """order_adapters() raises on cyclic dependencies."""

        class First(base.Adapter):
            name = "first"
            build_after = ("second",)

        class Second(base.Adapter):
            name = "second"
            build_after = ("first",)

        with pytest.raises(ValueError, match="cyclic build_after dependencies"):
            base.order_adapters([First(), Second()], attr="build_after")


class TestAdapterSubclass:
    """Test creating a concrete Adapter subclass."""

    def test_adapter_subclass_with_all_fields(self):
        """Concrete Adapter subclass works as expected."""

        class TestAdapter(base.Adapter):
            name = "test"
            source = "Test Data"
            tables = ("test_table",)
            provenance = "https://example.com"

            def build(self):
                pass

        adapter = TestAdapter()
        assert adapter.name == "test"
        assert adapter.source == "Test Data"
        assert adapter.tables == ("test_table",)
        assert adapter.provenance == "https://example.com"
        adapter.build()  # Should not raise

    def test_adapter_subclass_to_facts_override(self):
        """Subclass can override to_facts()."""

        class TestAdapter(base.Adapter):
            name = "test"
            tables = ()

            def to_facts(self):
                return pl.DataFrame({"col": [1, 2, 3]})

        adapter = TestAdapter()
        facts = adapter.to_facts()
        assert facts is not None
        assert facts.height == 3
