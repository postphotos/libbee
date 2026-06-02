"""Census + IMLS county adapter — the 2019 county equity frame.

Contributes two slices to facts: the library metrics (source IMLS) and the community-need metrics
(source Census), so each value carries its true provenance even though one adapter builds them.
"""

from __future__ import annotations

import polars as pl

from ..geo import FIPS2ST
from ..io.store import load
from ..sources import build_equity
from .base import Adapter, melt


class Equity(Adapter):
    name, source = "equity", "IMLS+Census"
    tables = ("county_equity",)
    provenance = "2,758 counties (2019): IMLS library × Census ACS need (cached) × gazetteer density"

    def build(self) -> None:
        build_equity.main()

    def to_facts(self) -> pl.DataFrame:
        # Only the Census *need* metrics — the county library metrics come from the (all-year)
        # county_panel adapter now, so this avoids a 2019 duplicate.
        # If Census API key was missing, county_equity won't exist; gracefully return empty.
        try:
            c = load("county_equity").with_columns(pl.col("fips").str.slice(0, 2).replace_strict(FIPS2ST, default=None).alias("st"))
            return melt(
                c.select(["fips", "NAME", "st", "poverty", "median_income", "no_broadband", "density"]),
                "county",
                "fips",
                "NAME",
                "Census",
                year=2019,
                state="st",
            )
        except FileNotFoundError:
            # County equity table wasn't built (Census API key was missing); return empty facts
            return pl.DataFrame(
                {
                    "geo_level": [],
                    "geo_id": [],
                    "geo_name": [],
                    "state": [],
                    "year": [],
                    "metric": [],
                    "value": [],
                    "source": [],
                }
            ).cast(
                {
                    "geo_level": pl.Utf8,
                    "geo_id": pl.Utf8,
                    "geo_name": pl.Utf8,
                    "state": pl.Utf8,
                    "year": pl.Int64,
                    "metric": pl.Utf8,
                    "value": pl.Float64,
                    "source": pl.Utf8,
                }
            )
