"""IMLS adapters — the national Public Library Survey, at three grains."""

from __future__ import annotations

import polars as pl

from ..geo import FIPS2ST
from ..io.store import load
from ..sources import build_county_panel, build_db, build_metro_panel, build_state_panel
from .base import Adapter, melt


class Core(Adapter):
    name, source = "core", "IMLS+MS+NOAA"
    tables = ("imls_admin", "ca_annual", "ca_county_recent", "ms_broadband", "noaa_daily")
    provenance = "IMLS PLS zips (1992–2023) + Microsoft measured broadband + NOAA daily highs"

    def build(self) -> None:
        build_db.main()

    # No direct facts rows — the conformed metrics come from the state/metro/county adapters.


class Metro(Adapter):
    name, source = "metro", "IMLS"
    tables = ("metro_panel", "metro_libraries", "metro_lookup")
    provenance = "the 30 largest US metros × year, rolled up from IMLS per-library"

    def build(self) -> None:
        build_metro_panel.main()

    def to_facts(self) -> pl.DataFrame:
        d = load("metro_panel").with_columns((pl.col("funding") / pl.col("visits")).alias("cost_per_visit"))
        return melt(d, "metro", "cbsa", "metro", self.source, year="year")


class State(Adapter):
    name, source = "state", "IMLS"
    tables = ("state_panel",)
    provenance = "state × year (1992–2023) from the harmonised IMLS per-library data"

    def build(self) -> None:
        build_state_panel.main()

    def to_facts(self) -> pl.DataFrame:
        d = load("state_panel").with_columns(pl.col("st").alias("name"))
        return melt(d, "state", "st", "name", self.source, year="year", state="st")


class CountyPanel(Adapter):
    name, source = "county_panel", "IMLS"
    tables = ("county_panel",)
    provenance = "county × year (1992–2023) from IMLS per-library (FIPS backfilled via a name crosswalk)"
    facts_after = ("equity",)

    def build(self) -> None:
        build_county_panel.main()

    def to_facts(self) -> pl.DataFrame:
        try:
            names = load("county_equity").select(["fips", "NAME"])  # canonical names where we have them
        except FileNotFoundError:
            # County equity table wasn't built (Census API key was missing); use empty names
            names = pl.DataFrame({"fips": pl.Series([], dtype=pl.Utf8), "NAME": pl.Series([], dtype=pl.Utf8)})
        d = (
            load("county_panel")
            .join(names, on="fips", how="left")
            .with_columns(
                pl.coalesce([pl.col("NAME"), pl.col("fips")]).alias("name"),
                pl.col("fips").str.slice(0, 2).replace_strict(FIPS2ST, default=None).alias("st"),
            )
        )
        return melt(d, "county", "fips", "name", self.source, year="year", state="st")
