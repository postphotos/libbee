"""HUD adapters — homelessness (+ grant spend), at metro and county grain."""

from __future__ import annotations

import polars as pl

from ..geo import FIPS2ST
from ..io.store import load
from ..sources import build_county_homeless, build_homeless
from .base import Adapter, melt


class HUD(Adapter):
    name, source = "hud", "HUD"
    tables = ("coc_homeless", "metro_homeless", "coc_spend", "metro_spend")
    provenance = "HUD Point-in-Time homeless counts + CoC grant awards (browser-UA past the Akamai wall)"
    facts_after = ("metro",)

    def build(self) -> None:
        build_homeless.main()

    def to_facts(self) -> pl.DataFrame:
        pop = load("metro_panel").select(["cbsa", "metro", "year", "popu_lsa"])
        mhl = load("metro_homeless").join(pop, on=["cbsa", "year"], how="left").with_columns((pl.col("homeless") / pl.col("popu_lsa") * 10000).alias("homeless_per10k"))
        return melt(mhl, "metro", "cbsa", "metro", self.source, year="year")


class CountyHomeless(Adapter):
    name, source = "county_homeless", "HUD"
    tables = ("county_homeless",)
    provenance = "HUD homelessness (2007–2024) + spend (2018–2024) allocated to every county via Byrne's CoC↔county crosswalk"
    build_after = ("hud",)
    facts_after = ("equity",)

    def build(self) -> None:
        build_county_homeless.main()

    def to_facts(self) -> pl.DataFrame:
        try:
            names = load("county_equity").select(["fips", "NAME"])  # names broadcast across years
        except FileNotFoundError:
            # County equity table wasn't built (Census API key was missing); use empty names
            names = pl.DataFrame({"fips": pl.Series([], dtype=pl.Utf8), "NAME": pl.Series([], dtype=pl.Utf8)})
        _st = pl.col("fips").str.slice(0, 2).replace_strict(FIPS2ST, default=None).alias("st")
        chl = load("county_homeless").rename({"spend_pc": "homeless_spend_pc"}).join(names, on="fips", how="inner").with_columns(_st)
        return melt(chl, "county", "fips", "NAME", self.source, year="year", state="st")
