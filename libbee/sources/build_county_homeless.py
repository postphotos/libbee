"""
build_county_homeless.py — push homelessness + homeless-spend down to every county.

HUD reports homelessness by Continuum of Care (CoC), not county. To run the
"do homelessness and libraries correlate?" test on all 2,758 counties, we use Tom
Byrne's HUD CoC↔county geography crosswalk (2017 CoC boundaries × Census counties),
which ships:
  • county_coc_match.csv — each county FIPS → its CoC(s), with pct_cnty_pop_coc, the
    share of the county's population inside that CoC (for the ~55 split counties).
  • coc_population.csv    — each CoC's total population (the denominator for a *rate*).

Method (ecological, standard): a CoC's homeless RATE = its PIT count / its population,
and its SPEND per capita = its HUD grant $ / its population. Every county inherits its
CoC's rate and spend-per-capita (population-weighted across CoCs for split counties).
So the rate is uniform within a CoC — an assumption we state plainly; it's the accepted
way to get county-level homelessness from CoC data.

Produces `county_homeless`: fips × year × {coc_number, homeless_per10k, spend_pc, spend_per_homeless}
for EVERY year HUD publishes a PIT count (2007–2024); spend_pc is filled only for the years HUD
awards data covers (2018–2024) and is null before that. The 2019 cross-section still lines up with
county_equity's library + ACS vintage. The CoC↔county crosswalk (2017 CoC boundaries) is held fixed
across years — the standard ecological assumption, stated plainly.

Reads coc_homeless / coc_spend / county_equity from the SQLite DB (so build_homeless and
build_equity must have run first). Run:  uv run python build_county_homeless.py
"""

from __future__ import annotations

import sqlite3
import urllib.request

import polars as pl

from libbee.io.paths import ROOT as _REPO

ROOT = _REPO
DB_PATH = ROOT / "data" / "db" / "libbee.db"
DB_URI = f"sqlite:///{DB_PATH}"
RAW = ROOT / "data" / "raw" / "hud"
BASE = "https://raw.githubusercontent.com/tomhbyrne/HUD-CoC-Geography-Crosswalk/master/output"


def _csv(name: str) -> pl.DataFrame:
    dest = RAW / name
    if not dest.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"crosswalk · downloading {name}")
        urllib.request.urlretrieve(f"{BASE}/{name}", dest)
    return pl.read_csv(dest, encoding="utf8-lossy", null_values=["NA"], infer_schema_length=5000)


def main() -> int:
    cw = (_csv("county_coc_match.csv").select(["county_fips", "coc_number", "pct_cnty_pop_coc"])
          .with_columns(pl.col("county_fips").cast(pl.Utf8).str.zfill(5).alias("fips"),
                        pl.col("pct_cnty_pop_coc").fill_null(100.0)))
    cpop = _csv("coc_population.csv").select(["coc_number", "total_population"])

    conn = sqlite3.connect(DB_PATH)
    ch = pl.read_database("SELECT coc_number, year, homeless FROM coc_homeless", conn)
    cs = pl.read_database("SELECT coc_number, year, spend FROM coc_spend", conn)
    conn.close()

    # CoC-level rate + spend per capita, every year a PIT count exists (spend null pre-2018).
    coc = (ch.join(cpop, on="coc_number", how="left").join(cs, on=["coc_number", "year"], how="left")
           .with_columns((pl.col("homeless") / pl.col("total_population") * 10000).alias("homeless_per10k"),
                         (pl.col("spend") / pl.col("total_population")).alias("spend_pc")))

    # Population-weighted down to county (handles the ~55 split counties), per year.
    j = cw.join(coc.select(["coc_number", "year", "homeless_per10k", "spend_pc"]), on="coc_number", how="left")
    county = (j.group_by("fips", "year").agg(
                (pl.col("homeless_per10k") * pl.col("pct_cnty_pop_coc")).sum()
                / pl.col("pct_cnty_pop_coc").sum(),
                # spend is weighted only over the CoCs that actually report it that year
                (pl.col("spend_pc") * pl.col("pct_cnty_pop_coc")).filter(pl.col("spend_pc").is_not_null()).sum()
                / pl.col("pct_cnty_pop_coc").filter(pl.col("spend_pc").is_not_null()).sum(),
                pl.col("coc_number").sort_by("pct_cnty_pop_coc", descending=True).first())
              .with_columns(
                  (pl.col("spend_pc") / (pl.col("homeless_per10k") / 10000)).alias("spend_per_homeless"))
              .filter(pl.col("homeless_per10k").is_not_null() & pl.col("year").is_not_null())
              .select("fips", "coc_number", "year", "homeless_per10k", "spend_pc", "spend_per_homeless")
              .sort("fips", "year"))

    county.write_database("county_homeless", DB_URI, engine="adbc", if_table_exists="replace")
    ny = county["year"].n_unique()
    nc = county["fips"].n_unique()
    yr = (county["year"].min(), county["year"].max())
    print(f"  → county_homeless: {county.height:,} rows · {nc:,} counties × {ny} years ({yr[0]}–{yr[1]})")
    print(f"wrote into {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
