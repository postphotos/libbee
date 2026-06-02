"""
build_county_panel.py — county × year library panel, as BROAD as the data allows.

The catch: county FIPS only appears in the IMLS files 2007–2021 (FIPSST/FIPSCO → INCITSST/INCITSCO);
1998–2006 and 2022–2023 carry only the county *name* (CNTY). So we learn a (state, county-name) →
FIPS crosswalk from the years that have both, and backfill the rest — the same technique the metro
panel uses for CBSA. That stretches county coverage from a single year (2019) to ~1998–2023.

Produces `county_panel`: fips × year × visits/checkouts/WiFi/hours/staff/funding (per-capita) +
cost_per_visit + outlets_per_100k. Reuses build_metro_panel.load_year_normalized.
"""

from __future__ import annotations

import sqlite3

import polars as pl

from libbee.io.paths import DB

from . import build_db as bd
from . import build_metro_panel as mp

DB_URI = f"sqlite:///{DB}"
_METRICS = ["popu_lsa", "visits", "checkouts", "wifi", "hours", "staff", "funding", "branches", "central"]


def main() -> int:
    parts = []
    for _y, _u, _h in bd.IMLS_YEARS:
        df = mp.load_year_normalized(_y, _u, _h)  # national per-library, harmonised
        parts.append(df.select(["year", "stabr", "cnty_key", "st", "co", *_METRICS]))
    lib = pl.concat(parts, how="vertical_relaxed")

    # Direct FIPS where the state+county codes exist (2007–2021).
    lib = lib.with_columns(
        pl.when(pl.col("st").is_not_null() & pl.col("co").is_not_null())
        .then(pl.col("st").cast(pl.Int64).cast(pl.Utf8).str.zfill(2)
              + pl.col("co").cast(pl.Int64).cast(pl.Utf8).str.zfill(3))
        .otherwise(None).alias("fips_direct"))

    # Learn (state, county-name) → FIPS from the rows that carry both, apply to the rest.
    xwalk = (lib.filter(pl.col("fips_direct").is_not_null() & pl.col("cnty_key").is_not_null())
             .select(["stabr", "cnty_key", "fips_direct"]).unique(subset=["stabr", "cnty_key"], keep="first")
             .rename({"fips_direct": "fips_xw"}))
    lib = (lib.join(xwalk, on=["stabr", "cnty_key"], how="left")
           .with_columns(pl.coalesce(["fips_direct", "fips_xw"]).alias("fips")))

    agg = (lib.filter(pl.col("fips").is_not_null() & (pl.col("popu_lsa") > 0))
           .group_by("fips", "year").agg(
               pl.col("popu_lsa").sum(),
               pl.col("visits").sum(), pl.col("checkouts").sum(),
               pl.when(pl.col("wifi").count() > 0).then(pl.col("wifi").sum()).otherwise(None).alias("wifi"),
               pl.col("hours").sum(), pl.col("staff").sum(), pl.col("funding").sum(),
               (pl.col("branches").sum() + pl.col("central").sum()).alias("outlets"))
           .filter(pl.col("popu_lsa") > 0)
           .with_columns(
               (pl.col("visits") / pl.col("popu_lsa")).alias("visits_pc"),
               (pl.col("checkouts") / pl.col("popu_lsa")).alias("checkouts_pc"),
               (pl.col("wifi") / pl.col("popu_lsa")).alias("wifi_pc"),
               (pl.col("hours") / pl.col("popu_lsa")).alias("hours_pc"),
               (pl.col("staff") / pl.col("popu_lsa")).alias("staff_pc"),
               (pl.col("funding") / pl.col("popu_lsa")).alias("funding_pc"),
               (pl.col("funding") / pl.col("visits")).alias("cost_per_visit"),
               (pl.col("outlets") / pl.col("popu_lsa") * 100_000).alias("outlets_per_100k"))
           .sort("fips", "year"))
    agg.write_database("county_panel", DB_URI, engine="adbc", if_table_exists="replace")

    conn = sqlite3.connect(DB)
    n = conn.execute("SELECT COUNT(*) FROM county_panel").fetchone()[0]
    ny = conn.execute("SELECT COUNT(DISTINCT year) FROM county_panel").fetchone()[0]
    nc = conn.execute("SELECT COUNT(DISTINCT fips) FROM county_panel").fetchone()[0]
    yr = conn.execute("SELECT MIN(year), MAX(year) FROM county_panel").fetchone()
    conn.close()
    print(f"  → county_panel: {n:,} rows · {nc:,} counties × {ny} years ({yr[0]}–{yr[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
