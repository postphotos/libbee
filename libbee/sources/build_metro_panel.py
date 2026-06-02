"""
build_metro_panel.py — national metro panel for the 30 largest US metros.

A companion to build_db.py. The IMLS Public Libraries Survey is national;
build_db.py keeps only California. This script re-reads the *same cached*
admin-entity files (no new downloads), keeps every state, and rolls each
library up to its metro (Core-Based Statistical Area) for the 30 largest
metros — across 1998–2023, every year IMLS collected visits and open hours.

The geographic key drifts across 26 years, so we chain three:
    • CBSA code .......... present 2011–2023 (use directly)
    • county FIPS ........ FIPSST/FIPSCO 2007–2014, INCITSST/INCITSCO 2015–2021
    • county name ........ 1998–2006 (only CNTY survives)
We learn a county→metro crosswalk from the overlap years (which carry CBSA
*and* a county key) and apply it backward. 1992–1997 are excluded for a hard
data reason: IMLS did not collect VISITS or HRS_OPEN until 1998.

It writes two additive tables into the existing libbee.db, leaving the five
California tables untouched:

    metro_panel   — one row per (metro, year): summed visits/hours/staff/etc.
                    + per-capita columns. ~780 rows (30 metros × 26 years).
    metro_lookup  — CBSA code → metro name, rank, 2025 population, states.

Run (after build_db.py):
    uv run python build_metro_panel.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from libbee.io.paths import ROOT as _REPO

import polars as pl

from . import build_db as b
from .imls_normalized import load_year_normalized

DB_PATH = _REPO / "data" / "db" / "libbee.db"
DB_URI = f"sqlite:///{DB_PATH}"

# FY1992 is the earliest IMLS public-use file. 1992–1997 carry checkouts,
# population, and staff but NOT visits or open hours (those begin FY1998).
METRO_PANEL_YEARS = list(range(1992, 2024))

# code -> (rank, metro name, 2025 population, states).  CBSA codes verified
# against the IMLS FY2023 admin-entity file (dominant state matches each metro).
METRO_CBSA: dict[str, tuple[int, str, int, str]] = {
    "35620": (1,  "New York-Newark-Jersey City",        19641225, "NY-NJ-PA"),
    "31080": (2,  "Los Angeles-Long Beach-Anaheim",     13288904, "CA"),
    "16980": (3,  "Chicago-Naperville-Elgin",            9879320, "IL-IN-WI"),
    "19100": (4,  "Dallas-Fort Worth-Arlington",         7978340, "TX"),
    "26420": (5,  "Houston-The Woodlands-Sugar Land",    7975220, "TX"),
    "12060": (6,  "Atlanta-Sandy Springs-Alpharetta",    6577299, "GA"),
    "47900": (7,  "Washington-Arlington-Alexandria",     6538392, "DC-VA-MD-WV"),
    "33100": (8,  "Miami-Fort Lauderdale-Pompano Beach", 6423080, "FL"),
    "37980": (9,  "Philadelphia-Camden-Wilmington",      6325972, "PA-NJ-DE-MD"),
    "38060": (10, "Phoenix-Mesa-Chandler",               5469313, "AZ"),
    "40140": (11, "Riverside-San Bernardino-Ontario",    5137831, "CA"),
    "14460": (12, "Boston-Cambridge-Newton",             4886137, "MA-NH"),
    "41860": (13, "San Francisco-Oakland-Berkeley",      4550629, "CA"),
    "19820": (14, "Detroit-Warren-Dearborn",             4179716, "MI"),
    "42660": (15, "Seattle-Tacoma-Bellevue",             4051726, "WA"),
    "33460": (16, "Minneapolis-St. Paul-Bloomington",    3722624, "MN-WI"),
    "41740": (17, "San Diego-Chula Vista-Carlsbad",      3530341, "CA"),
    "45300": (18, "Tampa-St. Petersburg-Clearwater",     3348027, "FL"),
    "19740": (19, "Denver-Aurora-Lakewood",              3088903, "CO"),
    "41700": (20, "San Antonio-New Braunfels",           2898825, "TX"),
    "41180": (21, "St. Louis",                           2883091, "MO-IL"),
    "12580": (22, "Baltimore-Columbia-Towson",           2866842, "MD"),
    "36740": (23, "Orlando-Kissimmee-Sanford",           2739553, "FL"),
    "16740": (24, "Charlotte-Concord-Gastonia",          2712935, "NC-SC"),
    "29820": (25, "Las Vegas-Henderson-Paradise",        2697683, "NV"),
    "38900": (26, "Portland-Vancouver-Hillsboro",        2684165, "OR-WA"),
    "12420": (27, "Austin-Round Rock-Georgetown",        2666215, "TX"),
    "40900": (28, "Sacramento-Roseville-Folsom",         2646821, "CA"),
    "18140": (29, "Columbus",                            2306564, "OH"),
    "38300": (30, "Pittsburgh",                          2303068, "PA"),
}

# Metrics that simply don't exist in early years; their all-null group sums must
# stay null (not 0). visits/hours <1998, wifi <2014, programs <2007.
NULLABLE = ["visits", "hours", "wifi", "programs"]


def metro_lookup_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [{"cbsa": code, "rank": rank, "metro": name, "pop_2025": pop, "states": st}
         for code, (rank, name, pop, st) in METRO_CBSA.items()]
    )


def build_panel() -> tuple[pl.DataFrame, pl.DataFrame]:
    years = {y: (u, h) for y, u, h in b.IMLS_YEARS}
    frames = [load_year_normalized(y, *years[y]) for y in METRO_PANEL_YEARS]
    base = pl.concat(frames, how="diagonal_relaxed")

    # Learn the county→metro crosswalk from rows that carry a CBSA *and* a county
    # key, then apply it to the years that don't (FIPS for 2007–10, name for 1998–2006).
    _metro_rows = base.filter(pl.col("cbsa_raw").is_in(list(METRO_CBSA)))
    fips_x = (_metro_rows.select(["stco_key", "cbsa_raw"]).drop_nulls()
              .unique(subset="stco_key", keep="first").rename({"cbsa_raw": "cbsa_fips"}))
    name_x = (_metro_rows.select(["stabr", "cnty_key", "cbsa_raw"]).drop_nulls()
              .unique(subset=["stabr", "cnty_key"], keep="first").rename({"cbsa_raw": "cbsa_name"}))
    print(f"  crosswalk: {fips_x.height} FIPS counties · {name_x.height} (state,name) counties")

    mapped = (
        base.join(fips_x, on="stco_key", how="left")
        .join(name_x, on=["stabr", "cnty_key"], how="left")
        .with_columns(pl.coalesce(["cbsa_raw", "cbsa_fips", "cbsa_name"]).alias("cbsa"))
        .filter(pl.col("cbsa").is_in(list(METRO_CBSA)))
    )
    for _y in METRO_PANEL_YEARS:
        _n = mapped.filter(pl.col("year") == _y).height
        print(f"  ✓ {_y}: {_n} metro library systems")

    panel = (
        mapped.group_by(["cbsa", "year"]).agg(
            pl.len().alias("systems"),
            pl.col("popu_lsa").sum(),
            pl.col("checkouts").sum(),
            pl.col("staff").sum(),
            pl.col("funding").sum(),
            pl.col("branches").sum(),
            pl.col("central").sum(),
            # null-if-all-null: a plain .sum() returns 0 in years a field didn't
            # exist and would masquerade as real data.
            *[pl.when(pl.col(_m).count() > 0).then(pl.col(_m).sum()).otherwise(None).alias(_m)
              for _m in NULLABLE],
        )
        .with_columns(
            # total physical service outlets = central libraries + branches
            (pl.col("central") + pl.col("branches")).alias("outlets"),
            (pl.col("visits") / pl.col("popu_lsa")).alias("visits_pc"),
            (pl.col("hours") / pl.col("popu_lsa")).alias("hours_pc"),
            (pl.col("staff") / pl.col("popu_lsa")).alias("staff_pc"),
            (pl.col("checkouts") / pl.col("popu_lsa")).alias("checkouts_pc"),
            (pl.col("wifi") / pl.col("popu_lsa")).alias("wifi_pc"),
            (pl.col("funding") / pl.col("popu_lsa")).alias("funding_pc"),
            (pl.col("programs") / pl.col("popu_lsa")).alias("programs_pc"),
        )
        .with_columns(
            # outlets per 100k residents — the physical-access measure
            (pl.col("outlets") / pl.col("popu_lsa") * 100_000).alias("outlets_per_100k"))
        .join(metro_lookup_frame().select(["cbsa", "rank", "metro"]), on="cbsa", how="left")
        .sort(["rank", "year"])
    )

    # Library-level table (one row per library-year) so downstream analysis can
    # build a *balanced* panel — libraries reporting in every year of a window —
    # and so escape the composition artifacts that inflate metro-summed hours.
    libraries = (
        mapped.select(["fscskey", "cbsa", "year", "stabr", "popu_lsa",
                       "visits", "checkouts", "wifi", "hours", "staff", "funding", "programs",
                       "branches", "central"])
        .filter(pl.col("fscskey").is_not_null())
        .join(metro_lookup_frame().select(["cbsa", "metro"]), on="cbsa", how="left")
        .sort(["cbsa", "year", "fscskey"])
    )
    return panel, libraries


def main() -> int:
    print("Metro panel · rolling 30 largest US metros up from cached IMLS (1992-2023)")
    panel, libraries = build_panel()
    lookup = metro_lookup_frame()

    panel.write_database("metro_panel", DB_URI, engine="adbc", if_table_exists="replace")
    lookup.write_database("metro_lookup", DB_URI, engine="adbc", if_table_exists="replace")
    libraries.write_database("metro_libraries", DB_URI, engine="adbc", if_table_exists="replace")

    conn = sqlite3.connect(DB_PATH)
    n_panel = conn.execute("SELECT COUNT(*) FROM metro_panel").fetchone()[0]
    n_metros = conn.execute("SELECT COUNT(DISTINCT cbsa) FROM metro_panel").fetchone()[0]
    yrs = conn.execute("SELECT MIN(year), MAX(year) FROM metro_panel").fetchone()
    n_lib = conn.execute("SELECT COUNT(*) FROM metro_libraries").fetchone()[0]
    conn.close()
    print(f"  → metro_panel:     {n_panel} rows · {n_metros} metros · {yrs[0]}-{yrs[1]}")
    print(f"  → metro_lookup:    {len(lookup)} metros")
    print(f"  → metro_libraries: {n_lib} library-years")
    print(f"\nwrote metro tables into {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
