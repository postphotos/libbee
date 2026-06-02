"""
build_equity.py — national county dataset for the "are libraries equalizers?" story.

Generates one local table, `county_equity`, joining:
  • Library provision per county (cached IMLS, 2019) — visits/hours/staff/funding per capita
  • Community need per county (Census ACS 5-Year 2019 Data API) — poverty, median income,
    home-broadband, unemployment

Pure Polars, no pandas. The Census Data API needs a free key; this script reads it from
the CENSUS_API_KEY environment variable and never stores it. Get one at
https://api.census.gov/data/key_signup.html

Run:
    CENSUS_API_KEY=xxxxx uv run python build_equity.py
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import urllib.request
import zipfile
from pathlib import Path

from libbee.io.paths import ROOT as _REPO

import polars as pl

from . import build_db as bd           # IMLS download URLs + helpers
from . import build_metro_panel as mp  # reuse the national per-library loader

DB_PATH = _REPO / "data" / "db" / "libbee.db"
DB_URI = f"sqlite:///{DB_PATH}"
ACS_YEAR = 2019  # ACS 5-Year 2015–2019 (pre-COVID, every county) · library FY2019

# ACS detailed-table variables → tidy names. Rates are computed from count pairs.
ACS_VARS = {
    "B01003_001E": "total_pop",                                  # total population (for density)
    "B17001_002E": "pov_below", "B17001_001E": "pov_univ",      # poverty
    "B19013_001E": "median_income",                              # median household income
    "B28002_004E": "bb_yes", "B28002_001E": "bb_univ",           # broadband subscription
    "B23025_005E": "unemployed", "B23025_003E": "labor_force",   # unemployment
}

# County land area (keyless) → population density for the rural/urban split.
GAZ_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
           "2019_Gazetteer/2019_Gaz_counties_national.zip")


def _read_dotenv_key(name: str) -> str | None:
    env_path = _REPO / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


def _census_api_key() -> str | None:
    return os.environ.get("CENSUS_API_KEY") or _read_dotenv_key("CENSUS_API_KEY")


def fetch_acs(year: int) -> pl.DataFrame | None:
    # Cache the raw ACS response to data/raw so a rebuild is self-contained & key-free after the
    # first fetch (and so the result can't drift if the API changes). The key is only needed once.
    _cache = _REPO / "data" / "raw" / "census" / f"acs5_{year}_counties.json"
    if _cache.exists():
        print(f"Census · ACS5 {year} (cached)")
        raw = json.loads(_cache.read_text())
    else:
        key = _census_api_key()
        if not key:
            print("⚠ Census API key missing — skipping county_equity (optional)")
            print("  To include Census community-need metrics, set CENSUS_API_KEY (free at https://api.census.gov/data/key_signup.html)")
            print("  It's cached after the first fetch, then never needed again.")
            return None
        _get = ",".join(["NAME", *ACS_VARS])
        url = f"https://api.census.gov/data/{year}/acs/acs5?get={_get}&for=county:*&key={key}"
        print(f"Census · ACS5 {year}, all counties (fetching)")
        _bytes = urllib.request.urlopen(url, timeout=90).read()
        _cache.parent.mkdir(parents=True, exist_ok=True)
        _cache.write_bytes(_bytes)
        raw = json.loads(_bytes)
    df = pl.DataFrame(raw[1:], schema=raw[0], orient="row")
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in ACS_VARS])
    df = df.with_columns(
        (pl.col("state") + pl.col("county")).alias("fips"),
        pl.col("B01003_001E").alias("total_pop"),
        (pl.col("B17001_002E") / pl.col("B17001_001E")).alias("poverty"),
        pl.col("B19013_001E").alias("median_income"),
        (1 - pl.col("B28002_004E") / pl.col("B28002_001E")).alias("no_broadband"),
        (pl.col("B23025_005E") / pl.col("B23025_003E")).alias("unemployment"),
    )
    print(f"  → {df.height} counties")
    return df.select(["fips", "NAME", "total_pop", "poverty", "median_income",
                      "no_broadband", "unemployment"])


def fetch_density() -> pl.DataFrame:
    print("Census · gazetteer county land area (keyless)")
    raw = urllib.request.urlopen(GAZ_URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        _name = next(n for n in zf.namelist() if n.endswith(".txt"))
        _txt = zf.open(_name).read().decode("latin-1").encode("utf-8")
    df = pl.read_csv(io.BytesIO(_txt), separator="\t", infer_schema_length=None)
    df = df.rename({c: c.strip() for c in df.columns})
    return df.with_columns(
        pl.col("GEOID").cast(pl.Utf8).str.zfill(5).alias("fips"),
        pl.col("ALAND_SQMI").cast(pl.Float64, strict=False).alias("land_sqmi"),
    ).select(["fips", "land_sqmi"])


def fetch_library_counties(year: int) -> pl.DataFrame:
    yrs = {y: (u, h) for y, u, h in bd.IMLS_YEARS}
    df = mp.load_year_normalized(year, *yrs[year])  # national per-library, has st/co + metrics
    lib = (
        df.filter(pl.col("st").is_not_null() & pl.col("co").is_not_null())
        .with_columns(
            (pl.col("st").cast(pl.Int64).cast(pl.Utf8).str.zfill(2)
             + pl.col("co").cast(pl.Int64).cast(pl.Utf8).str.zfill(3)).alias("fips"))
        .group_by("fips").agg(
            pl.len().alias("systems"),
            pl.col("popu_lsa").sum(),
            pl.col("visits").sum(),
            pl.col("hours").sum(),
            pl.col("staff").sum(),
            pl.col("funding").sum(),
            pl.col("checkouts").sum(),
            pl.col("branches").sum(),
            pl.col("central").sum(),
            pl.when(pl.col("wifi").count() > 0).then(pl.col("wifi").sum()).otherwise(None).alias("wifi"),
        )
        .filter(pl.col("popu_lsa") > 0)
        .with_columns(
            (pl.col("central") + pl.col("branches")).alias("outlets"),
            (pl.col("visits") / pl.col("popu_lsa")).alias("visits_pc"),
            (pl.col("hours") / pl.col("popu_lsa")).alias("hours_pc"),
            (pl.col("staff") / pl.col("popu_lsa")).alias("staff_pc"),
            (pl.col("funding") / pl.col("popu_lsa")).alias("funding_pc"),
            (pl.col("checkouts") / pl.col("popu_lsa")).alias("checkouts_pc"),
            (pl.col("wifi") / pl.col("popu_lsa")).alias("wifi_pc"),
            ((pl.col("central") + pl.col("branches")) / pl.col("popu_lsa") * 100_000).alias("outlets_per_100k"),
        )
    )
    print(f"Library · IMLS {year}: {lib.height} counties with library data")
    return lib


def main() -> int:
    acs = fetch_acs(ACS_YEAR)
    if acs is None:
        print("Skipping county_equity (requires CENSUS_API_KEY)")
        return 0
    
    dens = fetch_density()
    lib = fetch_library_counties(ACS_YEAR)
    county_equity = (
        lib.join(acs, on="fips", how="inner")
        .join(dens, on="fips", how="left")
        .with_columns((pl.col("total_pop") / pl.col("land_sqmi")).alias("density"))
        .filter(pl.col("visits_pc").is_not_null() & pl.col("hours_pc").is_not_null()
                & pl.col("poverty").is_not_null())
        .sort("popu_lsa", descending=True)
    )
    county_equity.write_database("county_equity", DB_URI, engine="adbc", if_table_exists="replace")

    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM county_equity").fetchone()[0]
    conn.close()
    print(f"\n  → county_equity: {n} counties (library × community), {ACS_YEAR}")
    print(f"wrote into {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
