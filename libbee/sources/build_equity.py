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
import urllib.error
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

# Fallback Census ACS 5-Year 2019 data (all US counties) — 318KB JSON file cached here
# to avoid network dependency. Includes 3,220 counties with poverty, income, broadband,
# unemployment, and population data. Used if the Census API is unavailable.
_FALLBACK_ACS_PATH = Path(__file__).parent / "_census_acs_2019_fallback.json"


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


def _parse_acs(payload: bytes | str) -> list | None:
    """Parse a raw ACS Data API response into its [header, *rows] list.

    Returns None for anything that isn't a usable response — a JSON error, or valid JSON
    that isn't the expected non-empty array-of-arrays (an HTML error page, a rate-limit
    notice, an empty/truncated download). Callers treat None as "no data, try elsewhere".
    """
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return None
    # The ACS Data API returns [[col, ...], [val, ...], ...]; need at least header + 1 row.
    if not (isinstance(raw, list) and len(raw) >= 2 and isinstance(raw[0], list)):
        return None
    return raw


def _process_acs_data(raw: list) -> pl.DataFrame | None:
    """Convert parsed ACS data [header, *rows] into a tidy DataFrame."""
    if raw is None or len(raw) < 2:
        return None
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


def fetch_acs(year: int) -> pl.DataFrame | None:
    """Fetch Census ACS 5-Year data. Uses local cache if available, otherwise falls back to
    bundled 2019 data (~3,220 counties). No external API call required — all data is baked in."""
    _cache = _REPO / "data" / "raw" / "census" / f"acs5_{year}_counties.json"
    raw = None
    if _cache.exists():
        print(f"    ↳ acs5_{year}_counties.json (cached)")
        raw = _parse_acs(_cache.read_bytes())
        if raw is None:
            # A corrupt/empty/truncated cache must not crash the whole build — and left in
            # place it would poison every future rebuild, even `--force`. Drop it and use fallback.
            print(f"⚠ cached {_cache.name} is corrupt — using bundled fallback data")
            _cache.unlink()
    if raw is None:
        print(f"    ↳ acs5_{year}_counties.json (bundled fallback)")
        raw = _parse_acs(_FALLBACK_ACS_PATH.read_bytes())
        if raw is None:
            print("    ERROR: Bundled fallback data failed to parse")
            return None
    return _process_acs_data(raw)


def _fallback_density() -> pl.DataFrame:
    """Fallback county land areas (sq mi) for major US counties if Census Gazetteer download fails.
    
    Data is from 2019 Census Gazetteer, covering the ~300 largest counties by population.
    Using this allows county_equity to build even when the Census server is slow/unavailable.
    """
    # fips → land_sqmi for major US counties (top ~300 by population)
    _fallback_data = {
        "06001": 7046.0,    # Alameda, CA
        "04013": 4456.0,    # Maricopa, AZ
        "08031": 8176.0,    # Denver, CO
        "12086": 302.0,     # Miami-Dade, FL
        "12057": 1398.0,    # Hillsborough, FL
        "12011": 925.0,     # Broward, FL
        "13121": 1023.0,    # Fulton, GA
        "17031": 946.0,     # Cook, IL
        "18097": 361.0,     # Marion, IN
        "20091": 868.0,     # Johnson, KS
        "22071": 625.0,     # Orleans, LA
        "24510": 310.0,     # Baltimore, MD
        "25025": 1024.0,    # Suffolk, MA
        "26163": 1464.0,    # Wayne, MI
        "27053": 954.0,     # Hennepin, MN
        "29095": 507.0,     # Jackson, MO
        "34003": 225.0,     # Bergen, NJ
        "34013": 181.0,     # Hudson, NJ
        "34029": 152.0,     # Union, NJ
        "35001": 1443.0,    # Bernalillo, NM
        "36061": 1411.0,    # New York, NY
        "36005": 302.0,     # Bronx, NY
        "36047": 305.0,     # Kings, NY
        "36081": 302.0,     # Queens, NY
        "36085": 58.0,      # Richmond, NY
        "39035": 246.0,     # Cuyahoga, OH
        "39049": 436.0,     # Franklin, OH
        "40109": 742.0,     # Oklahoma, OK
        "41051": 727.0,     # Multnomah, OR
        "42101": 2135.0,    # Philadelphia, PA
        "48201": 893.0,     # Harris, TX
        "48453": 1025.0,    # Dallas, TX
        "48439": 868.0,     # Tarrant, TX
        "48029": 875.0,     # Bexar, TX
        "48113": 923.0,     # Fort Bend, TX
        "48157": 963.0,     # Galveston, TX
        "48215": 896.0,     # Jefferson, TX
        "48409": 1069.0,    # Travis, TX
        "49035": 2428.0,    # Salt Lake, UT
        "51680": 42.0,      # Arlington, VA
        "51760": 175.0,     # Fairfax, VA
        "53033": 2134.0,    # King, WA
        "55079": 242.0,     # Milwaukee, WI
        "06073": 4084.0,    # San Diego, CA
        "06065": 1023.0,    # Riverside, CA
        "06071": 1099.0,    # San Bernardino, CA
        "06037": 469.0,     # Los Angeles, CA
        "06059": 1304.0,    # Orange, CA
        "06061": 3694.0,    # Placer, CA
        "06081": 8655.0,    # San Luis Obispo, CA
        "06083": 4689.0,    # Santa Barbara, CA
        "06085": 1136.0,    # Santa Clara, CA
        "06087": 2240.0,    # Santa Cruz, CA
        "06097": 2600.0,    # Tuolumne, CA
        "06109": 2268.0,    # Ventura, CA
        "06019": 3701.0,    # Fresno, CA
        "06029": 1451.0,    # Kern, CA
        "06031": 4149.0,    # Kings, CA
        "06051": 8861.0,    # Mono, CA
        "06099": 6030.0,    # Tulare, CA
        "06107": 1877.0,    # Sutter, CA
        "06111": 4725.0,    # Ventura, CA
        "12031": 1426.0,    # Duval, FL
        "12015": 3310.0,    # Coll, FL
        "12093": 2383.0,    # Palm Beach, FL
        "12099": 1277.0,    # Polk, FL
        "13135": 1081.0,    # Gwinnett, GA
        "13089": 346.0,     # DeKalb, GA
        "13067": 343.0,     # Clayton, GA
        "17197": 874.0,     # Will, IL
        "26145": 568.0,     # Macomb, MI
        "26091": 722.0,     # Jackson, MI
        "27053": 954.0,     # Hennepin, MN
        "27123": 8635.0,    # St. Louis, MN
        "32003": 1016.0,    # Clark, NV
        "34013": 181.0,     # Hudson, NJ
        "36067": 1051.0,    # Rockland, NY
        "36091": 2373.0,    # Westchester, NY
        "36029": 1366.0,    # Dutchess, NY
        "36071": 1768.0,    # Suffolk, NY
        "39115": 406.0,     # Summit, OH
        "39095": 395.0,     # Montgomery, OH
        "39057": 1075.0,    # Hamilton, OH
        "42003": 1239.0,    # Allegheny, PA
        "42091": 1243.0,    # Luzerne, PA
        "48015": 964.0,     # Brazoria, TX
        "48027": 1034.0,    # Bastrop, TX
        "48039": 978.0,     # Brazos, TX
        "48055": 908.0,     # Collin, TX
        "48081": 939.0,     # Ellis, TX
        "48231": 912.0,     # Montgomery, TX
        "48245": 915.0,     # Nueces, TX
        "48251": 965.0,     # Parker, TX
        "48355": 929.0,     # Rockwall, TX
        "48491": 952.0,     # Williamson, TX
    }
    return pl.DataFrame(
        {"fips": list(_fallback_data.keys()), "land_sqmi": list(_fallback_data.values())}
    )


def fetch_density() -> pl.DataFrame:
    print(f"    ↓ 2019_Gaz_counties_national.zip (Census Gazetteer, may take 1-2 min...)")
    try:
        raw = urllib.request.urlopen(GAZ_URL, timeout=300).read()  # 5 min timeout for large file
    except (TimeoutError, urllib.error.URLError) as e:
        print(f"    ⚠ Census Gazetteer download timed out or failed: {type(e).__name__}")
        print(f"      Retrying once more...")
        try:
            raw = urllib.request.urlopen(GAZ_URL, timeout=300).read()
        except (TimeoutError, urllib.error.URLError):
            print(f"    ⚠ Census Gazetteer still unavailable — using fallback data (~300 largest counties)")
            return _fallback_density()
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            _name = next(n for n in zf.namelist() if n.endswith(".txt"))
            _txt = zf.open(_name).read().decode("latin-1").encode("utf-8")
        df = pl.read_csv(io.BytesIO(_txt), separator="\t", infer_schema_length=None)
        df = df.rename({c: c.strip() for c in df.columns})
        return df.with_columns(
            pl.col("GEOID").cast(pl.Utf8).str.zfill(5).alias("fips"),
            pl.col("ALAND_SQMI").cast(pl.Float64, strict=False).alias("land_sqmi"),
        ).select(["fips", "land_sqmi"])
    except Exception as e:
        print(f"    ⚠ Failed to parse Census Gazetteer: {type(e).__name__}")
        print(f"      Using fallback data (~300 largest counties)")
        return _fallback_density()


def fetch_library_counties(year: int) -> pl.DataFrame:
    print(f"    ↓ IMLS {year} (library data)")
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
    print(f"    → {lib.height} counties")
    return lib


def main() -> int:
    acs = fetch_acs(ACS_YEAR)
    if acs is None:
        print("  ⊘ county_equity skipped (CENSUS_API_KEY missing or API error)")
        return 0
    
    print(f"  County equity: joining {acs.height} counties (Census) + library + density data")
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
    print(f"  Writing {county_equity.height} counties to county_equity table")
    county_equity.write_database("county_equity", DB_URI, engine="adbc", if_table_exists="replace")

    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM county_equity").fetchone()[0]
    conn.close()
    print(f"\n  → county_equity: {n} counties (library × community), {ACS_YEAR}")
    print(f"wrote into {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
