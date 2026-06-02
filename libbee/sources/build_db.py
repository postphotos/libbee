"""
build_db.py — one shot to assemble the unified Libbee database.

Pulls real public data from three sources and writes a single SQLite DB
(`data/db/libbee.db`) that every phase notebook reads from.

**Note:** The IMLS admin-entity table (per-library system) is filtered to **California only**
to keep the demo manageable. Metro, state, and HUD data are national.

    IMLS Public Libraries Survey, 1992–2023   (admin-entity files, CA only; state/metro national)
    Microsoft USBroadbandUsagePercentages     (CA counties)
    NOAA Daily Summaries / GHCN-Daily         (a few CA stations, summer 2023)

Dataframes are Polars throughout; the SQLite write goes through Polars'
ADBC engine, so this pipeline is pandas-free end to end.  We build into a
temporary DB and only swap it over the live file on full success — a failed
or offline run can never leave you with a half-written database.

Run:
    uv run python build_db.py
"""

from __future__ import annotations

import io
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

from libbee.io.paths import ROOT as _REPO
from libbee.io.data_sources_config import IMLS_YEARS

import polars as pl
import requests

ROOT = _REPO
RAW = ROOT / "data" / "raw" / "imls"
DB_DIR = ROOT / "data" / "db"
DB_PATH = DB_DIR / "libbee.db"

RAW.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "libbee-demo/1.0 (devrel marimo talk)"}


# Some years bury the admin-entity CSV under a sub-folder (e.g. CSV/, PLS_FY*_AE/).
# We just glob for the first .csv whose name looks "admin-entity-shaped".
def _find_ae_csv(z: zipfile.ZipFile, hint: str) -> str:
    names = z.namelist()
    # exact match first
    for n in names:
        if n.endswith(hint):
            return n
    # heuristic — files holding ~thousands of rows are the AE file
    candidates = [
        n for n in names
        if n.lower().endswith(".csv")
        and "outlet" not in n.lower()
        and "sum" not in n.lower()
        and "state" not in n.lower()
    ]
    # pick largest
    candidates.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no AE-shaped CSV in {z.filename}; saw {names}")
    return candidates[0]


# Map every alias we've seen across 32 years onto a tidy canonical name.
# Values not present in a year stay null.  This is the schema-drift fix.
IMLS_RENAME = {
    "STABR":     "stabr",
    "FSCSKEY":   "fscskey",
    "FSCSKey":   "fscskey",
    "CNTY":      "county",
    "POPU":      "popu_lsa",        # 1992-style population
    "POPU_LSA":  "popu_lsa",        # 2010+ style
    "POPU_UND":  "popu_und",
    "POPU_UNDUP":"popu_und",
    "VISITS":    "visits",
    "TOTCIR":    "totcir",
    "WIFISESS":  "wifisess",
    "GPTERMS":   "gpterms",         # public-access terminals
    "PITUSR":    "pitusr",          # public internet computer users
    "HRS_OPEN":  "hrs_open",
    "TOTSTAFF":  "totstaff",
    "TOTPEMP":   "totstaff",        # 1992-style staffing
    "BRANLIB":   "branlib",         # branch count
    "CENTLIB":   "centlib",         # central library count
    "BKMOB":     "bkmob",           # bookmobile count
    "YR":        "yr_sub",
    "YR_SUB":    "yr_sub",
    "OBEREG":    "obereg",
    "FIPSST":    "fipsst",
    "FIPSCO":    "fipsco",
}

CANONICAL = [
    "year", "stabr", "fscskey", "county",
    "popu_lsa", "visits", "totcir", "wifisess",
    "gpterms", "pitusr", "hrs_open", "totstaff",
    "branlib", "centlib", "bkmob",
    "fipsst", "fipsco",
]

# IMLS uses sentinel codes for missing values; treat them as null on read.
IMLS_NULLS = ["", " ", "-1", "-3", "-4", "-9", "M"]

# Numeric columns to coerce after the all-string read.
IMLS_NUMERIC = [
    "popu_lsa", "visits", "totcir", "wifisess", "gpterms", "pitusr",
    "hrs_open", "totstaff", "branlib", "centlib", "bkmob",
]


def fetch_zip(year: int, url: str) -> Path:
    dest = RAW / f"imls_{year}.zip"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    print(f"  ↓ downloading {year} from {url}")
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_imls_year(year: int, url: str, hint: str) -> pl.DataFrame | None:
    try:
        zpath = fetch_zip(year, url)
    except Exception as e:
        print(f"  ✗ {year}: download failed — {e}")
        return None

    with zipfile.ZipFile(zpath) as z:
        try:
            inner = _find_ae_csv(z, hint)
        except FileNotFoundError as e:
            print(f"  ✗ {year}: {e}")
            return None
        with z.open(inner) as fh:
            raw = fh.read()

    # Early IMLS files are Windows-1252-ish.  Polars' CSV reader only speaks
    # utf-8, so decode latin-1 → re-encode utf-8 before parsing.  Read every
    # column as a string (infer_schema_length=0) and coerce the numerics
    # ourselves — that's the schema-drift-proof path across 32 years.
    buf = io.BytesIO(raw.decode("latin-1").encode("utf-8"))
    try:
        df = pl.read_csv(
            buf,
            infer_schema_length=0,        # all columns → Utf8
            null_values=IMLS_NULLS,
            truncate_ragged_lines=True,
        )
    except Exception as e:
        print(f"  ✗ {year}: parse failed — {e}")
        return None

    # Normalise column case so 'stabr'/'STABR' both work.
    if "STABR" not in df.columns:
        df = df.rename({c: c.upper() for c in df.columns})

    # Filter to California — the trim that makes the whole pipeline laptop-sized.
    df = df.filter(pl.col("STABR").str.strip_chars().str.to_uppercase() == "CA")

    # Map known aliases to canonical names, taking the first source seen per
    # target so a year carrying both POPU and POPU_LSA can't collide.
    keep: dict[str, str] = {}
    taken: set[str] = set()
    for src, tgt in IMLS_RENAME.items():
        if src in df.columns and tgt not in taken:
            keep[src] = tgt
            taken.add(tgt)
    df = df.rename(keep)

    # Stamp the survey year, then make sure every canonical column exists.
    df = df.with_columns(pl.lit(year).cast(pl.Int64).alias("year"))
    for c in CANONICAL:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c))

    return df.select(CANONICAL)


def build_imls_frame() -> pl.DataFrame:
    print("IMLS · downloading + harmonising 1992-2023 (California only)")
    frames: list[pl.DataFrame] = []
    for year, url, hint in IMLS_YEARS:
        d = load_imls_year(year, url, hint)
        if d is None or d.is_empty():
            continue
        frames.append(d)
        print(f"  ✓ {year}: {len(d):>4} CA library systems")

    # diagonal_relaxed tolerates per-year column/type drift before we cast.
    big = pl.concat(frames, how="diagonal_relaxed")

    # Numeric coerce, then send IMLS negative sentinels to null.
    big = big.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in IMLS_NUMERIC]
    ).with_columns(
        [
            pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in IMLS_NUMERIC
        ]
    ).with_columns(
        [pl.col(c).cast(pl.Int64, strict=False) for c in ["fipsst", "fipsco"]]
    )

    print(f"  → imls_admin: {len(big):,} rows, {big['year'].n_unique()} years")
    return big


# ----------------------------------------------------------------------------
# Microsoft broadband usage (county, 2020)
# ----------------------------------------------------------------------------

MS_BB_URL = (
    "https://raw.githubusercontent.com/microsoft/USBroadbandUsagePercentages/"
    "master/dataset/broadband_data_2020October.csv"
)


def build_broadband_frame() -> pl.DataFrame:
    print("Microsoft · downloading broadband usage by county")
    r = requests.get(MS_BB_URL, headers=UA, timeout=60)
    r.raise_for_status()
    df = pl.read_csv(io.BytesIO(r.content), infer_schema_length=0)
    df = df.rename({c: c.strip().lower().replace(" ", "_") for c in df.columns})
    # ST, COUNTY ID, COUNTY NAME, BROADBAND AVAILABILITY PER FCC, BROADBAND USAGE
    df = df.rename({
        "st": "stabr",
        "county_id": "fips",
        "county_name": "county_name",
        "broadband_availability_per_fcc": "bb_availability",
        "broadband_usage": "bb_usage",
    })
    df = df.filter(pl.col("stabr") == "CA").with_columns(
        pl.col("fips").cast(pl.Int64, strict=False),
        pl.col("bb_availability").cast(pl.Float64, strict=False),
        pl.col("bb_usage").cast(pl.Float64, strict=False),
    ).select(["stabr", "fips", "county_name", "bb_availability", "bb_usage"])
    print(f"  → ms_broadband: {len(df)} CA counties")
    return df


# ----------------------------------------------------------------------------
# NOAA daily summaries — three CA stations, summer 2023
# ----------------------------------------------------------------------------

NOAA_STATIONS = [
    ("USW00023232", "Sacramento Exec AP"),
    ("USW00093193", "Fresno Yosemite AP"),
    ("USW00093134", "Los Angeles Downtown"),
]


def fetch_noaa(station: str, start: str, end: str) -> pl.DataFrame:
    url = (
        "https://www.ncei.noaa.gov/access/services/data/v1"
        f"?dataset=daily-summaries&stations={station}"
        f"&startDate={start}&endDate={end}"
        "&dataTypes=TMAX&format=json&units=standard"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pl.DataFrame(r.json())


def build_noaa_frame() -> pl.DataFrame:
    print("NOAA · daily TMAX for 3 CA stations, Apr-Oct 2023")
    frames = []
    for sid, label in NOAA_STATIONS:
        d = fetch_noaa(sid, "2023-04-01", "2023-10-31")
        d = d.with_columns(pl.lit(label).alias("station_label"))
        frames.append(d)
        time.sleep(0.4)
        print(f"  ✓ {label}: {len(d)} days")

    big = pl.concat(frames, how="diagonal_relaxed").rename(
        {"DATE": "date", "STATION": "station", "TMAX": "temp_f"}
    )
    # Match the historical schema: 'YYYY-MM-DD 00:00:00' date string, integer °F.
    big = big.with_columns(
        (pl.col("date").str.slice(0, 10) + pl.lit(" 00:00:00")).alias("date"),
        pl.col("temp_f").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
    ).select(["date", "station", "temp_f", "station_label"])
    print(f"  → noaa_daily: {len(big):,} rows across {len(NOAA_STATIONS)} stations")
    return big


# ----------------------------------------------------------------------------
# Derived rollups — small, demo-shaped tables built on top of the raw three.
# Keep the originals around so the audience can drop down to them live.
# ----------------------------------------------------------------------------

def build_derived(conn: sqlite3.Connection) -> None:
    print("Derived · pre-baking tidy demo tables")

    # 1) ca_annual : one row per year, the headline CA totals.
    #    This is what Phase 1 / 2 / 5 chart directly.
    conn.executescript("""
        DROP TABLE IF EXISTS ca_annual;
        CREATE TABLE ca_annual AS
        SELECT
          year,
          COUNT(*)                            AS systems,
          SUM(popu_lsa)                       AS popu_lsa,
          SUM(visits)                         AS visits,
          SUM(totcir)                         AS checkouts,
          SUM(wifisess)                       AS wifi_sessions,
          SUM(gpterms)                        AS public_terminals,
          SUM(pitusr)                         AS terminal_users
        FROM imls_admin
        GROUP BY year
        ORDER BY year;
    """)

    # 2) ca_county_recent : one row per CA county for the most recent year
    #    that has wifi data (2023), joined to Microsoft's broadband-usage
    #    telemetry.  Drives Phase 3's scatter.
    #
    #    IMLS reports county as a NAME ('ALAMEDA') and its FIPS columns
    #    drift across years (FIPSCO → LSAGEOID).  Microsoft uses the FIPS
    #    code and full name ('Alameda County').  Easiest stable join:
    #    upper-case both sides and strip the trailing ' County'.
    conn.executescript("""
        DROP TABLE IF EXISTS ca_county_recent;
        CREATE TABLE ca_county_recent AS
        WITH lib AS (
          SELECT
            UPPER(TRIM(county))               AS county_key,
            SUM(popu_lsa)                     AS popu_lsa,
            SUM(visits)                       AS visits,
            SUM(wifisess)                     AS wifi_sessions,
            SUM(gpterms)                      AS public_terminals
          FROM imls_admin
          WHERE year = 2023 AND county IS NOT NULL
          GROUP BY county_key
        )
        SELECT
          b.fips,
          b.county_name,
          b.bb_availability,                   -- what FCC says is *possible*
          b.bb_usage,                          -- what Microsoft observes *actually*
          lib.popu_lsa,
          lib.visits,
          lib.wifi_sessions,
          lib.public_terminals,
          ROUND(lib.wifi_sessions * 1.0
                / NULLIF(lib.popu_lsa, 0) * 1000, 1) AS wifi_per_1k
        FROM ms_broadband b
        LEFT JOIN lib
          ON UPPER(REPLACE(b.county_name, ' County', '')) = lib.county_key
        ORDER BY b.bb_usage;
    """)

    # Sanity readback.
    n1 = conn.execute("SELECT COUNT(*) FROM ca_annual").fetchone()[0]
    n2 = conn.execute("SELECT COUNT(*) FROM ca_county_recent").fetchone()[0]
    print(f"  → ca_annual:        {n1} year-rows")
    print(f"  → ca_county_recent: {n2} county-rows")


# ----------------------------------------------------------------------------
# go
# ----------------------------------------------------------------------------

def main() -> int:
    # Build into a temp DB and swap on success — a failed/offline run must
    # never leave the live libbee.db half-written.
    tmp_path = DB_DIR / "libbee.build.db"
    if tmp_path.exists():
        tmp_path.unlink()
    tmp_uri = f"sqlite:///{tmp_path}"

    # Raw tables — Polars frames written through the ADBC SQLite engine.
    imls = build_imls_frame()
    imls.write_database("imls_admin", tmp_uri, engine="adbc", if_table_exists="replace")
    broadband = build_broadband_frame()
    broadband.write_database("ms_broadband", tmp_uri, engine="adbc", if_table_exists="replace")
    noaa = build_noaa_frame()
    noaa.write_database("noaa_daily", tmp_uri, engine="adbc", if_table_exists="replace")

    # Derived tables — pure SQL over the raw three.
    conn = sqlite3.connect(tmp_path)
    try:
        build_derived(conn)
        conn.commit()
        for table in ["imls_admin", "ms_broadband", "noaa_daily",
                      "ca_annual", "ca_county_recent"]:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  ▢ {table:>14}: {n:,} rows")
    finally:
        conn.close()

    # Atomic-ish swap: only now do we replace the live database.
    if DB_PATH.exists():
        DB_PATH.unlink()
    tmp_path.rename(DB_PATH)
    print(f"\nwrote {DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
