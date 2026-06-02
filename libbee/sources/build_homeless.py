"""
build_homeless.py — HUD Point-in-Time (PIT) homelessness, for the libraries story.

Source: HUD's "2007-2024 PIT Counts by CoC" workbook (one sheet per year, 389 CoCs).
HUD's huduser.gov hides behind an Akamai bot wall that 202s a plain client; a browser
User-Agent gets the real file. We download it once to data/raw/ and never refetch.

Homelessness is reported by Continuum of Care (CoC), not county. Two products:

  • coc_homeless   — every CoC × year × overall homeless count + HUD's own urban/rural
                     category. Clean, no mapping assumptions. Good for the national
                     "urban vs rural" cut.
  • metro_homeless — the 30 biggest metros × year, summing the principal *urban-core*
                     CoCs that sit inside each metro (hand-mapped below). Lets us lay
                     homelessness next to the metro library panel (hours, visits) and
                     ask: do they move together? where did cut hours overlap with
                     rising homelessness?

The metro→CoC map is the urban core only (Major City + same-metro county CoCs), not
the entire CBSA — so treat the count as "the metro's core CoCs," a consistent series
for co-movement, not a Census-exact metro total.

Run:  uv run --with fastexcel python build_homeless.py
"""

from __future__ import annotations

import sqlite3
import urllib.request
from pathlib import Path

from libbee.io.paths import ROOT as _REPO

import polars as pl

ROOT = _REPO
DB_PATH = ROOT / "data" / "db" / "libbee.db"
DB_URI = f"sqlite:///{DB_PATH}"
RAW = ROOT / "data" / "raw" / "hud" / "pit_coc.xlsb"
RAW_AWARDS = ROOT / "data" / "raw" / "hud" / "coc_awards.xlsx"
PIT_URL = "https://www.huduser.gov/portal/sites/default/files/xls/2007-2024-PIT-Counts-by-CoC.xlsb"
AWARDS_URL = ("https://files.hudexchange.info/resources/documents/"
              "CoC-Program-Awards-by-Component-and-Project-Type.xlsx")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# Each metro (by CBSA, matching metro_lookup) → the urban-core CoCs inside it.
# Kept to the principal city + same-metro county CoCs for a consistent co-movement series.
METRO_COC = {
    "35620": ["NY-600", "NY-603", "NY-604", "NY-606", "NJ-504", "NJ-506", "NJ-511"],  # NYC
    "31080": ["CA-600", "CA-606", "CA-607", "CA-612", "CA-602"],                       # LA
    "16980": ["IL-510", "IL-511", "IL-514", "IL-517", "IL-502", "IL-500", "IL-506"],   # Chicago
    "19100": ["TX-600", "TX-601"],                                                     # Dallas-FW
    "26420": ["TX-700"],                                                               # Houston
    "12060": ["GA-500", "GA-502", "GA-506", "GA-508"],                                 # Atlanta
    "47900": ["DC-500", "VA-600", "VA-601", "VA-603", "VA-604", "VA-602", "MD-600", "MD-601"],  # DC
    "33100": ["FL-600", "FL-601", "FL-605"],                                           # Miami
    "37980": ["PA-500", "PA-502", "PA-504", "PA-505", "PA-511", "NJ-503", "NJ-502"],   # Philadelphia
    "38060": ["AZ-502"],                                                               # Phoenix
    "40140": ["CA-608", "CA-609"],                                                     # Riverside-SB
    "14460": ["MA-500", "MA-509", "MA-502", "MA-511"],                                 # Boston
    "41860": ["CA-501", "CA-502", "CA-505", "CA-507", "CA-512"],                       # SF-Oakland
    "19820": ["MI-501", "MI-502", "MI-503", "MI-504"],                                 # Detroit
    "42660": ["WA-500", "WA-503", "WA-504"],                                           # Seattle
    "33460": ["MN-500", "MN-501", "MN-503"],                                           # Minneapolis-SP
    "41740": ["CA-601"],                                                               # San Diego
    "45300": ["FL-501", "FL-502", "FL-519"],                                           # Tampa
    "19740": ["CO-503"],                                                               # Denver
    "41700": ["TX-500"],                                                               # San Antonio
    "41180": ["MO-500", "MO-501", "IL-508"],                                           # St. Louis
    "12580": ["MD-501", "MD-505", "MD-503", "MD-504", "MD-506"],                       # Baltimore
    "36740": ["FL-507"],                                                               # Orlando
    "16740": ["NC-505", "NC-509"],                                                     # Charlotte
    "29820": ["NV-500"],                                                               # Las Vegas
    "38900": ["OR-501", "OR-506", "OR-507", "WA-508"],                                 # Portland
    "12420": ["TX-503"],                                                               # Austin
    "40900": ["CA-503", "CA-515", "CA-521", "CA-525"],                                 # Sacramento
    "18140": ["OH-503"],                                                               # Columbus
    "38300": ["PA-600", "PA-603"],                                                     # Pittsburgh
}


def _download(url: str, dest: Path, label: str) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"{label} · downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})  # browser UA beats Akamai 202
    data = urllib.request.urlopen(req, timeout=180).read()
    dest.write_bytes(data)
    print(f"  → {len(data):,} bytes")


def download_pit() -> None:
    _download(PIT_URL, RAW, "PIT")


def load_spend_long() -> pl.DataFrame:
    """Project-level CoC awards (FY2018–2024) → (coc_number, year, spend) totals per CoC-year."""
    import fastexcel

    wb = fastexcel.read_excel(str(RAW_AWARDS))
    years = [s for s in wb.sheet_names if s.isdigit()]
    frames = []
    for y in years:
        df = pl.read_excel(RAW_AWARDS, sheet_name=y, engine="calamine")
        amt = next(c for c in df.columns if "award" in c.lower() and "amount" in c.lower())
        keep = (df.select(pl.col("CoC Number").str.strip_chars().alias("coc_number"),
                          pl.col(amt).cast(pl.Float64, strict=False).alias("spend"))
                .filter(pl.col("coc_number").str.len_chars() > 0)
                .group_by("coc_number").agg(pl.col("spend").sum())
                .with_columns(pl.lit(int(y)).alias("year")))
        frames.append(keep)
    return pl.concat(frames).select("coc_number", "year", "spend").sort("coc_number", "year")


def build_metro_spend(spend: pl.DataFrame) -> pl.DataFrame:
    rows = [pl.DataFrame({"cbsa": [c] * len(v), "coc_number": v}) for c, v in METRO_COC.items()]
    metro_map = pl.concat(rows)
    return (metro_map.join(spend, on="coc_number", how="inner")
            .group_by("cbsa", "year")
            .agg(pl.col("spend").sum(), pl.col("coc_number").n_unique().alias("n_cocs"))
            .sort("cbsa", "year"))


def load_pit_long() -> pl.DataFrame:
    """All year sheets → tidy (coc_number, coc_name, category, year, homeless)."""
    import fastexcel

    wb = fastexcel.read_excel(str(RAW))
    years = [s for s in wb.sheet_names if s.isdigit()]
    frames = []
    for y in years:
        df = pl.read_excel(RAW, sheet_name=y, engine="calamine")
        cols = df.columns
        homeless_col = next(c for c in cols if c.strip().lower() == "overall homeless")
        keep = df.select(
            pl.col("CoC Number").str.strip_chars().alias("coc_number"),
            pl.col("CoC Name").str.strip_chars().alias("coc_name"),
            (pl.col("CoC Category").str.strip_chars() if "CoC Category" in cols
             else pl.lit(None)).alias("category"),
            pl.col(homeless_col).cast(pl.Float64, strict=False).alias("homeless"),
        ).filter(pl.col("coc_number").str.len_chars() > 0)
        keep = keep.with_columns(pl.lit(int(y)).alias("year"))
        frames.append(keep)
    long = pl.concat(frames, how="vertical_relaxed")
    # Category isn't on every historical sheet; backfill from the most recent year per CoC.
    latest_cat = (long.filter(pl.col("category").is_not_null())
                  .sort("year").group_by("coc_number").agg(pl.col("category").last()))
    long = (long.drop("category").join(latest_cat, on="coc_number", how="left"))
    return long.select("coc_number", "coc_name", "category", "year", "homeless").sort("coc_number", "year")


def build_metro(long: pl.DataFrame) -> pl.DataFrame:
    """Sum each metro's urban-core CoCs per year → metro_homeless."""
    have = set(long["coc_number"].unique().to_list())
    rows = []
    for cbsa, cocs in METRO_COC.items():
        missing = [c for c in cocs if c not in have]
        if missing:
            print(f"  ! cbsa {cbsa}: CoCs not in PIT data: {missing}")
        rows.append(pl.DataFrame({"cbsa": [cbsa] * len(cocs), "coc_number": cocs}))
    metro_map = pl.concat(rows)
    joined = metro_map.join(long, on="coc_number", how="inner")
    metro_homeless = (
        joined.group_by("cbsa", "year")
        .agg(pl.col("homeless").sum().alias("homeless"),
             pl.col("coc_number").n_unique().alias("n_cocs"))
        .sort("cbsa", "year")
    )
    return metro_homeless


def main() -> int:
    download_pit()
    _download(AWARDS_URL, RAW_AWARDS, "Awards")
    long = load_pit_long()
    print(f"PIT · {long['coc_number'].n_unique()} CoCs × {long['year'].n_unique()} years "
          f"({long['year'].min()}–{long['year'].max()})")
    metro = build_metro(long)
    print(f"metro_homeless · {metro['cbsa'].n_unique()} metros, {metro.height} metro-years")

    spend = load_spend_long()
    print(f"coc_spend · {spend['coc_number'].n_unique()} CoCs × {spend['year'].n_unique()} years "
          f"({spend['year'].min()}–{spend['year'].max()}), ${float(spend['spend'].sum())/1e9:.1f}B total")
    metro_spend = build_metro_spend(spend)
    print(f"metro_spend · {metro_spend['cbsa'].n_unique()} metros, {metro_spend.height} metro-years")

    long.write_database("coc_homeless", DB_URI, engine="adbc", if_table_exists="replace")
    metro.write_database("metro_homeless", DB_URI, engine="adbc", if_table_exists="replace")
    spend.write_database("coc_spend", DB_URI, engine="adbc", if_table_exists="replace")
    metro_spend.write_database("metro_spend", DB_URI, engine="adbc", if_table_exists="replace")
    conn = sqlite3.connect(DB_PATH)
    for t in ("coc_homeless", "metro_homeless", "coc_spend", "metro_spend"):
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  → {t}: {n} rows")
    conn.close()
    print(f"wrote into {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
