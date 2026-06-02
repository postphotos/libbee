"""
build_state_panel.py — state × year library panel (1992–2023), for the over-time views.

Reuses the harmonised national per-library loader (build_metro_panel.load_year_normalized,
which already reconciles 32 years of IMLS schemas) and rolls it up to one row per state per
year — visits / checkouts / WiFi / hours / funding, totals and per-capita, plus cost-per-visit.
This is what powers "CA + MA vs AL + WY *over time*" in the explorer.

Run:  uv run python build_state_panel.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from libbee.io.paths import ROOT as _REPO

import polars as pl

from . import build_db as bd
from . import build_metro_panel as mp

DB_PATH = _REPO / "data" / "db" / "libbee.db"
DB_URI = f"sqlite:///{DB_PATH}"


def main() -> int:
    yrs = {y: (u, h) for y, u, h in bd.IMLS_YEARS}
    frames = []
    for _y in sorted(yrs):
        _u, _h = yrs[_y]
        df = mp.load_year_normalized(_y, _u, _h)  # national per-library, harmonised
        agg = (df.filter(pl.col("stabr").is_not_null() & (pl.col("popu_lsa") > 0))
               .group_by("stabr", "year").agg(
                   pl.col("popu_lsa").sum(),
                   pl.col("visits").sum(), pl.col("checkouts").sum(),
                   pl.when(pl.col("wifi").count() > 0).then(pl.col("wifi").sum()).otherwise(None).alias("wifi"),
                   pl.col("hours").sum(), pl.col("staff").sum(), pl.col("funding").sum())
               .with_columns(
                   (pl.col("visits") / pl.col("popu_lsa")).alias("visits_pc"),
                   (pl.col("checkouts") / pl.col("popu_lsa")).alias("checkouts_pc"),
                   (pl.col("wifi") / pl.col("popu_lsa")).alias("wifi_pc"),
                   (pl.col("hours") / pl.col("popu_lsa")).alias("hours_pc"),
                   (pl.col("funding") / pl.col("popu_lsa")).alias("funding_pc"),
                   (pl.col("funding") / pl.col("visits")).alias("cost_per_visit")))
        frames.append(agg.rename({"stabr": "st"}))
        print(f"  {_y}: {agg.height} states")
    state_panel = pl.concat(frames, how="vertical_relaxed").sort("st", "year")
    state_panel.write_database("state_panel", DB_URI, engine="adbc", if_table_exists="replace")

    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM state_panel").fetchone()[0]
    ny = conn.execute("SELECT COUNT(DISTINCT year) FROM state_panel").fetchone()[0]
    ns = conn.execute("SELECT COUNT(DISTINCT st) FROM state_panel").fetchone()[0]
    conn.close()
    print(f"\n  → state_panel: {n} rows · {ns} states × {ny} years")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
