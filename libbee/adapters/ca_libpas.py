"""
CA-LibPAS adapter — California State Library statistics (countingopinions.com / LibPAS).

Richer and more recent than the national IMLS survey: the FY2023-24 **Outlet Summary** report, with
per-outlet rows carrying **library names** (Jurisdiction + City) — so it extends California a year
past IMLS and unlocks true city/library-level drill-downs. Self-contained: no public API, but the
public report.php page embeds the data as an HTML table; we cache it once and parse it.

This is the worked example for adding a dataset — fetch (cached) in build(), conform to a source
table, then declare the facts contribution in to_facts(). Nothing else in the package changes.
"""

from __future__ import annotations

import polars as pl

from ..io.fetch import cached
from ..io.store import load, write_table
from .base import Adapter, melt

REPORT_URL = "https://ca.countingopinions.com/pireports/report.php?0519b8c3b57c6335455172eb40aaaf96&live"  # public "Outlet Summary Report", FY2023-24
FY_END = 2024  # FY 2023-24 → IMLS-style end year

_COLS = {
    "FSCSKey": "fscskey",
    "Jurisdiction Name": "jurisdiction",
    "City": "city",
    "Total Outlet Staff FTE": "staff",
    "Actual Hours Open, Annually": "hours",
    "Population Served": "pop",
    "Total outlet operating expenditure": "funding",
    "Volumes Held": "volumes",
    "Physical Circulation": "circulation",
    "Number of Internet Terminals - General Public": "terminals",
}
_NUMERIC = ["staff", "hours", "pop", "funding", "volumes", "circulation", "terminals"]


class CALibPAS(Adapter):
    name, source = "ca_libpas", "CA-LibPAS"
    tables = ("ca_libraries",)
    provenance = "California State Library LibPAS — FY2023-24 outlet summary, jurisdiction-level, with names"

    def build(self) -> None:
        import io

        import pandas as pd  # read_html (needs lxml)

        html = cached(REPORT_URL, "ca_libpas/ca_libpas_outlet_FY2023-24.html", browser_ua=True, min_size=100_000).read_text(errors="ignore")
        raw = pl.from_pandas(pd.read_html(io.StringIO(html))[0])
        keep = {k: v for k, v in _COLS.items() if k in raw.columns}
        df = raw.select([pl.col(k).alias(v) for k, v in keep.items()])
        df = df.with_columns(
            [  # cells arrive as "1,234" / "$5,678"
                pl.col(c).cast(pl.Utf8).str.replace_all(r"[^0-9.\-]", "").replace("", None).cast(pl.Float64, strict=False).alias(c) for c in _NUMERIC if c in df.columns
            ]
        ).filter(pl.col("jurisdiction").is_not_null())
        # Population Served is a jurisdiction attribute repeated per outlet → max, not sum.
        lib = (
            df.group_by("jurisdiction")
            .agg(
                pl.col("city").first(),
                pl.col("pop").max(),
                pl.col("staff").sum(),
                pl.col("hours").sum(),
                pl.col("funding").sum(),
                pl.col("volumes").sum(),
                pl.col("circulation").sum(),
                pl.col("terminals").sum(),
                pl.len().alias("outlets"),
            )
            .filter(pl.col("pop") > 0)
            .with_columns(
                (pl.col("circulation") / pl.col("pop")).alias("checkouts_pc"),
                (pl.col("funding") / pl.col("pop")).alias("funding_pc"),
                (pl.col("staff") / pl.col("pop")).alias("staff_pc"),
                (pl.col("hours") / pl.col("pop")).alias("hours_pc"),
                (pl.col("volumes") / pl.col("pop")).alias("collection_pc"),
                pl.lit(FY_END).alias("year"),
                pl.lit("CA").alias("st"),
            )
            .sort("jurisdiction")
        )
        write_table(lib, "ca_libraries")
        print(f"  → ca_libraries: {lib.height} CA library jurisdictions (FY2023-24, with names)")

    def to_facts(self) -> pl.DataFrame:
        d = load("ca_libraries").with_columns(pl.col("jurisdiction").alias("name"))
        return melt(d, "library", "jurisdiction", "name", self.source, year="year", state="st")
