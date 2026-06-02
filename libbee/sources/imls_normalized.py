"""Shared national IMLS per-library loader with schema harmonization and in-process caching."""

from __future__ import annotations

import io
import zipfile
from functools import lru_cache

import polars as pl

from . import build_db as b

# Metrics we roll up per library-year. First present source column wins.
METRIC_SRC = {
    "popu_lsa": ["POPU_LSA", "POPU"],
    "visits": ["VISITS"],
    "checkouts": ["TOTCIR"],
    "wifi": ["WIFISESS"],
    "hours": ["HRS_OPEN"],
    "staff": ["TOTSTAFF", "TOTPEMP"],
    "funding": ["TOTINCM", "TOTOPEXP"],
    "programs": ["TOTPRO"],
    "branches": ["BRANLIB"],
    "central": ["CENTLIB"],
}


@lru_cache(maxsize=None)
def _load_year_normalized_cached(year: int, url: str, hint: str) -> pl.DataFrame:
    zpath = b.fetch_zip(year, url)
    with zipfile.ZipFile(zpath) as zf:
        inner = b._find_ae_csv(zf, hint)
        raw = zf.open(inner).read()
    df = pl.read_csv(
        io.BytesIO(raw.decode("latin-1").encode("utf-8")),
        infer_schema_length=0,
        null_values=b.IMLS_NULLS,
        truncate_ragged_lines=True,
    )
    if "STABR" not in df.columns:
        df = df.rename({c: c.upper() for c in df.columns})
    cols = df.columns

    st_col = "FIPSST" if "FIPSST" in cols else ("INCITSST" if "INCITSST" in cols else None)
    co_col = "FIPSCO" if "FIPSCO" in cols else ("INCITSCO" if "INCITSCO" in cols else None)
    cbsa_expr = pl.col("CBSA") if "CBSA" in cols else pl.lit(None, dtype=pl.Utf8)
    st_expr = pl.col(st_col).cast(pl.Int64, strict=False) if st_col else pl.lit(None, dtype=pl.Int64)
    co_expr = pl.col(co_col).cast(pl.Int64, strict=False) if co_col else pl.lit(None, dtype=pl.Int64)
    fscs_col = next((c for c in ["FSCSKEY", "FSCSKey", "FSCS_KEY"] if c in cols), None)
    fscs_expr = pl.col(fscs_col) if fscs_col else pl.lit(None, dtype=pl.Utf8)

    def _metric_expr(srcs: list[str]) -> pl.Expr:
        present = [pl.col(src).cast(pl.Float64, strict=False) for src in srcs if src in cols]
        return pl.coalesce(present) if present else pl.lit(None, dtype=pl.Float64)

    out = df.select(
        pl.lit(year).cast(pl.Int64).alias("year"),
        fscs_expr.alias("fscskey"),
        pl.col("STABR").alias("stabr"),
        pl.col("CNTY").str.strip_chars().str.to_uppercase().alias("cnty_key"),
        cbsa_expr.alias("cbsa_raw"),
        st_expr.alias("st"),
        co_expr.alias("co"),
        *[_metric_expr(srcs).alias(target) for target, srcs in METRIC_SRC.items()],
    ).with_columns(
        pl.when(pl.col("st").is_not_null() & pl.col("co").is_not_null())
        .then(pl.col("st").cast(pl.Utf8) + "-" + pl.col("co").cast(pl.Utf8))
        .otherwise(None)
        .alias("stco_key")
    )
    return out.with_columns(
        [pl.when(pl.col(metric) < 0).then(None).otherwise(pl.col(metric)).alias(metric) for metric in METRIC_SRC]
    )


def load_year_normalized(year: int, url: str, hint: str) -> pl.DataFrame:
    """Return a harmonized national IMLS per-library frame for one year."""
    return _load_year_normalized_cached(year, url, hint).clone()