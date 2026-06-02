"""Orchestration — the self-contained fetch → conform → unify → fingerprint pipeline."""

from __future__ import annotations

from ..adapters import REGISTRY, order_adapters
from ..io import manifest
from ..io.store import flatten
from . import unify


def build(force: bool = False, verbose: bool = True) -> None:
    """Build everything that's missing (or all, if force=True): fetch raw (cached) → conform →
    flatten to Parquet/DuckDB → melt into `facts` → write the reproducibility MANIFEST."""
    ordered = order_adapters(REGISTRY, attr="build_after")
    for a in ordered:
        if force or not a.present():
            if verbose:
                print(f"· {a.name} ({a.source}) — {a.provenance}")
            a.build()
    src_tables = [t for a in ordered for t in a.tables]
    flatten(src_tables)
    unify.build()
    flatten(["facts", "geo_dim"])
    manifest.build()


def verify() -> bool:
    """Prove the current data matches the committed MANIFEST (reproducible / no drift)."""
    return manifest.verify()
