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
    if verbose:
        print("\n📊 Building libbee data product...\n")
    for a in ordered:
        if force or not a.present():
            if verbose:
                print(f"  ▸ {a.name} ({a.source})")
                print(f"    {a.provenance}")
            a.build()
    if verbose:
        print("\n  📦 Flattening to Parquet + DuckDB...")
    src_tables = [t for a in ordered if a.present() for t in a.tables]
    flatten(src_tables)
    if verbose:
        print("  🔗 Unifying into facts table...")
    unify.build()
    if verbose:
        print("  📊 Creating dimension tables...")
    flatten(["facts", "geo_dim"])
    if verbose:
        print("  ✓ Writing MANIFEST...")
    manifest.build()
    if verbose:
        print("\n✅ Build complete!\n")


def verify() -> bool:
    """Prove the current data matches the committed MANIFEST (reproducible / no drift)."""
    return manifest.verify()
