"""Reproducibility fingerprint — content hashes of every frame + raw input (no timestamps)."""

from __future__ import annotations

import hashlib
import json

import polars as pl

from .paths import FLAT, MANIFEST, RAW


def _content_hash(df: pl.DataFrame) -> str:
    return hashlib.sha256(df.sort(df.columns).write_csv().encode()).hexdigest()


def _file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute() -> dict:
    tables = {p.stem: {"rows": (df := pl.read_parquet(p)).height, "cols": df.width, "sha256": _content_hash(df)} for p in sorted(FLAT.glob("*.parquet"))}
    raw = {str(p.relative_to(RAW)): {"bytes": p.stat().st_size, "sha256": _file_hash(p)} for p in sorted(RAW.rglob("*")) if p.is_file()}
    return {"tables": tables, "raw_inputs": raw}


def build() -> dict:
    man = compute()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(f"  → MANIFEST.json: {len(man['tables'])} tables, {len(man['raw_inputs'])} raw inputs")
    return man


def verify() -> bool:
    if not MANIFEST.exists():
        print("no MANIFEST.json — run libbee.build() first")
        return False
    saved, cur = json.loads(MANIFEST.read_text()), compute()
    ok = True
    for name, meta in saved.get("tables", {}).items():
        got = cur["tables"].get(name)
        if got is None or got["sha256"] != meta["sha256"]:
            print(f"  ✗ {name}: {'missing' if got is None else 'content changed'}")
            ok = False
    print("✅ data matches MANIFEST (reproducible)" if ok else "⚠️  data differs from MANIFEST")
    return ok
