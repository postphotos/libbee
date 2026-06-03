"""One caching download helper — so every adapter fetches reproducibly the same way.

A source is fetched once into data/raw/ and never re-fetched; that cached byte-for-byte input is
what the MANIFEST checksums, so a rebuild can't drift even if the upstream URL changes.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from .paths import RAW

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def cached(url: str, name: str, *, browser_ua: bool = False, min_size: int = 1) -> Path:
    """Download `url` to data/raw/`name` once and return the path (cached thereafter)."""
    dest = RAW / name
    if dest.exists() and dest.stat().st_size >= min_size:
        print(f"    ↳ {name} (cached)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    ↓ {name}")
    headers = {"User-Agent": UA} if browser_ua else {}
    req = urllib.request.Request(url, headers=headers)
    tmp = dest.with_name(f".{dest.name}.tmp")
    try:
        with urllib.request.urlopen(req, timeout=180) as response, tmp.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        if tmp.stat().st_size < min_size:
            raise ValueError(f"Downloaded file too small: {tmp.stat().st_size} < {min_size}")
        tmp.replace(dest)
        return dest
    finally:
        if tmp.exists():
            tmp.unlink()
