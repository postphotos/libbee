"""Where the data lives — organised, not a dump.

    data/
      raw/<source>/   one folder per source (imls, hud, census, ca_libpas, geo) — cached inputs
      db/             libbee.db (SQLite source) + libbee.duckdb (columnar)
      flat/           *.parquet — the conformed, demo-ready frames
      MANIFEST.json   the reproducibility fingerprint

The project root is the nearest ancestor with a pyproject.toml (so it works wherever the project is
checked out, independent of .git). Installed as a wheel: `$LIBBEE_DATA`, else ./data under the cwd.
"""

import os
from pathlib import Path


def data_root(start: "Path | None" = None) -> Path:
    env = os.environ.get("LIBBEE_DATA")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path(__file__)).resolve()
    for p in here.parents:  # the project root owns both pyproject.toml and data/
        if (p / "pyproject.toml").exists():
            return p / "data"
    return Path.cwd() / "data"


DATA = data_root()
ROOT = DATA.parent

# raw inputs, one folder per source
RAW = DATA / "raw"
RAW_IMLS = RAW / "imls"
RAW_HUD = RAW / "hud"
RAW_CENSUS = RAW / "census"
RAW_CALIBPAS = RAW / "ca_libpas"
RAW_GEO = RAW / "geo"

# databases, together
DB_DIR = DATA / "db"
DB = DB_DIR / "libbee.db"
DUCKDB = DB_DIR / "libbee.duckdb"

FLAT = DATA / "flat"
MANIFEST = DATA / "MANIFEST.json"
