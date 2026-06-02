"""
libbee — US (American) public-library funding, usage, and operations (1992–2023, IMLS) + community
civics metrics (Census, HUD, homelessness) as a small data-product library.

The point (per "make the data invisible"): you should never touch 32 years of IMLS PLS schemas, a
HUD homelessness API, a Census search tool, or a California LibPAS HTML report. You ask one question of one table:

    >>> import libbee
    >>> libbee.facts()                              # the unified long table (every source)
    >>> libbee.load("county_equity")                # any individual frame, by name
    >>> libbee.build()                              # self-contained fetch + build + fingerprint
    >>> libbee.verify()                             # prove the data matches the committed MANIFEST

…and the analysis surface comes with it — pure, frame-in / frame-out, no rendering:

    >>> libbee.analysis.optimal_allocation(...)     # the "cuts": leaderboards, cohorts, allocators
    >>> libbee.geo.states_geojson()                 # FIPS / Census regions / topojson
    >>> libbee.inflation.deflate(df, cols=[...])    # nominal → constant dollars (CPI-U)

Layout — the **core is pure data** (no marimo / altair / playwright):

    libbee.adapters        one self-describing adapter per source (fetch → cache → conform)
    libbee.frames          load the finished frames
    libbee.analysis        the pure analytical "cuts" (100% unit-tested)
    libbee.geo             geography (FIPS, Census regions, topojson decode)
    libbee.inflation       CPI-U deflator (nominal → real dollars)
    libbee.metrics         metric helpers
    libbee.config/schemas  validated settings + metadata-level data contracts (Pydantic)

`facts` conforms every source into `(geo_level, geo_id, geo_name, state, year, metric, value, source)`.
"""

from . import analysis, geo, inflation, metrics
from .adapters import REGISTRY as adapters
from .io import config, frames, schemas
from .io.config import settings
from .io.store import export_all, export_table, load, scan, tables
from .pipeline import build, verify

__all__ = [
    # data product
    "adapters",
    "build",
    "verify",
    "load",
    "scan",
    "tables",
    "export_table",
    "export_all",
    "facts",
    "geo_dim",
    # analytical surface (pure, render-free)
    "analysis",
    "frames",
    "geo",
    "inflation",
    "metrics",
    # validated config + metadata-level data contracts (Pydantic)
    "config",
    "settings",
    "schemas",
]
__version__ = "0.1.0"


def facts():
    """The unified long fact table — every source, one schema."""
    return load("facts")


def geo_dim():
    """The geography dimension (canonical names per geo_id)."""
    return load("geo_dim")
