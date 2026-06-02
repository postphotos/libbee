"""libbee.io — the data-plumbing layer: paths, IO, frames, fetch, fingerprints.

Where bytes meet disk. ``paths`` resolves the data dirs; ``store`` moves data between SQLite, Parquet and
DuckDB; ``frames`` loads the finished demo frames; ``fetch`` is the cached HTTP getter the adapters use;
``manifest`` content-fingerprints the built data for reproducibility. The analytical core
(``libbee.analysis``, ``libbee.geo``, …) sits on top of this and never the other way round.
"""
