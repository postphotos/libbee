# 🐝 libbee

[![PyPI Version](https://img.shields.io/pypi/v/libbee.svg)](https://pypi.org/project/libbee/)
[![Python Version](https://img.shields.io/pypi/pyversions/libbee.svg)](https://pypi.org/project/libbee/)
[![License](https://img.shields.io/pypi/l/libbee.svg)](https://github.com/postphotos/libbee/blob/master/LICENSE)
[![Ruff Style](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25-green.svg)](https://github.com/postphotos/libbee)

An unofficial Python package for US public library and community civics data.

`libbee` consolidates over 30 years of public data into a single, unified, cleanly structured `facts` table. It pulls from four primary data sources:
* **IMLS (Institute of Museum and Library Services)**: The authoritative Public Libraries Survey (PLS) tracking funding, visits, staff, and circulation for every library system in America (1992–2023).
* **US Census Bureau (ACS)**: The American Community Survey 5-Year estimates, supplying cross-sectional community need metrics like median household income, poverty rates, and broadband access.
* **HUD (Dept. of Housing and Urban Development)**: Point-in-Time (PIT) estimates and Continuum of Care (CoC) funding awards, providing local homelessness statistics.
* **California State Library (LibPAS)**: Highly detailed annual library statistics and budget allocations specifically for California library systems.

## Why does this exist?
Working directly with the raw source data for public libraries is notoriously painful. If you try to do this from scratch, you have to deal with:
* **Shifting schemas:** Over 32 years, column names shift constantly (e.g., `HOURS` to `OP_HRS` to `HRS_OPEN`).
* **Silent data corruption:** Missing fields use arbitrary negative numeric codes (`-1`, `-3`, `-9`) instead of standard nulls. If loaded directly, these ruin your sums and averages.
* **Geospatial nightmares:** Joining library panel data against Census or HUD datasets requires cross-walking mismatched boundaries.
* **Formatting mess:** The raw data is scattered across 100+ MB of ASCII, Windows-1252, and UTF-8 encoded ZIPs, DBFs, and binary Excel files.

`libbee` automates the downloading, cleaning, and conforming. It maps the entities, fixes the nulls, normalizes the encodings, and packs everything into fast, columnar Parquet tables (via Polars/PyArrow) that take up about 11 MB on disk.

## Install

```bash
pip install libbee                 # Core data loading
pip install "libbee[analysis]"     # Adds scikit-learn, statsmodels, shap
pip install "libbee[duckdb]"       # Adds DuckDB SQL engine
pip install "libbee[notebook]"     # Adds interactive marimo dashboard
pip install "libbee[all]"          # Everything
```
*Note: Python ≥ 3.11 is required. The package ships code-only; data is built and cached locally.*

## Quickstart

Because we don't ship the raw data in the wheel, you need to build the local cache once:

```bash
# Downloads, cleans, and caches everything (takes a minute or two)
libbee build
```

Then, use it in Python:

```python
import libbee
import polars as pl

# 1. Load the unified facts table (720k+ rows, conformed to one schema)
df_facts = libbee.facts()

# 2. Or query a specific conformed frame
df_equity = libbee.load("county_equity")

# 3. Use lazy evaluation for fast filtered queries
lf = libbee.scan("facts")
df_summary = (
    lf.filter(
        (pl.col("geo_level") == "state") &
        (pl.col("metric") == "visits_pc") &
        (pl.col("year") == 2019)
    )
    .sort("value", descending=True)
    .collect()
)
```

## Documentation & Advanced Usage
Looking for the Data Dictionary, DuckDB integration, inflation adjustments, or analysis examples (like Difference-in-Differences models)?

👉 **[Read the Docs](DOCS.md)**

## Citation & License
When using these conformed datasets in publications, please cite the primary publishing agencies: IMLS (Public Libraries Survey), US Census Bureau (ACS), HUD (PIT/CoC), and the CA State Library.

Licensed under the MIT License.