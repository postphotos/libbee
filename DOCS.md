# `libbee` Documentation & Cookbooks

This document covers the package architecture, the data dictionary, and practical code examples for policy analysis, budgeting, and modeling.

## 1. Data Dictionary

Call `libbee.tables()` in Python to list all conformed tables. You can query any of these using `libbee.load("<name>")`.

| Frame Name | Grain | Source(s) | Description |
| :--- | :--- | :--- | :--- |
| **`facts`** | Long format | IMLS, HUD, Census, CA-LibPAS | **The unified long table** — every conformed metric aligned to a single schema. |
| **`county_equity`** | County × year (2019) | IMLS, Census | Cross-sectional table joining library metrics to Census indicators. |
| **`county_panel`** | County × year | IMLS, Census | Longitudinal county library metrics and demographics, 1992–2023. |
| **`state_panel`** | State × year | IMLS, Census | Longitudinal state-level rollup and per-capita metrics. |
| **`metro_panel`** | Metro × year | IMLS, HUD | Longitudinal CBSA-level rollups for the 30 largest metros. |
| **`county_homeless`** | County × year | HUD | Homelessness point-in-time (PIT) estimates and HUD grants. |
| **`geo_dim`** | Geographic ID | IMLS, Census | Canonical dimension mapping FIPS and CBSA codes to names. |
| **`metro_lookup`** | Metro (CBSA) | Census | Core-Based Statistical Area (CBSA) mapping. |
| **`ca_annual`** | Library × year | CA State Library | Detailed annual LibPAS stats (California only). |

### The Unified `facts` Schema
The `facts` table melts all sources into a highly compressible, predictable structure:

* `geo_level` (`str`): `state`, `metro`, `county`, or `library`.
* `geo_id` (`str`): FIPS code, CBSA code, or IMLS FSCS library ID.
* `geo_name` (`str`): Human-readable name.
* `state` (`str`): Two-letter US state code.
* `year` (`int`): Observation year.
* `metric` (`str`): The conformed metric identifier (e.g., `visits_pc`).
* `value` (`float`): The actual number.
* `source` (`str`): `IMLS`, `Census`, `HUD`, or `CA-LibPAS`.

---

## 2. Environment Variables

* **`LIBBEE_DATA`**: Directory where Parquet tables are cached. Defaults to `./data` in your CWD.
* **`CENSUS_API_KEY`**: Optional. Get one for free from the [US Census Bureau](https://api.census.gov/data/key_signup.html) to download ACS metrics. It caches locally on the first run, so you only need it once.

---

## 3. Code Cookbooks

### Basic Data Profiling
Use Polars aggregation to quickly audit metrics. This is especially useful for verifying cross-sectional demographics (like Census ACS data) where certain variables may only have a single snapshot year.

```python
import polars as pl
import libbee

# Load the unified facts table
facts = libbee.facts()

# Filter down to specific metrics
df = facts.filter(pl.col("metric").is_in(["density", "median_income"]))

print(f"Total rows: {df.height}")
print(
    df.group_by("metric").agg(
        [
            pl.col("value").min().alias("min"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").max().alias("max"),
            pl.col("year").n_unique().alias("unique_years"),
            pl.col("year").min().alias("year_min"),
            pl.col("year").max().alias("year_max"),
        ]
    )
)
```

### Using DuckDB (SQL Interface)
If you prefer SQL over DataFrames, `libbee` automatically registers its Parquet cache with DuckDB.

```python
import duckdb
import libbee

# Make sure data is built
libbee.build()

conn = duckdb.connect(str(libbee.config.DUCKDB))
res = conn.execute("""
    SELECT year, AVG(value) as avg_visits
    FROM facts
    WHERE metric = 'visits_pc' AND geo_level = 'state'
    GROUP BY year 
    ORDER BY year DESC
""").pl()
```

### Inflation Adjustments
Funding metrics are nominal. Use the built-in BLS CPI-U index to deflate them to constant dollars.

```python
import libbee

df = libbee.load("state_panel")

# Convert 'funding_pc' into constant 2023 dollars
df_real = libbee.inflation.deflate(
    df,
    cols=["funding_pc"],
    year="year",
    base=2023
)
```

### Difference-in-Differences (DiD) Design Matrices
Generate design matrices for policy-impact event studies quickly.

```python
from libbee.analysis import events

df_did = events.did_panel(
    panel=libbee.load("metro_panel"),
    baseline=2014,
    post=2018,
    treat_thresh=-5.0,  # e.g., treatment is a 5% funding cut
    unit="metro"
)
```

### Budget Allocation Modeling
Allocate a fixed national/state budget to counties using greedy optimization based on predicted marginal returns.

```python
import polars as pl
import libbee
from libbee.analysis import greedy_allocation

county_equity = libbee.load("county_equity")

# Setup the marginal cost/benefit per county
marginals = county_equity.select([
    "fips", "NAME", "state",
    (pl.col("popu_lsa") * 5.0).alias("dollars_needed"),  # $5 bump per resident
    (pl.col("popu_lsa") * 0.2).alias("extra_visits"),    # Predicted 0.2 visits bump
])

# Maximize visits with a $50M budget
chosen, summary = greedy_allocation(marginals, budget=50_000_000)
print(f"Funded {summary['n_counties']} counties; gained {summary['extra_visits']:.0f} visits.")
```

### Clustering
Categorize library systems into structural archetypes (e.g., "lean rural", "underfunded/high-need").

```python
import polars as pl
import libbee
from libbee.analysis import cluster_labels

county_equity = libbee.load("county_equity")

profile = county_equity.group_by("cluster").agg(
    pl.col("funding_pc").mean(),
    pl.col("visits_pc").mean(),
    pl.col("poverty").mean(),
    pl.col("density").mean(),
)

labels = cluster_labels(profile)
print(labels)
```

---

## 4. Command Line Interface (CLI)

`libbee` includes a native command line helper for scripting, orchestration, and validation:

```bash
# Build conformed cache (fetches raw if not already cached)
libbee build

# Force rebuild of cached data
libbee build --force

# Verify local Parquet data matches MANIFEST.json (zero-drift check)
libbee verify

# Print available data adapters and their underlying data URLs
libbee adapters

# Export data tables to portable CSV or JSON
libbee export facts --format csv --out ./exports/
libbee export --all --format json --out ./json_dumps/

# Launch the interactive marimo dashboard explorer
# Copies the bundled example notebook locally and launches it
libbee demo
```

---

## 5. Reproducibility & Integrity

To ensure scientific reproducibility, `libbee` caches raw source files byte-for-byte in `data/raw/`. It maps these against a hash manifest (`data/MANIFEST.json`).

Run `libbee verify` (CLI) or `libbee.verify()` (Python) to hash local files and check them against the manifest. If upstream data drifts or a file is corrupted, it will throw an error.

---

## 6. Extending the Package (Custom Adapters)

Want to add your own local datasets into the unified facts schema?

1. Create a subclass of `Adapter` (from `libbee/adapters/base.py`).
2. Define the `name`, `source`, `tables`, and `build` method.
3. Melt your custom columns into the canonical schema inside `to_facts()`.
4. Add your adapter to `REGISTRY` in `libbee/adapters/__init__.py`. 
