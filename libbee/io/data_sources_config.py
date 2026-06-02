"""Temporal configuration for data sources.

This module centralizes year ranges, download URLs, and filenames for all data sources.
When new years become available, update the configurations here rather than editing build scripts.

Usage::

    from libbee.io.data_sources_config import IMLS_YEARS
    for year, url, filename in IMLS_YEARS:
        print(f"IMLS {year}: {url}")
"""

from __future__ import annotations

# Public Libraries Survey (IMLS) — (year, download_url, filename_in_zip)
# Source: https://www.imls.gov/research-tools/data-collection/public-libraries-survey/explore-pls-data
IMLS_YEARS: list[tuple[int, str, str]] = [
    (1992, "https://www.imls.gov/sites/default/files/pupld92a_csv.zip", "PUPLDF92.csv"),
    (1993, "https://www.imls.gov/sites/default/files/pupld93a_csv.zip", "PUPLDF93.csv"),
    (1994, "https://www.imls.gov/sites/default/files/pupld94a_csv.zip", "PUPLDF94.csv"),
    (1995, "https://www.imls.gov/sites/default/files/pupld95a_csv.zip", "PUPLDF95.csv"),
    (1996, "https://www.imls.gov/sites/default/files/pupld96a_csv.zip", "PUPLDF96.csv"),
    (1997, "https://www.imls.gov/sites/default/files/pupld97a_csv.zip", "PUPLDF97.csv"),
    (1998, "https://www.imls.gov/sites/default/files/pupldf98_csv.zip", "PUPLDF98.csv"),
    (1999, "https://www.imls.gov/sites/default/files/pupldf99_csv.zip", "PUPLDF99.csv"),
    (2000, "https://www.imls.gov/sites/default/files/pupldf00_csv.zip", "PUPLDF00.csv"),
    (2001, "https://www.imls.gov/sites/default/files/pupld01b_csv.zip", "PUPLDF01.csv"),
    (2002, "https://www.imls.gov/sites/default/files/pupld02b_csv.zip", "PUPLDF02.csv"),
    (2003, "https://www.imls.gov/sites/default/files/pupld03a_csv.zip", "PUPLDF03.csv"),
    (2004, "https://www.imls.gov/sites/default/files/pupld04a_csv.zip", "PUPLDF04.csv"),
    (2005, "https://www.imls.gov/sites/default/files/pupld05a_csv.zip", "PUPLDF05.csv"),
    (2006, "https://www.imls.gov/sites/default/files/pupld06a_csv.zip", "pupld06a.csv"),
    (2007, "https://www.imls.gov/sites/default/files/pupld07a_csv.zip", "pupld07a.csv"),
    (2008, "https://www.imls.gov/sites/default/files/pupld08a_csv.zip", "pupld08a.csv"),
    (2009, "https://www.imls.gov/sites/default/files/pupld09a_csv.zip", "pupld09a.csv"),
    (2010, "https://www.imls.gov/sites/default/files/pupld10a_csv.zip", "pupld10a.csv"),
    (2011, "https://www.imls.gov/sites/default/files/pupld11b_csv.zip", "pupld11a.csv"),
    (2012, "https://www.imls.gov/sites/default/files/pupld12a_csv.zip", "pupld12a.csv"),
    (2013, "https://www.imls.gov/sites/default/files/pupld13a_csv.zip", "pupld13a.csv"),
    (2014, "https://www.imls.gov/sites/default/files/pls_fy2014_data_files_csv.zip", "pls_fy2014_pupld14a.csv"),
    (2015, "https://www.imls.gov/sites/default/files/pls_fy2015_data_files_csv.zip", "pls_fy2015_pupld15a.csv"),
    (2016, "https://www.imls.gov/sites/default/files/pls_fy2016_data_files_csv.zip", "pls_fy2016_pupld16a.csv"),
    (2017, "https://www.imls.gov/sites/default/files/pls_fy2017_data_files_csv.zip", "pls_fy2017_pupld17a.csv"),
    (2018, "https://www.imls.gov/sites/default/files/pls_fy2018_data_files_csv.zip", "pls_fy2018_pupld18a.csv"),
    (2019, "https://www.imls.gov/sites/default/files/2021-05/pls_fy2019_csv.zip", "PLS_FY2019_AE_pud19i.csv"),
    (2020, "https://www.imls.gov/sites/default/files/2022-07/pls_fy2020_csv.zip", "PLS_FY20_AE_pud20i.csv"),
    (2021, "https://www.imls.gov/sites/default/files/2023-06/pls_fy2021_csv.zip", "PLS_FY21_AE_pud21i.csv"),
    (2022, "https://www.imls.gov/sites/default/files/2024-06/pls_fy2022_csv.zip", "PLS_FY22_AE_pud22i.csv"),
    (2023, "https://www.imls.gov/sites/default/files/2025-08/pls_fy2023_csv.zip", "PLS_FY23_AE_pud23i.csv"),
]

# Census ACS cross-section year for equity analysis (2019 = 5-year 2015–2019, pre-COVID)
CENSUS_ACS_YEAR: int = 2019

# HUD Continuums of Care (CoC) years available
HUD_PIT_YEARS: list[int] = [2019, 2020, 2023, 2024]
HUD_AWARD_YEARS: list[int] = [2018, 2019, 2020, 2021]
