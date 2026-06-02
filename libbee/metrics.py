"""The conformed metric vocabulary — one shared list so every adapter and `facts` agree.

To support a metric a new dataset brings, add it here; `melt()` then picks it up automatically.
"""

CONFORMED = [
    "visits_pc",
    "checkouts_pc",
    "funding_pc",
    "wifi_pc",
    "hours_pc",
    "staff_pc",
    "cost_per_visit",
    "outlets_per_100k",
    "collection_pc",
    "homeless_per10k",
    "homeless_spend_pc",
    "poverty",
    "median_income",
    "no_broadband",
    "density",
]

# Human labels (used by the tools; optional).
LABELS = {
    "visits_pc": "library visits / resident",
    "checkouts_pc": "checkouts / resident",
    "funding_pc": "library $ / resident",
    "wifi_pc": "WiFi sessions / resident",
    "hours_pc": "open hours / resident",
    "staff_pc": "staff / resident",
    "cost_per_visit": "$ per visit",
    "outlets_per_100k": "outlets / 100k",
    "collection_pc": "volumes / resident",
    "homeless_per10k": "homelessness / 10k",
    "homeless_spend_pc": "homeless grant $ / resident",
    "poverty": "poverty rate",
    "median_income": "median household income",
    "no_broadband": "no home broadband",
    "density": "people / sq mi",
}
