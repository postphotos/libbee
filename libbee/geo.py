"""Shared geography helpers (FIPS ↔ state, names, regions, US-states GeoJSON), used by adapters
and the notebooks — so neither has to redefine a 50-row FIPS dict or re-implement centroids."""

from __future__ import annotations

FIPS2ST = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

ST_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

ST2FIPS = {v: k for k, v in FIPS2ST.items()}
FIPS2NAME = {fips: ST_NAME[st] for fips, st in FIPS2ST.items()}

# Regional presets — "select our area items" in one click (used by the cohort map's preset tab).
REGIONS: dict[str, list[str]] = {
    "Mountain West": ["CO", "UT", "WY", "MT", "ID", "NV", "AZ", "NM"],
    "West Coast": ["CA", "OR", "WA"],
    "New England": ["MA", "CT", "RI", "NH", "VT", "ME"],
    "Midwest": ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
    "South Atlantic": ["FL", "GA", "NC", "SC", "VA", "WV", "MD", "DE", "DC"],
    "Big four (CA · MA vs AL · WY)": ["CA", "MA", "AL", "WY"],
    "Colorado + neighbors": ["CO", "WY", "NE", "KS", "NM", "UT", "AZ", "OK"],
}

# Census region per state — the cluster used to colour the leaderboard so regional patterns pop.
CENSUS_REGION: dict[str, str] = {
    **{s: "Northeast" for s in ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"]},
    **{s: "Midwest" for s in ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"]},
    **{
        s: "South"
        for s in [
            "DE",
            "DC",
            "FL",
            "GA",
            "MD",
            "NC",
            "SC",
            "VA",
            "WV",
            "AL",
            "KY",
            "MS",
            "TN",
            "AR",
            "LA",
            "OK",
            "TX",
        ]
    },
    **{s: "West" for s in ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"]},
}
REGION_ORDER = ["Northeast", "Midwest", "South", "West"]
# A colour-blind-safe hue per region (for cluster colouring on a single plot).
REGION_COLOR = {"Northeast": "#2563eb", "Midwest": "#16a34a", "South": "#d97706", "West": "#9333ea"}

# Partisan lean per state — the 2020 presidential popular-vote winner in each state (Biden = "blue",
# Trump = "red"). A public, fixed historical result; used by the §M5 scenario presets that ask what a
# blue-only vs red-only library-funding policy is predicted to do. (DC = blue; ME/NE by statewide vote.)
STATE_PARTY: dict[str, str] = {
    **{
        s: "blue"
        for s in ["AZ", "CA", "CO", "CT", "DC", "DE", "GA", "HI", "IL", "ME", "MD", "MA", "MI", "MN", "NV", "NH", "NJ", "NM", "NY", "OR", "PA", "RI", "VT", "VA", "WA", "WI"]
    },
    **{s: "red" for s in ["AL", "AK", "AR", "FL", "ID", "IN", "IA", "KS", "KY", "LA", "MS", "MO", "MT", "NE", "NC", "ND", "OH", "OK", "SC", "SD", "TN", "TX", "UT", "WV", "WY"]},
}
# A hue per party for the scenario charts (blue / red, colour-blind-distinguishable).
PARTY_COLOR = {"blue": "#2563eb", "red": "#dc2626"}

# Traditional/AP-style state abbreviations for human county labels ("Los Angeles County, Calif.").
# The 8 short states are never abbreviated in AP style (spelled out): AK, HI, ID, IA, ME, OH, TX, UT.
CENSUS_ABBR: dict[str, str] = {
    "AL": "Ala.",
    "AK": "Alaska",
    "AZ": "Ariz.",
    "AR": "Ark.",
    "CA": "Calif.",
    "CO": "Colo.",
    "CT": "Conn.",
    "DE": "Del.",
    "DC": "D.C.",
    "FL": "Fla.",
    "GA": "Ga.",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Ill.",
    "IN": "Ind.",
    "IA": "Iowa",
    "KS": "Kan.",
    "KY": "Ky.",
    "LA": "La.",
    "ME": "Maine",
    "MD": "Md.",
    "MA": "Mass.",
    "MI": "Mich.",
    "MN": "Minn.",
    "MS": "Miss.",
    "MO": "Mo.",
    "MT": "Mont.",
    "NE": "Neb.",
    "NV": "Nev.",
    "NH": "N.H.",
    "NJ": "N.J.",
    "NM": "N.M.",
    "NY": "N.Y.",
    "NC": "N.C.",
    "ND": "N.D.",
    "OH": "Ohio",
    "OK": "Okla.",
    "OR": "Ore.",
    "PA": "Pa.",
    "RI": "R.I.",
    "SC": "S.C.",
    "SD": "S.D.",
    "TN": "Tenn.",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vt.",
    "VA": "Va.",
    "WA": "Wash.",
    "WV": "W.Va.",
    "WI": "Wis.",
    "WY": "Wyo.",
}

# Short airport-style codes for the top metros — compact labels (LAX, CHI, SEA, DEN) for dense plots.
# Keyed by the metro's lead city (the token before the first '-'); use metro_code() to resolve a name.
METRO_CODE: dict[str, str] = {
    "New York": "NYC",
    "Los Angeles": "LAX",
    "Chicago": "CHI",
    "Dallas": "DFW",
    "Houston": "HOU",
    "Washington": "DC",
    "Miami": "MIA",
    "Philadelphia": "PHL",
    "Atlanta": "ATL",
    "Phoenix": "PHX",
    "Boston": "BOS",
    "San Francisco": "SFO",
    "Riverside": "RIV",
    "Detroit": "DTW",
    "Seattle": "SEA",
    "Minneapolis": "MSP",
    "San Diego": "SAN",
    "Tampa": "TPA",
    "Denver": "DEN",
    "St. Louis": "STL",
    "Baltimore": "BWI",
    "Charlotte": "CLT",
    "Orlando": "MCO",
    "San Antonio": "SAT",
    "Portland": "PDX",
    "Sacramento": "SMF",
    "Pittsburgh": "PIT",
    "Las Vegas": "LAS",
    "Austin": "AUS",
    "Columbus": "CMH",
}


def metro_code(metro_name: str) -> str:
    """Short code for a metro (CBSA) name, e.g. 'Denver-Aurora-Lakewood' -> 'DEN'. Falls back to the
    first three letters of the lead city if the metro isn't in METRO_CODE."""
    lead = metro_name.replace(",", "-").split("-")[0].strip()
    return METRO_CODE.get(lead, lead[:3].upper())


def region_of(state_abbr: str) -> str:
    """Census region ('Northeast' | 'Midwest' | 'South' | 'West') for a 2-letter state.

    Example::

        libbee.geo.region_of("CA")     # 'West'
        libbee.geo.metro_code("Seattle-Tacoma-Bellevue")   # 'SEA'
        # add a region column to any state frame:
        df.with_columns(pl.col("st").replace_strict(libbee.geo.CENSUS_REGION, default=None).alias("region"))
    """
    return CENSUS_REGION.get(state_abbr, "—")


_GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"


def us_states_geojson() -> list[dict]:
    """The US-states GeoJSON *features* (FIPS `id` + name + geometry), downloaded once and cached
    under data/raw/geo/. Inline (not a URL) so map clicks return real rows."""
    import json
    import urllib.request

    from .io.paths import RAW_GEO

    dest = RAW_GEO / "us_states.geojson"
    if not dest.exists():
        RAW_GEO.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_GEOJSON_URL, dest)
    with open(dest) as fh:
        return json.load(fh)["features"]


_TOPO_URL = "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/us-10m.json"


def us_topojson() -> dict:
    """The vega-datasets US TopoJSON (object `states`, integer-FIPS ids), cached under data/raw/geo/.
    Unlike the hand GeoJSON, its ring winding renders correctly under d3/albersUsa — use it as the
    choropleth basemap with a `transform_lookup` on `id`."""
    import json
    import urllib.request

    from .io.paths import RAW_GEO

    dest = RAW_GEO / "us_10m.topojson"
    if not dest.exists():
        RAW_GEO.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_TOPO_URL, dest)
    with open(dest) as fh:
        return json.load(fh)


def _topo_decode_arcs(topo: dict) -> list:
    """Decode every quantized, delta-encoded arc into absolute lon/lat points."""
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    out = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append([x * sx + tx, y * sy + ty])
        out.append(pts)
    return out


def _topo_ring(arc_ids: list, decoded: list) -> list:
    """Stitch a ring from its arc indices (negative index = that arc reversed)."""
    coords: list = []
    for idx in arc_ids:
        seg = decoded[idx] if idx >= 0 else decoded[-idx - 1][::-1]
        coords.extend(seg if not coords else seg[1:])  # drop the shared endpoint
    return coords


def states_geojson() -> dict:
    """Decode ONLY the `states` object of the cached us-10m TopoJSON into a small geographic GeoJSON
    FeatureCollection (`id` = FIPS, `properties.name`). Dropping every county arc takes the inline
    basemap from ~6 MB-when-expanded to ~0.2 MB — small enough to embed in a chart with the data baked
    into properties, while albersUsa still auto-fits it. This is `libbee` parsing the data smaller.

    Example::

        fc = libbee.geo.states_geojson()                 # {'type': 'FeatureCollection', 'features': [...]}
        alt.Chart(alt.Data(values=fc, format=alt.DataFormat(type="json", property="features"))) \
           .mark_geoshape().project("albersUsa")        # inline choropleth, no external topojson fetch
    """
    topo = us_topojson()
    decoded = _topo_decode_arcs(topo)
    feats = []
    for g in topo["objects"]["states"]["geometries"]:
        gt = g["type"]
        if gt == "Polygon":
            geom = {"type": "Polygon", "coordinates": [_topo_ring(r, decoded) for r in g["arcs"]]}
        elif gt == "MultiPolygon":
            geom = {
                "type": "MultiPolygon",
                "coordinates": [[_topo_ring(r, decoded) for r in poly] for poly in g["arcs"]],
            }
        else:
            continue
        feats.append(
            {
                "type": "Feature",
                "id": g.get("id"),
                "properties": dict(g.get("properties") or {}),
                "geometry": geom,
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _flatten_coords(coords, out: list) -> None:
    if coords and isinstance(coords[0], (int, float)):
        out.append(coords)
    else:
        for c in coords:
            _flatten_coords(c, out)


def centroids(features: list[dict] | None = None) -> list[dict]:
    """One {st, name, lon, lat} per state — the naive coordinate centroid, good enough to place a
    clickable label. Defaults to the cached US-states GeoJSON."""
    feats = features if features is not None else us_states_geojson()
    rows = []
    for f in feats:
        fips = str(f["id"]).zfill(2)
        st = FIPS2ST.get(fips)
        if not st:
            continue
        pts: list = []
        _flatten_coords(f["geometry"]["coordinates"], pts)
        rows.append(
            {
                "st": st,
                "name": ST_NAME.get(st, st),
                "lon": sum(p[0] for p in pts) / len(pts),
                "lat": sum(p[1] for p in pts) / len(pts),
            }
        )
    return rows
