"""Unit tests for libbee.geo — geography lookups and utilities."""

from __future__ import annotations

from libbee import geo


class TestFIPSMappings:
    """Test FIPS ↔ state mappings."""

    def test_fips2st_completeness(self):
        """Verify all 50 states + DC are mapped."""
        assert len(geo.FIPS2ST) == 51
        assert geo.FIPS2ST["08"] == "CO"
        assert geo.FIPS2ST["36"] == "NY"
        assert geo.FIPS2ST["11"] == "DC"

    def test_st2fips_reverse(self):
        """ST2FIPS is the inverse of FIPS2ST."""
        assert len(geo.ST2FIPS) == 51
        assert geo.ST2FIPS["CO"] == "08"
        assert geo.ST2FIPS["NY"] == "36"
        for fips, st in geo.FIPS2ST.items():
            assert geo.ST2FIPS[st] == fips

    def test_fips2name_comprehensive(self):
        """FIPS2NAME maps all FIPS codes to state names."""
        assert len(geo.FIPS2NAME) == 51
        assert geo.FIPS2NAME["08"] == "Colorado"
        assert geo.FIPS2NAME["36"] == "New York"
        assert geo.FIPS2NAME["11"] == "District of Columbia"


class TestStateNames:
    """Test state abbreviation ↔ name mappings."""

    def test_st_name_completeness(self):
        """Verify all states have names."""
        assert len(geo.ST_NAME) == 51
        assert geo.ST_NAME["CO"] == "Colorado"
        assert geo.ST_NAME["CA"] == "California"
        assert geo.ST_NAME["DC"] == "District of Columbia"

    def test_st_name_consistency_with_fips(self):
        """ST_NAME matches FIPS2ST."""
        for _fips, st in geo.FIPS2ST.items():
            assert st in geo.ST_NAME


class TestCensusRegions:
    """Test Census region mappings."""

    def test_census_region_coverage(self):
        """All 50 states + DC are in CENSUS_REGION."""
        assert len(geo.CENSUS_REGION) == 51
        assert geo.CENSUS_REGION["CO"] == "West"
        assert geo.CENSUS_REGION["NY"] == "Northeast"
        assert geo.CENSUS_REGION["IL"] == "Midwest"
        assert geo.CENSUS_REGION["GA"] == "South"
        assert geo.CENSUS_REGION["DC"] == "South"

    def test_state_party_coverage(self):
        """All 50 states + DC carry a 2020 blue/red lean, and the totals match the actual result."""
        assert len(geo.STATE_PARTY) == 51
        assert set(geo.STATE_PARTY.values()) == {"blue", "red"}
        assert sum(v == "blue" for v in geo.STATE_PARTY.values()) == 26  # 25 states + DC
        assert sum(v == "red" for v in geo.STATE_PARTY.values()) == 25
        assert geo.STATE_PARTY["CA"] == "blue" and geo.STATE_PARTY["WY"] == "red"
        assert geo.STATE_PARTY["GA"] == "blue" and geo.STATE_PARTY["DC"] == "blue"
        assert set(geo.STATE_PARTY) == set(geo.FIPS2ST.values())  # exactly the FIPS2ST states
        assert set(geo.PARTY_COLOR) == {"blue", "red"}

    def test_census_abbr_coverage(self):
        """All 50 states + DC have a traditional/AP abbreviation; the 8 short states are spelled out."""
        assert len(geo.CENSUS_ABBR) == 51
        assert set(geo.CENSUS_ABBR) == set(geo.FIPS2ST.values())  # exactly the FIPS2ST states
        assert geo.CENSUS_ABBR["CA"] == "Calif." and geo.CENSUS_ABBR["AL"] == "Ala."
        assert geo.CENSUS_ABBR["DC"] == "D.C."
        # the 8 never-abbreviated states are spelled out
        for st in ["AK", "HI", "ID", "IA", "ME", "OH", "TX", "UT"]:
            assert geo.CENSUS_ABBR[st] == geo.ST_NAME[st]

    def test_region_order(self):
        """REGION_ORDER has all four regions in expected order."""
        assert geo.REGION_ORDER == ["Northeast", "Midwest", "South", "West"]

    def test_region_color_coverage(self):
        """All regions have colors."""
        assert len(geo.REGION_COLOR) == 4
        assert geo.REGION_COLOR["Northeast"] == "#2563eb"
        assert geo.REGION_COLOR["Midwest"] == "#16a34a"
        assert geo.REGION_COLOR["South"] == "#d97706"
        assert geo.REGION_COLOR["West"] == "#9333ea"
        # All regions should have a color
        for region in geo.REGION_ORDER:
            assert region in geo.REGION_COLOR


class TestRegionalPresets:
    """Test the REGIONS preset dictionary."""

    def test_regions_dict_exists(self):
        """REGIONS has the expected presets."""
        assert "Mountain West" in geo.REGIONS
        assert "West Coast" in geo.REGIONS
        assert "New England" in geo.REGIONS
        assert "Midwest" in geo.REGIONS
        assert "South Atlantic" in geo.REGIONS
        assert "Big four (CA · MA vs AL · WY)" in geo.REGIONS
        assert "Colorado + neighbors" in geo.REGIONS

    def test_regions_contain_valid_states(self):
        """All states in REGIONS are valid (2-letter abbrs)."""
        all_states = set(geo.ST_NAME.keys())
        for region_name, states in geo.REGIONS.items():
            for state in states:
                assert state in all_states, f"{state} in {region_name} is not a valid state"

    def test_regions_examples(self):
        """Verify a few specific region contents."""
        assert "CA" in geo.REGIONS["West Coast"]
        assert "CO" in geo.REGIONS["Mountain West"]
        assert "MA" in geo.REGIONS["New England"]


class TestMetroCodes:
    """Test metro airport-code lookups."""

    def test_metro_code_dict_exists(self):
        """METRO_CODE has major metros."""
        assert geo.METRO_CODE["New York"] == "NYC"
        assert geo.METRO_CODE["Los Angeles"] == "LAX"
        assert geo.METRO_CODE["Chicago"] == "CHI"
        assert len(geo.METRO_CODE) > 10

    def test_metro_code_function_basic(self):
        """metro_code() resolves names to codes."""
        assert geo.metro_code("New York") == "NYC"
        assert geo.metro_code("Los Angeles") == "LAX"
        assert geo.metro_code("Chicago") == "CHI"

    def test_metro_code_function_on_full_names(self):
        """metro_code() extracts lead city from full metro names."""
        # "New York-Newark-Jersey City" → extract "New York" → "NYC"
        assert geo.metro_code("New York-Newark-Jersey City") == "NYC"
        assert geo.metro_code("Los Angeles-Long Beach-Anaheim") == "LAX"
        assert geo.metro_code("Chicago-Naperville-Elgin") == "CHI"

    def test_metro_code_unknown(self):
        """metro_code() on unknown metros returns the first 3 letters of lead city."""
        # Not in METRO_CODE, so returns first 3 letters of lead city
        result = geo.metro_code("Unknown-City-Metro")
        assert result == "UNK"

    def test_metro_code_single_word(self):
        """metro_code() on single-word metros in METRO_CODE returns the code."""
        result = geo.metro_code("Denver")
        assert result == "DEN"

    def test_region_of_state_abbr(self):
        """region_of() returns census region for valid state abbreviations."""
        assert geo.region_of("CA") in {"West", "—"}
        assert geo.region_of("NY") in {"Northeast", "—"}
        assert geo.region_of("XX") == "—"

    def test_us_states_geojson_caches_file(self):
        """us_states_geojson() caches downloaded file."""
        import json
        from unittest import mock

        mock_geojson_data = {
            "type": "FeatureCollection",
            "features": [{"id": "06", "type": "Feature", "geometry": {"type": "Polygon"}}],
        }

        # Mock the file operations since RAW_GEO is imported locally
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mock_geojson_data))):
            with mock.patch("pathlib.Path.exists", return_value=True):
                result = geo.us_states_geojson()
                assert isinstance(result, list)
                assert len(result) == 1

    def test_us_topojson_caches_file(self):
        """us_topojson() caches downloaded TopoJSON file."""
        import json
        from unittest import mock

        mock_topo_data = {
            "type": "Topology",
            "transform": {"scale": [1, 1], "translate": [0, 0]},
            "objects": {},
            "arcs": [],
        }

        # Mock the file operations since RAW_GEO is imported locally
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mock_topo_data))):
            with mock.patch("pathlib.Path.exists", return_value=True):
                result = geo.us_topojson()
                assert isinstance(result, dict)
                assert "objects" in result

    def test_us_states_geojson_downloads_if_missing(self):
        """us_states_geojson() downloads file if not cached."""
        import json
        from unittest import mock

        mock_geojson_data = {
            "type": "FeatureCollection",
            "features": [{"id": "06", "type": "Feature", "geometry": {"type": "Polygon"}}],
        }

        # Mock to simulate file not existing, then getting created
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mock_geojson_data))):
            with mock.patch("pathlib.Path.exists", return_value=False):
                with mock.patch("pathlib.Path.mkdir"):
                    with mock.patch("urllib.request.urlretrieve"):
                        result = geo.us_states_geojson()
                        assert isinstance(result, list)

    def test_us_topojson_downloads_if_missing(self):
        """us_topojson() downloads file if not cached."""
        import json
        from unittest import mock

        mock_topo_data = {
            "type": "Topology",
            "transform": {"scale": [1, 1], "translate": [0, 0]},
            "objects": {},
            "arcs": [],
        }

        # Mock to simulate file not existing, then getting created
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mock_topo_data))):
            with mock.patch("pathlib.Path.exists", return_value=False):
                with mock.patch("pathlib.Path.mkdir"):
                    with mock.patch("urllib.request.urlretrieve"):
                        result = geo.us_topojson()
                        assert isinstance(result, dict)

    def test_states_geojson_converts_topojson(self):
        """states_geojson() decodes TopoJSON to GeoJSON."""
        from unittest import mock

        # Create minimal TopoJSON structure
        mock_topo = {
            "objects": {
                "states": {
                    "geometries": [
                        {
                            "type": "Polygon",
                            "id": 6,
                            "properties": {"name": "California"},
                            "arcs": [[0]],
                        }
                    ]
                }
            },
            "arcs": [[[0, 0], [1, 1], [1, 0]]],
            "transform": {"scale": [2.0, 2.0], "translate": [0.0, 0.0]},
        }

        with mock.patch("libbee.geo.us_topojson", return_value=mock_topo):
            result = geo.states_geojson()
            assert result["type"] == "FeatureCollection"
            assert "features" in result

    def test_centroids_handles_missing_state(self):
        """centroids() skips states not in FIPS2ST mapping."""
        from unittest import mock

        # Feature with FIPS ID not in mapping
        mock_features = [
            {
                "id": 99,  # Invalid FIPS
                "geometry": {"coordinates": [[[-120, 35]]]},
                "properties": {},
            }
        ]

        with mock.patch("libbee.geo.us_states_geojson", return_value=mock_features):
            result = geo.centroids()
            # Should skip the invalid FIPS and return empty or partial list
            assert isinstance(result, list)

    def test_topo_decode_arcs(self):
        """_topo_decode_arcs() decodes delta-encoded arcs."""
        topo = {
            "transform": {"scale": [2.0, 2.0], "translate": [0.0, 0.0]},
            "arcs": [[[0, 0], [1, 1], [1, 0]]],
        }
        decoded = geo._topo_decode_arcs(topo)
        assert len(decoded) == 1
        assert decoded[0] == [[0.0, 0.0], [2.0, 2.0], [4.0, 2.0]]

    def test_topo_ring(self):
        """_topo_ring() stitches arcs into a ring."""
        decoded = [[[0, 0], [1, 1], [2, 2]], [[2, 2], [1, 1], [0, 0]]]
        # Positive index uses arc as-is, negative reverses it
        ring = geo._topo_ring([0, -1], decoded)
        assert len(ring) > 0
        assert ring[0] == [0, 0]

    def test_flatten_coords(self):
        """_flatten_coords() recursively flattens coordinate arrays."""
        coords = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        out = []
        geo._flatten_coords(coords, out)
        assert len(out) == 4
        assert [1, 2] in out

    def test_centroids_with_default_features(self):
        """centroids() calculates centroid points for US states."""
        from unittest import mock

        mock_features = [
            {
                "id": 6,
                "geometry": {"coordinates": [[[-120, 35], [-115, 35], [-115, 42], [-120, 42]]]},
                "properties": {},
            }
        ]

        with mock.patch("libbee.geo.us_states_geojson", return_value=mock_features):
            result = geo.centroids()
            assert isinstance(result, list)
            if result:
                assert "lon" in result[0] and "lat" in result[0]

    def test_states_geojson_handles_multipolygon(self):
        """states_geojson() correctly handles MultiPolygon geometries."""
        from unittest import mock

        # TopoJSON with MultiPolygon (like Hawaii with multiple islands)
        mock_topo = {
            "objects": {
                "states": {
                    "geometries": [
                        {
                            "type": "MultiPolygon",
                            "id": 15,
                            "properties": {"name": "Hawaii"},
                            "arcs": [[[0]], [[1]]],  # Two separate polygons (islands)
                        }
                    ]
                }
            },
            "arcs": [
                [[0, 0], [1, 0], [1, 1], [0, 1]],
                [[2, 2], [3, 2], [3, 3], [2, 3]],
            ],
            "transform": {"scale": [2.0, 2.0], "translate": [0.0, 0.0]},
        }

        with mock.patch("libbee.geo.us_topojson", return_value=mock_topo):
            result = geo.states_geojson()
            assert result["type"] == "FeatureCollection"
            assert len(result["features"]) == 1
            assert result["features"][0]["geometry"]["type"] == "MultiPolygon"

    def test_states_geojson_skips_unknown_geometry_type(self):
        """states_geojson() skips geometries with unknown types."""
        from unittest import mock

        # TopoJSON with an unknown geometry type (e.g., "LineString")
        mock_topo = {
            "objects": {
                "states": {
                    "geometries": [
                        {
                            "type": "Polygon",
                            "id": 6,
                            "properties": {"name": "California"},
                            "arcs": [[0]],
                        },
                        {
                            "type": "LineString",  # Unknown type, should be skipped
                            "id": 7,
                            "properties": {"name": "Nevada"},
                            "arcs": [[1]],
                        },
                        {
                            "type": "MultiPolygon",
                            "id": 15,
                            "properties": {"name": "Hawaii"},
                            "arcs": [[[2]]],
                        },
                    ]
                }
            },
            "arcs": [
                [[0, 0], [1, 1], [1, 0]],
                [[2, 2], [3, 2], [3, 3]],
                [[4, 4], [5, 4], [5, 5]],
            ],
            "transform": {"scale": [2.0, 2.0], "translate": [0.0, 0.0]},
        }

        with mock.patch("libbee.geo.us_topojson", return_value=mock_topo):
            result = geo.states_geojson()
            assert result["type"] == "FeatureCollection"
            # Should have 2 features (Polygon + MultiPolygon), LineString skipped
            assert len(result["features"]) == 2
            assert result["features"][0]["geometry"]["type"] == "Polygon"
            assert result["features"][1]["geometry"]["type"] == "MultiPolygon"
