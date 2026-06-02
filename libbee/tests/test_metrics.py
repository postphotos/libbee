"""Unit tests for libbee.metrics — conformed metric vocabulary."""

from __future__ import annotations

from libbee import metrics


class TestConformedMetrics:
    """Test the CONFORMED metrics list."""

    def test_conformed_list_exists(self):
        """CONFORMED is a list of metric names."""
        assert isinstance(metrics.CONFORMED, list)
        assert len(metrics.CONFORMED) > 0

    def test_conformed_contains_key_metrics(self):
        """CONFORMED includes all key metrics."""
        expected = {
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
        }
        assert set(metrics.CONFORMED) == expected

    def test_conformed_all_strings(self):
        """All items in CONFORMED are strings."""
        assert all(isinstance(m, str) for m in metrics.CONFORMED)

    def test_conformed_no_duplicates(self):
        """CONFORMED has no duplicate metrics."""
        assert len(metrics.CONFORMED) == len(set(metrics.CONFORMED))


class TestLabels:
    """Test the LABELS dictionary."""

    def test_labels_dict_exists(self):
        """LABELS is a dict."""
        assert isinstance(metrics.LABELS, dict)

    def test_labels_keys_are_conformed(self):
        """All LABELS keys are in CONFORMED."""
        for key in metrics.LABELS:
            assert key in metrics.CONFORMED

    def test_labels_values_are_strings(self):
        """All LABELS values are strings."""
        assert all(isinstance(v, str) for v in metrics.LABELS.values())

    def test_labels_contain_key_metrics(self):
        """LABELS includes descriptions for key metrics."""
        assert "visits_pc" in metrics.LABELS
        assert "funding_pc" in metrics.LABELS
        assert "homeless_per10k" in metrics.LABELS

    def test_labels_examples(self):
        """LABELS have expected human-readable descriptions."""
        assert "visits" in metrics.LABELS["visits_pc"].lower()
        assert "$" in metrics.LABELS["funding_pc"].lower()
        assert "wifi" in metrics.LABELS["wifi_pc"].lower()

    def test_labels_non_empty(self):
        """LABELS has at least some entries."""
        assert len(metrics.LABELS) > 0
