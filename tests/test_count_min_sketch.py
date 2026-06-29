"""Tests for Count-Min Sketch."""

import pytest

from prob_toolkit.count_min_sketch import CountMinSketch


class TestCountMinSketch:
    def test_add_and_estimate(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        for _ in range(100):
            cms.add("item")
        assert cms.estimate("item") >= 100

    def test_never_underestimates(self):
        """Count-Min Sketch should never underestimate."""
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        for _ in range(50):
            cms.add("target")
        estimate = cms.estimate("target")
        assert estimate >= 50, f"Underestimated: {estimate} < 50"

    def test_overestimate_bound(self):
        """Over-estimation should be bounded by epsilon * total_count."""
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        n = 10000

        # Add many items
        for i in range(n):
            cms.add(f"item_{i}")

        # The estimate for any item should be at most (true_count + epsilon * n)
        for i in range(100):
            est = cms.estimate(f"item_{i}")
            assert est <= 1 + cms.error_bound + 1, f"Over-estimate too high: {est}"

    def test_different_items(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        cms.add("a", count=10)
        cms.add("b", count=20)
        assert cms.estimate("a") >= 10
        assert cms.estimate("b") >= 20

    def test_add_with_count(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        cms.add("item", count=5)
        assert cms.estimate("item") >= 5

    def test_total_count(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        cms.add("a", count=3)
        cms.add("b", count=7)
        assert cms.total_count == 10

    def test_merge(self):
        cms1 = CountMinSketch(epsilon=0.001, delta=0.01)
        cms2 = CountMinSketch(epsilon=0.001, delta=0.01)
        cms1.add("a", count=10)
        cms2.add("a", count=5)
        cms1.merge(cms2)
        assert cms1.estimate("a") >= 15

    def test_merge_incompatible(self):
        cms1 = CountMinSketch(epsilon=0.001, delta=0.01)
        cms2 = CountMinSketch(epsilon=0.01, delta=0.1)
        with pytest.raises(ValueError):
            cms1.merge(cms2)

    def test_clear(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        cms.add("item", count=100)
        cms.clear()
        assert cms.estimate("item") == 0
        assert cms.total_count == 0

    def test_dimensions(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        assert cms.width > 0
        assert cms.depth > 0

    def test_error_bound(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        cms.add("item", count=1000)
        assert cms.error_bound == int(0.001 * 1000)

    def test_width_depth_override(self):
        cms = CountMinSketch(width=100, depth=5)
        assert cms.width == 100
        assert cms.depth == 5

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            CountMinSketch(epsilon=0, delta=0.01)
        with pytest.raises(ValueError):
            CountMinSketch(epsilon=1.5, delta=0.01)

    def test_invalid_delta(self):
        with pytest.raises(ValueError):
            CountMinSketch(epsilon=0.001, delta=0)
        with pytest.raises(ValueError):
            CountMinSketch(epsilon=0.001, delta=1)

    def test_serialization_roundtrip(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        cms.add("hello", count=10)
        cms.add("world", count=20)

        data = cms.to_bytes()
        cms2 = CountMinSketch.from_bytes(data)

        assert cms2.width == cms.width
        assert cms2.depth == cms.depth
        assert cms2.total_count == cms.total_count
        assert cms2.estimate("hello") == cms.estimate("hello")
        assert cms2.estimate("world") == cms.estimate("world")

    def test_getitem(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        cms.add("item", count=42)
        assert cms["item"] >= 42

    def test_repr(self):
        cms = CountMinSketch(epsilon=0.001, delta=0.01)
        r = repr(cms)
        assert "CountMinSketch" in r
