"""Tests for HyperLogLog."""

import random
import string

import pytest

from prob_toolkit.hyperloglog import HyperLogLog


class TestHyperLogLog:
    def test_empty_estimate(self):
        hll = HyperLogLog(precision=14)
        assert hll.estimate() >= 0

    def test_add_and_estimate(self):
        hll = HyperLogLog(precision=14)
        for i in range(1000):
            hll.add(f"item_{i}")
        estimate = hll.estimate()
        # Should be close to 1000 (within 10% for precision=14)
        assert 800 < estimate < 1200, f"Estimate {estimate} too far from 1000"

    def test_cardinality_estimation_accuracy(self):
        """Test that HLL provides accurate cardinality estimates."""
        hll = HyperLogLog(precision=16)  # Higher precision for better accuracy
        n = 100000

        items = set()
        for _ in range(n):
            item = "".join(random.choices(string.ascii_lowercase, k=10))
            items.add(item)
            hll.add(item)

        estimate = hll.estimate()
        error = abs(estimate - n) / n
        assert error < 0.05, f"Error {error:.2%} too high for {n} items"

    def test_duplicates_dont_affect_estimate(self):
        """Adding the same element multiple times shouldn't change estimate."""
        hll = HyperLogLog(precision=14)
        for _ in range(1000):
            hll.add("same_item")
        # Should still estimate ~1
        assert hll.estimate() < 10

    def test_merge(self):
        hll1 = HyperLogLog(precision=14)
        hll2 = HyperLogLog(precision=14)

        for i in range(500):
            hll1.add(f"item_{i}")
        for i in range(500, 1000):
            hll2.add(f"item_{i}")

        hll1.merge(hll2)
        estimate = hll1.estimate()
        assert 800 < estimate < 1200, f"Estimate {estimate} too far from 1000"

    def test_merge_incompatible(self):
        hll1 = HyperLogLog(precision=14)
        hll2 = HyperLogLog(precision=16)
        with pytest.raises(ValueError):
            hll1.merge(hll2)

    def test_clear(self):
        hll = HyperLogLog(precision=14)
        for i in range(100):
            hll.add(f"item_{i}")
        hll.clear()
        assert hll.estimate() == 0

    def test_num_buckets(self):
        hll = HyperLogLog(precision=10)
        assert hll.num_buckets == 1024

    def test_registers(self):
        hll = HyperLogLog(precision=10)
        regs = hll.registers
        assert len(regs) == 1024
        assert all(r == 0 for r in regs)

    def test_invalid_precision(self):
        with pytest.raises(ValueError):
            HyperLogLog(precision=2)
        with pytest.raises(ValueError):
            HyperLogLog(precision=20)

    def test_bytes_input(self):
        hll = HyperLogLog(precision=14)
        hll.add(b"binary")
        assert hll.estimate() >= 0

    def test_int_input(self):
        hll = HyperLogLog(precision=14)
        hll.add(42)
        assert hll.estimate() >= 0

    def test_serialization_roundtrip(self):
        hll = HyperLogLog(precision=14)
        for i in range(100):
            hll.add(f"item_{i}")

        data = hll.to_bytes()
        hll2 = HyperLogLog.from_bytes(data)

        assert hll2.precision == hll.precision
        assert hll2.num_buckets == hll.num_buckets
        # Estimates should be very close
        assert abs(hll.estimate() - hll2.estimate()) < 10

    def test_repr(self):
        hll = HyperLogLog(precision=14)
        r = repr(hll)
        assert "HyperLogLog" in r
        assert "precision=14" in r

    def test_streaming_simulation(self):
        """Simulate a streaming scenario with millions of unique items."""
        hll = HyperLogLog(precision=14)
        seen = set()
        n = 100000

        for _ in range(n):
            item = str(random.randint(0, n * 10))
            seen.add(item)
            hll.add(item)

        true_count = len(seen)
        estimate = hll.estimate()
        error = abs(estimate - true_count) / true_count

        assert error < 0.1, f"Streaming error {error:.2%} too high"
