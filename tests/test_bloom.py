"""Tests for Bloom filter implementations."""

import random
import string

import pytest

from prob_toolkit.bloom import BloomFilter, CountingBloomFilter, ScalableBloomFilter


class TestBloomFilter:
    def test_add_and_contains(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add("hello")
        bf.add("world")
        assert "hello" in bf
        assert "world" in bf
        assert "goodbye" not in bf  # Probably not

    def test_empty_filter(self):
        bf = BloomFilter(capacity=100)
        assert "anything" not in bf

    def test_multiple_adds(self):
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        items = [f"item_{i}" for i in range(100)]
        for item in items:
            bf.add(item)
        for item in items:
            assert item in bf

    def test_false_positive_rate(self):
        """Test that false positive rate is approximately as specified."""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        items = [f"item_{i}" for i in range(1000)]
        for item in items:
            bf.add(item)

        # Check non-existent items
        false_positives = 0
        test_count = 10000
        for _ in range(test_count):
            non_item = "".join(random.choices(string.ascii_lowercase, k=16))
            if non_item in bf:
                false_positives += 1

        fp_rate = false_positives / test_count
        assert fp_rate < 0.05, f"False positive rate {fp_rate:.4f} too high"

    def test_len(self):
        bf = BloomFilter(capacity=100)
        assert len(bf) == 0
        bf.add("a")
        bf.add("b")
        assert len(bf) == 2

    def test_clear(self):
        bf = BloomFilter(capacity=100)
        bf.add("test")
        assert "test" in bf
        bf.clear()
        assert "test" not in bf
        assert len(bf) == 0

    def test_fill_ratio(self):
        bf = BloomFilter(capacity=100, error_rate=0.1)
        assert bf.fill_ratio == 0.0
        bf.add("item")
        assert bf.fill_ratio > 0.0

    def test_merge(self):
        bf1 = BloomFilter(capacity=100, error_rate=0.01)
        bf2 = BloomFilter(capacity=100, error_rate=0.01)
        bf1.add("a")
        bf2.add("b")
        bf1.merge(bf2)
        assert "a" in bf1
        assert "b" in bf1

    def test_merge_incompatible(self):
        bf1 = BloomFilter(capacity=100, error_rate=0.01)
        bf2 = BloomFilter(capacity=1000, error_rate=0.001)
        with pytest.raises(ValueError):
            bf1.merge(bf2)

    def test_intersection(self):
        bf1 = BloomFilter(capacity=100, error_rate=0.01)
        bf2 = BloomFilter(capacity=100, error_rate=0.01)
        bf1.add("a")
        bf1.add("b")
        bf2.add("b")
        bf2.add("c")
        result = bf1.intersection(bf2)
        assert "b" in result
        # "a" might or might not be in the intersection due to the nature of Bloom filters

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            BloomFilter(capacity=0)

    def test_invalid_error_rate(self):
        with pytest.raises(ValueError):
            BloomFilter(capacity=100, error_rate=0.0)
        with pytest.raises(ValueError):
            BloomFilter(capacity=100, error_rate=1.0)

    def test_bytes_input(self):
        bf = BloomFilter(capacity=100)
        bf.add(b"binary data")
        assert b"binary data" in bf

    def test_int_input(self):
        bf = BloomFilter(capacity=100)
        bf.add(42)
        assert 42 in bf

    def test_optimal_parameters(self):
        """Test that optimal parameters are computed correctly."""
        bf = BloomFilter(capacity=1000, error_rate=0.001)
        # For n=1000, p=0.001, optimal k ≈ 10, m ≈ 14377
        assert bf.num_hashes >= 5
        assert bf.num_hashes <= 15
        assert bf.bit_size > 5000

    def test_serialization_roundtrip(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add("hello")
        bf.add("world")

        data = bf.to_bytes()
        bf2 = BloomFilter.from_bytes(data)

        assert bf2.bit_size == bf.bit_size
        assert bf2.num_hashes == bf.num_hashes
        assert "hello" in bf2
        assert "world" in bf2

    def test_repr(self):
        bf = BloomFilter(capacity=100)
        r = repr(bf)
        assert "BloomFilter" in r
        assert "capacity=100" in r


class TestCountingBloomFilter:
    def test_add_and_contains(self):
        cbf = CountingBloomFilter(capacity=100, error_rate=0.01)
        cbf.add("hello")
        cbf.add("world")
        assert "hello" in cbf
        assert "world" in cbf
        assert "goodbye" not in cbf

    def test_remove(self):
        cbf = CountingBloomFilter(capacity=100, error_rate=0.01)
        cbf.add("hello")
        assert "hello" in cbf
        assert cbf.remove("hello") is True
        assert "hello" not in cbf

    def test_remove_nonexistent(self):
        cbf = CountingBloomFilter(capacity=100, error_rate=0.01)
        assert cbf.remove("nonexistent") is False

    def test_len(self):
        cbf = CountingBloomFilter(capacity=100)
        cbf.add("a")
        cbf.add("b")
        assert len(cbf) == 2

    def test_clear(self):
        cbf = CountingBloomFilter(capacity=100)
        cbf.add("test")
        cbf.clear()
        assert len(cbf) == 0
        assert "test" not in cbf

    def test_fill_ratio(self):
        cbf = CountingBloomFilter(capacity=100)
        assert cbf.fill_ratio == 0.0
        cbf.add("item")
        assert cbf.fill_ratio > 0.0

    def test_repr(self):
        cbf = CountingBloomFilter(capacity=100)
        r = repr(cbf)
        assert "CountingBloomFilter" in r


class TestScalableBloomFilter:
    def test_add_and_contains(self):
        sbf = ScalableBloomFilter(initial_capacity=10, error_rate=0.01)
        for i in range(100):
            sbf.add(f"item_{i}")
        for i in range(100):
            assert f"item_{i}" in sbf

    def test_grows(self):
        sbf = ScalableBloomFilter(initial_capacity=10, error_rate=0.01)
        assert sbf.num_filters == 1
        for i in range(100):
            sbf.add(f"item_{i}")
        assert sbf.num_filters > 1

    def test_len(self):
        sbf = ScalableBloomFilter(initial_capacity=10)
        sbf.add("a")
        sbf.add("b")
        assert len(sbf) == 2

    def test_clear(self):
        sbf = ScalableBloomFilter(initial_capacity=10)
        for i in range(50):
            sbf.add(f"item_{i}")
        sbf.clear()
        assert len(sbf) == 0
        assert sbf.num_filters == 1

    def test_repr(self):
        sbf = ScalableBloomFilter()
        r = repr(sbf)
        assert "ScalableBloomFilter" in r
