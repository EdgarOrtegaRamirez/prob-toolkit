"""Tests for Cuckoo filter."""

import random
import string

from prob_toolkit.cuckoo import CuckooFilter


class TestCuckooFilter:
    def test_add_and_contains(self):
        cf = CuckooFilter(capacity=100, error_rate=0.01)
        cf.add("hello")
        cf.add("world")
        assert "hello" in cf
        assert "world" in cf
        assert "goodbye" not in cf

    def test_empty_filter(self):
        cf = CuckooFilter(capacity=100)
        assert "anything" not in cf

    def test_remove(self):
        cf = CuckooFilter(capacity=100, error_rate=0.01)
        cf.add("hello")
        assert cf.remove("hello") is True
        assert "hello" not in cf

    def test_remove_nonexistent(self):
        cf = CuckooFilter(capacity=100)
        assert cf.remove("nonexistent") is False

    def test_len(self):
        cf = CuckooFilter(capacity=100)
        assert len(cf) == 0
        cf.add("a")
        cf.add("b")
        assert len(cf) == 2

    def test_fill_ratio(self):
        cf = CuckooFilter(capacity=100)
        assert cf.fill_ratio == 0.0
        cf.add("item")
        assert cf.fill_ratio > 0.0

    def test_bits_per_element(self):
        cf = CuckooFilter(capacity=100, fingerprint_bits=16)
        cf.add("item")
        bpe = cf.bits_per_element
        assert bpe > 0

    def test_false_positive_rate(self):
        """Test that false positive rate is approximately as specified."""
        cf = CuckooFilter(capacity=1000, error_rate=0.01)
        items = [f"item_{i}" for i in range(500)]
        for item in items:
            cf.add(item)

        false_positives = 0
        test_count = 5000
        for _ in range(test_count):
            non_item = "".join(random.choices(string.ascii_lowercase, k=16))
            if non_item in cf:
                false_positives += 1

        fp_rate = false_positives / test_count
        assert fp_rate < 0.05, f"False positive rate {fp_rate:.4f} too high"

    def test_clear(self):
        cf = CuckooFilter(capacity=100)
        cf.add("test")
        cf.clear()
        assert len(cf) == 0
        assert "test" not in cf

    def test_bytes_input(self):
        cf = CuckooFilter(capacity=100)
        cf.add(b"binary data")
        assert b"binary data" in cf

    def test_int_input(self):
        cf = CuckooFilter(capacity=100)
        cf.add(42)
        assert 42 in cf

    def test_multiple_inserts_and_removes(self):
        cf = CuckooFilter(capacity=100, error_rate=0.001, fingerprint_bits=16)
        items = [f"item_{i}" for i in range(50)]
        for item in items:
            cf.add(item)
        for item in items:
            assert item in cf
        for item in items[:25]:
            cf.remove(item)
        for item in items[:25]:
            assert item not in cf
        for item in items[25:]:
            assert item in cf

    def test_repr(self):
        cf = CuckooFilter(capacity=100)
        r = repr(cf)
        assert "CuckooFilter" in r
