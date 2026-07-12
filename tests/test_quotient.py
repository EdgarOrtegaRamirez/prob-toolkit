"""Tests for Quotient Filter."""

from prob_toolkit.quotient import QuotientFilter


class TestQuotientFilter:
    def test_add_and_contains(self):
        qf = QuotientFilter(capacity=100, fingerprint_bits=16)
        qf.add("hello")
        qf.add("world")
        assert "hello" in qf
        assert "world" in qf
        assert "goodbye" not in qf

    def test_empty_filter(self):
        qf = QuotientFilter(capacity=100)
        assert "anything" not in qf

    def test_remove(self):
        qf = QuotientFilter(capacity=100, fingerprint_bits=16)
        qf.add("hello")
        assert qf.remove("hello") is True
        assert "hello" not in qf

    def test_remove_nonexistent(self):
        qf = QuotientFilter(capacity=100)
        assert qf.remove("nonexistent") is False

    def test_len(self):
        qf = QuotientFilter(capacity=100)
        assert len(qf) == 0
        qf.add("a")
        qf.add("b")
        assert len(qf) == 2

    def test_fill_ratio(self):
        qf = QuotientFilter(capacity=100)
        assert qf.fill_ratio == 0.0
        qf.add("item")
        assert qf.fill_ratio > 0.0

    def test_bits_per_element(self):
        qf = QuotientFilter(capacity=100, fingerprint_bits=16)
        qf.add("item")
        bpe = qf.bits_per_element
        assert bpe > 0

    def test_clear(self):
        qf = QuotientFilter(capacity=100)
        qf.add("test")
        qf.clear()
        assert len(qf) == 0
        assert "test" not in qf

    def test_bytes_input(self):
        qf = QuotientFilter(capacity=100)
        qf.add(b"binary")
        assert b"binary" in qf

    def test_int_input(self):
        qf = QuotientFilter(capacity=100)
        qf.add(42)
        assert 42 in qf

    def test_multiple_operations(self):
        qf = QuotientFilter(capacity=100, fingerprint_bits=16)
        items = [f"item_{i}" for i in range(20)]
        for item in items:
            qf.add(item)
        for item in items:
            assert item in qf
        for item in items[:10]:
            qf.remove(item)
        for item in items[:10]:
            assert item not in qf
        for item in items[10:]:
            assert item in qf

    def test_repr(self):
        qf = QuotientFilter(capacity=100, fingerprint_bits=16)
        r = repr(qf)
        assert "QuotientFilter" in r

    def test_invalid_capacity(self):
        import pytest

        with pytest.raises(ValueError):
            QuotientFilter(capacity=0)

    def test_invalid_fingerprint_bits(self):
        import pytest

        with pytest.raises(ValueError):
            QuotientFilter(capacity=100, fingerprint_bits=0)
        with pytest.raises(ValueError):
            QuotientFilter(capacity=100, fingerprint_bits=33)

    def test_estimated_error_rate(self):
        qf = QuotientFilter(capacity=100, fingerprint_bits=16)
        assert qf.estimated_error_rate >= 0
        qf.add("item")
        assert qf.estimated_error_rate >= 0
