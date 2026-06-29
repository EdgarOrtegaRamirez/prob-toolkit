"""Tests for hash functions."""

import struct

from prob_toolkit.hashing import (
    fnv1a_32,
    make_double_hash,
    murmur3_32,
    xxhash_64,
)


class TestMurmur3:
    def test_empty_input(self):
        assert murmur3_32(b"") == 0

    def test_deterministic(self):
        assert murmur3_32(b"hello") == murmur3_32(b"hello")

    def test_different_inputs(self):
        assert murmur3_32(b"hello") != murmur3_32(b"world")

    def test_seed_affects_output(self):
        h1 = murmur3_32(b"hello", seed=0)
        h2 = murmur3_32(b"hello", seed=1)
        assert h1 != h2

    def test_returns_32bit(self):
        h = murmur3_32(b"test data")
        assert 0 <= h <= 0xFFFFFFFF

    def test_string_input(self):
        h = murmur3_32(b"hello")
        assert isinstance(h, int)

    def test_large_input(self):
        data = b"x" * 10000
        h = murmur3_32(data)
        assert 0 <= h <= 0xFFFFFFFF

    def test_distribution(self):
        """Test that hash values are reasonably distributed."""
        counts = [0] * 16
        for i in range(16000):
            h = murmur3_32(struct.pack("<I", i))
            counts[h % 16] += 1
        # Each bucket should have roughly 1000 entries
        for count in counts:
            assert 800 < count < 1200, f"Poor distribution: {counts}"


class TestFnv1a:
    def test_empty_input(self):
        assert fnv1a_32(b"") == 0x811C9DC5

    def test_deterministic(self):
        assert fnv1a_32(b"hello") == fnv1a_32(b"hello")

    def test_different_inputs(self):
        assert fnv1a_32(b"hello") != fnv1a_32(b"world")

    def test_returns_32bit(self):
        h = fnv1a_32(b"test")
        assert 0 <= h <= 0xFFFFFFFF


class TestXxhash:
    def test_empty_input(self):
        h = xxhash_64(b"")
        assert isinstance(h, int)

    def test_deterministic(self):
        assert xxhash_64(b"hello") == xxhash_64(b"hello")

    def test_different_inputs(self):
        assert xxhash_64(b"hello") != xxhash_64(b"world")

    def test_returns_64bit(self):
        h = xxhash_64(b"test")
        assert 0 <= h <= 0xFFFFFFFFFFFFFFFF


class TestDoubleHash:
    def test_returns_different_for_different_i(self):
        dh = make_double_hash(murmur3_32, fnv1a_32)
        values = [dh(b"test", i) for i in range(10)]
        assert len(set(values)) > 1

    def test_returns_int(self):
        dh = make_double_hash(murmur3_32, fnv1a_32)
        assert isinstance(dh(b"test", 0), int)
