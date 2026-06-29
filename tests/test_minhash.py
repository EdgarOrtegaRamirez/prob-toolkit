"""Tests for MinHash."""

import pytest

from prob_toolkit.minhash import MinHash, WeightedMinHash


class TestMinHash:
    def test_identical_sets(self):
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)
        items = ["a", "b", "c", "d", "e"]
        for item in items:
            mh1.add(item)
            mh2.add(item)
        assert mh1.jaccard_similarity(mh2) == 1.0

    def test_disjoint_sets(self):
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)
        for item in ["a", "b", "c"]:
            mh1.add(item)
        for item in ["x", "y", "z"]:
            mh2.add(item)
        # Should be close to 0 (some false matches possible)
        sim = mh1.jaccard_similarity(mh2)
        assert sim < 0.3, f"Similarity {sim} too high for disjoint sets"

    def test_known_overlap(self):
        """Test with known Jaccard similarity."""
        mh1 = MinHash(num_hashes=256)
        mh2 = MinHash(num_hashes=256)

        set_a = set(range(100))
        set_b = set(range(50, 150))

        for item in set_a:
            mh1.add(item)
        for item in set_b:
            mh2.add(item)

        # True Jaccard: |50-149| / |0-149| = 100 / 150 = 0.667
        true_jaccard = len(set_a & set_b) / len(set_a | set_b)
        estimated = mh1.jaccard_similarity(mh2)

        # Should be reasonably close
        assert abs(estimated - true_jaccard) < 0.15, (
            f"Estimated {estimated:.3f} too far from true {true_jaccard:.3f}"
        )

    def test_distance(self):
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)
        for item in ["a", "b", "c"]:
            mh1.add(item)
            mh2.add(item)
        assert mh1.distance(mh2) == 0.0

    def test_merge(self):
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)
        mh1.add("a")
        mh1.add("b")
        mh2.add("c")
        mh2.add("d")

        mh1.merge(mh2)
        # After merge, should contain all items
        assert mh1.jaccard_similarity(mh1) == 1.0

    def test_merge_incompatible(self):
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=256)
        with pytest.raises(ValueError):
            mh1.merge(mh2)

    def test_add_many(self):
        mh = MinHash(num_hashes=128)
        mh.add_many(["a", "b", "c", "d"])
        mh2 = MinHash(num_hashes=128)
        mh2.add("a")
        mh2.add("b")
        mh2.add("c")
        mh2.add("d")
        assert mh.jaccard_similarity(mh2) == 1.0

    def test_clear(self):
        mh = MinHash(num_hashes=128)
        mh.add("item")
        mh.clear()
        mh2 = MinHash(num_hashes=128)
        assert mh.jaccard_similarity(mh2) == 1.0

    def test_signature_length(self):
        mh = MinHash(num_hashes=128)
        assert len(mh.signature) == 128

    def test_invalid_num_hashes(self):
        with pytest.raises(ValueError):
            MinHash(num_hashes=0)
        with pytest.raises(ValueError):
            MinHash(num_hashes=-1)

    def test_bytes_input(self):
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)
        mh1.add(b"hello")
        mh2.add(b"hello")
        assert mh1.jaccard_similarity(mh2) == 1.0

    def test_int_input(self):
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)
        mh1.add(42)
        mh2.add(42)
        assert mh1.jaccard_similarity(mh2) == 1.0

    def test_serialization_roundtrip(self):
        mh = MinHash(num_hashes=128)
        for item in ["a", "b", "c", "d", "e"]:
            mh.add(item)

        data = mh.to_bytes()
        mh2 = MinHash.from_bytes(data)

        assert mh2.num_hashes == mh.num_hashes
        assert mh.signature == mh2.signature
        assert mh.jaccard_similarity(mh2) == 1.0

    def test_repr(self):
        mh = MinHash(num_hashes=128)
        r = repr(mh)
        assert "MinHash" in r
        assert "num_hashes=128" in r

    def test_performance_large_sets(self):
        """Test with large sets to verify performance."""
        mh1 = MinHash(num_hashes=64)
        mh2 = MinHash(num_hashes=64)

        # Add 10000 items to each set
        for i in range(10000):
            mh1.add(f"set_a_{i}")
        for i in range(10000):
            mh2.add(f"set_b_{i}")

        # Very different sets, similarity should be low
        sim = mh1.jaccard_similarity(mh2)
        assert sim < 0.1


class TestWeightedMinHash:
    def test_basic(self):
        wmh1 = WeightedMinHash(num_hashes=128)
        wmh2 = WeightedMinHash(num_hashes=128)
        wmh1.add("item", weight=1.0)
        wmh2.add("item", weight=1.0)
        sim = wmh1.jaccard_similarity(wmh2)
        assert sim > 0.5  # Should be similar

    def test_different_weights(self):
        wmh1 = WeightedMinHash(num_hashes=128)
        wmh1.add("a", weight=1.0)
        wmh1.add("b", weight=2.0)
        assert len(wmh1.signature) == 128

    def test_zero_weight_ignored(self):
        wmh = WeightedMinHash(num_hashes=128)
        wmh.add("item", weight=0.0)
        # Signature should still be default
        assert all(s == (2**64 - 1, 2**64 - 1) for s in wmh.signature)

    def test_repr(self):
        wmh = WeightedMinHash(num_hashes=128)
        r = repr(wmh)
        assert "WeightedMinHash" in r
