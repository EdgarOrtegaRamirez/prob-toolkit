"""Cuckoo filter implementation.

Cuckoo filters offer better space efficiency than Bloom filters for
applications requiring deletion support. Based on the paper:
"Cuckoo Filter: Practically Better Than Bloom" by Fan et al. (2014).
"""

import math
import struct
from typing import Any

from prob_toolkit.hashing import HashFunction, murmur3_32

# Fingerprint size in bits (default: 8 bits = 1 byte)
DEFAULT_FINGERPRINT_BITS = 8
MAX_KICKS = 500  # Maximum number of cuckoo kicks before failure


class CuckooFilter:
    """Cuckoo filter for probabilistic membership testing with deletion support.

    A Cuckoo filter stores fingerprints in a hash table and uses cuckoo
    hashing with bucket-based placement. Each bucket holds up to `bucket_size`
    fingerprints, and insertion kicks out existing fingerprints to alternate
    locations (fingerprint XOR'd with hash).

    Key advantages over Bloom filters:
    - Supports deletion
    - Better space efficiency at low-to-moderate FP rates
    - Better cache locality

    Args:
        capacity: Expected number of elements.
        error_rate: Desired false positive rate (0.0 to 1.0).
        bucket_size: Number of fingerprints per bucket (1-4, default 4).
        fingerprint_bits: Bits per fingerprint (default 8).
        hash_func: Hash function for bucket indexing.

    Example:
        >>> cf = CuckooFilter(capacity=1000, error_rate=0.01)
        >>> cf.add("hello")
        >>> cf.add("world")
        >>> "hello" in cf  # True
        >>> cf.remove("hello")  # True
        >>> "hello" in cf  # False
    """

    def __init__(
        self,
        capacity: int = 10000,
        error_rate: float = 0.01,
        bucket_size: int = 4,
        fingerprint_bits: int = DEFAULT_FINGERPRINT_BITS,
        hash_func: HashFunction | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0.0 < error_rate < 1.0):
            raise ValueError("error_rate must be between 0 and 1 (exclusive)")
        if not (1 <= bucket_size <= 4):
            raise ValueError("bucket_size must be between 1 and 4")
        if not (1 <= fingerprint_bits <= 32):
            raise ValueError("fingerprint_bits must be between 1 and 32")

        self._capacity = capacity
        self._error_rate = error_rate
        self._bucket_size = bucket_size
        self._fingerprint_bits = fingerprint_bits
        self._fingerprint_mask = (1 << fingerprint_bits) - 1
        self._hash_func = hash_func or murmur3_32
        self._count = 0

        # Calculate optimal number of buckets
        # For a given FP rate and bucket size:
        # m = ceil(n / bucket_size * 1.09) approximately
        # With fingerprint bits b, we need:
        # error_rate ≈ 2 * bucket_size * (1/2^b) for bucket-based cuckoo
        # Using the more accurate formula from the paper
        fp_fp_rate = error_rate / (2 * bucket_size)
        self._fp_bits_needed = max(
            fingerprint_bits,
            int(math.ceil(-math.log2(fp_fp_rate))) if fp_fp_rate > 0 else fingerprint_bits,
        )
        self._fp_bits_needed = min(self._fp_bits_needed, 32)

        # Number of buckets (power of 2 for efficient modular hashing)
        num_buckets = max(1, int(math.ceil(capacity / bucket_size * 1.5)))
        self._num_buckets = self._next_power_of_2(num_buckets)
        self._bucket_index_mask = self._num_buckets - 1

        # Storage: each bucket is an array of fingerprints
        # We use a flat array of size num_buckets * bucket_size
        self._buckets = [0] * (self._num_buckets * self._bucket_size)

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """Find the next power of 2 >= n."""
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1

    def _to_bytes(self, item: Any) -> bytes:
        if isinstance(item, bytes):
            return item
        elif isinstance(item, str):
            return item.encode("utf-8")
        elif isinstance(item, int):
            return struct.pack("<q", item)
        else:
            return str(item).encode("utf-8")

    def _fingerprint(self, data: bytes) -> int:
        """Compute a compact fingerprint of the data."""
        h = self._hash_func(data, seed=42)
        return h & self._fingerprint_mask

    def _bucket_index(self, data: bytes) -> int:
        """Compute the primary bucket index."""
        h = self._hash_func(data, seed=0)
        return h & self._bucket_index_mask

    def _alt_bucket(self, bucket: int, fp: int) -> int:
        """Compute the alternate bucket index.

        Uses XOR with fingerprint for efficient computation:
        alt = bucket XOR hash(fingerprint)
        This ensures that both bucket and alt_bucket can reach each other.
        """
        h = self._hash_func(struct.pack("<I", fp), seed=1)
        return (bucket ^ (h & self._bucket_index_mask)) & self._bucket_index_mask

    def _get_bucket_fingerprints(self, bucket: int) -> list[int]:
        """Get all fingerprints in a bucket."""
        start = bucket * self._bucket_size
        return self._buckets[start : start + self._bucket_size]

    def _set_bucket_fingerprint(self, bucket: int, slot: int, fp: int) -> None:
        """Set a fingerprint in a specific bucket slot."""
        self._buckets[bucket * self._bucket_size + slot] = fp

    def _find_in_bucket(self, bucket: int, fp: int) -> int:
        """Find a fingerprint in a bucket. Returns slot index or -1."""
        start = bucket * self._bucket_size
        for i in range(self._bucket_size):
            if self._buckets[start + i] == fp:
                return i
        return -1

    def _find_empty_slot(self, bucket: int) -> int:
        """Find an empty slot in a bucket. Returns slot index or -1."""
        start = bucket * self._bucket_size
        for i in range(self._bucket_size):
            if self._buckets[start + i] == 0:
                return i
        return -1

    def add(self, item: Any) -> bool:
        """Add an item to the Cuckoo filter.

        Args:
            item: Item to add.

        Returns:
            True if successfully added.
            False if insertion would exceed the maximum kick limit
            (filter is too full or has high collision rate).
        """
        data = self._to_bytes(item)
        fp = self._fingerprint(data)
        if fp == 0:
            fp = 1  # Avoid 0 sentinel (used for empty slots)
        b1 = self._bucket_index(data)

        # Try to insert in primary bucket
        slot = self._find_empty_slot(b1)
        if slot >= 0:
            self._set_bucket_fingerprint(b1, slot, fp)
            self._count += 1
            return True

        # Try alternate bucket
        b2 = self._alt_bucket(b1, fp)
        slot = self._find_empty_slot(b2)
        if slot >= 0:
            self._set_bucket_fingerprint(b2, slot, fp)
            self._count += 1
            return True

        # Cuckoo kicking: randomly evict existing fingerprints
        current_bucket = b1
        for _ in range(MAX_KICKS):
            # Randomly select a slot to evict
            slot = self._hash_func(struct.pack("<I", fp), seed=current_bucket) % self._bucket_size
            victim = self._buckets[current_bucket * self._bucket_size + slot]

            # Replace with new fingerprint
            self._set_bucket_fingerprint(current_bucket, slot, fp)
            fp = victim

            # Move evicted fingerprint to its alternate bucket
            current_bucket = self._alt_bucket(current_bucket, fp)

            # Try to insert in the new bucket
            slot = self._find_empty_slot(current_bucket)
            if slot >= 0:
                self._set_bucket_fingerprint(current_bucket, slot, fp)
                self._count += 1
                return True

        # Failed to insert after maximum kicks
        return False

    def remove(self, item: Any) -> bool:
        """Remove an item from the Cuckoo filter.

        Args:
            item: Item to remove.

        Returns:
            True if the item was likely present and removed.
            False if the item was definitely not present.
        """
        data = self._to_bytes(item)
        fp = self._fingerprint(data)
        if fp == 0:
            fp = 1
        b1 = self._bucket_index(data)
        b2 = self._alt_bucket(b1, fp)

        # Try to find and remove from primary bucket
        slot = self._find_in_bucket(b1, fp)
        if slot >= 0:
            self._set_bucket_fingerprint(b1, slot, 0)
            self._count -= 1
            return True

        # Try alternate bucket
        slot = self._find_in_bucket(b2, fp)
        if slot >= 0:
            self._set_bucket_fingerprint(b2, slot, 0)
            self._count -= 1
            return True

        return False

    def contains(self, item: Any) -> bool:
        """Check if an item might be in the filter."""
        data = self._to_bytes(item)
        fp = self._fingerprint(data)
        if fp == 0:
            fp = 1
        b1 = self._bucket_index(data)
        b2 = self._alt_bucket(b1, fp)

        return self._find_in_bucket(b1, fp) >= 0 or self._find_in_bucket(b2, fp) >= 0

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        return self._count

    @property
    def fill_ratio(self) -> float:
        """Fraction of slots that are occupied."""
        total_slots = self._num_buckets * self._bucket_size
        occupied = sum(1 for fp in self._buckets if fp != 0)
        return occupied / total_slots

    @property
    def bits_per_element(self) -> float:
        """Average bits used per element."""
        total_bits = self._num_buckets * self._bucket_size * self._fingerprint_bits
        return total_bits / self._count if self._count > 0 else 0

    def clear(self) -> None:
        """Remove all elements."""
        self._buckets = [0] * (self._num_buckets * self._bucket_size)
        self._count = 0

    def __repr__(self) -> str:
        return (
            f"CuckooFilter(capacity={self._capacity}, error_rate={self._error_rate}, "
            f"bucket_size={self._bucket_size}, fingerprint_bits={self._fingerprint_bits})"
        )
