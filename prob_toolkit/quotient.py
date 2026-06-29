"""Quotient Filter implementation.

A quotient hash table is a cache-friendly alternative to Bloom filters
that supports insertion, deletion, and membership testing. It uses
quotienting (division/modulo) for hashing, keeping related entries
adjacent in memory for better cache performance.

Based on the paper by Binns et al. (2006).

Key properties:
- Space efficient (similar to Bloom filters)
- Supports deletion via tombstones
- Better cache locality than standard hash tables
- Clustered storage with metadata bits per slot
"""

import math
import struct
from typing import Any

from prob_toolkit.hashing import HashFunction, murmur3_32


class QuotientFilter:
    """Quotient filter for probabilistic membership testing.

    A quotient hash table that stores fingerprints using quotient
    hashing. Each fingerprint is split into a quotient (bucket index)
    and remainder (stored in the slot).

    The filter maintains clusters of occupied slots, with metadata
    bits (is_occupied, is_continuation, is_shifted) per slot.

    Args:
        capacity: Expected number of elements.
        fingerprint_bits: Bits per fingerprint (default 16).
            Higher = lower false positive rate.
        hash_func: Hash function to use.

    Example:
        >>> qf = QuotientFilter(capacity=1000)
        >>> qf.add("hello")
        >>> qf.add("world")
        >>> "hello" in qf  # True
        >>> qf.remove("hello")
        >>> "hello" in qf  # False
    """

    def __init__(
        self,
        capacity: int = 1000,
        fingerprint_bits: int = 16,
        hash_func: HashFunction | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (1 <= fingerprint_bits <= 32):
            raise ValueError("fingerprint_bits must be between 1 and 32")

        self._capacity = capacity
        self._fingerprint_bits = fingerprint_bits
        self._fingerprint_mask = (1 << fingerprint_bits) - 1
        self._hash_func = hash_func or murmur3_32
        self._count = 0

        # Choose number of buckets (power of 2 for efficient modulo)
        num_buckets = max(1, self._next_power_of_2(capacity))
        self._num_buckets = num_buckets
        self._bucket_bits = int(math.log2(num_buckets))
        self._bucket_mask = num_buckets - 1

        # Each slot stores: fingerprint (remainder) + metadata bits
        # Metadata: is_occupied, is_continuation, is_shifted, is_deleted
        self._slots = [0] * num_buckets
        self._occupied = [False] * num_buckets
        self._continuation = [False] * num_buckets
        self._shifted = [False] * num_buckets
        self._deleted = [False] * num_buckets

    @staticmethod
    def _next_power_of_2(n: int) -> int:
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

    def _hash_to_quotient_remainder(self, data: bytes) -> tuple[int, int]:
        """Hash data and split into quotient (bucket) and remainder (fingerprint)."""
        h = self._hash_func(data)
        remainder = h & self._fingerprint_mask
        quotient = (h >> self._fingerprint_bits) & self._bucket_mask
        return quotient, remainder

    def add(self, item: Any) -> bool:
        """Add an item to the quotient filter.

        Args:
            item: Item to add.

        Returns:
            True if successfully added.
            False if the filter is too full.
        """
        data = self._to_bytes(item)
        quotient, remainder = self._hash_to_quotient_remainder(data)

        # Check if already present
        if self._check_present(quotient, remainder):
            return True  # Already present

        # Find the insertion point
        # Scan from the quotient bucket position
        start = quotient
        pos = start

        # Find an empty slot
        empty_slot = -1
        scan_count = 0
        while scan_count < self._num_buckets:
            if not self._occupied[pos] and not self._shifted[pos]:
                empty_slot = pos
                break
            pos = (pos + 1) % self._num_buckets
            scan_count += 1

        if empty_slot == -1:
            return False  # Filter is full

        # Insert at the empty slot
        self._slots[empty_slot] = remainder
        self._occupied[empty_slot] = True
        self._shifted[empty_slot] = (empty_slot != quotient)
        self._continuation[empty_slot] = False
        self._deleted[empty_slot] = False

        self._count += 1
        return True

    def _check_present(self, quotient: int, remainder: int) -> bool:
        """Check if a fingerprint is already in the filter.

        Simple approach: scan the cluster starting from the quotient bucket
        and look for a matching fingerprint.
        """
        pos = quotient
        scan_count = 0

        # Scan forward from the home bucket
        while scan_count < self._num_buckets:
            if (
                self._occupied[pos]
                and not self._deleted[pos]
                and self._slots[pos] == remainder
            ):
                return True
            # If we hit an empty, non-shifted, non-deleted slot, we've left the cluster
            if not self._occupied[pos] and not self._shifted[pos] and not self._deleted[pos]:
                break
            pos = (pos + 1) % self._num_buckets
            scan_count += 1
            if pos == quotient:
                break  # Wrapped around

        return False

    def remove(self, item: Any) -> bool:
        """Remove an item from the quotient filter.

        Uses tombstone deletion to mark slots as deleted without
        breaking the cluster structure.

        Args:
            item: Item to remove.

        Returns:
            True if the item was likely present and removed.
            False if the item was definitely not present.
        """
        data = self._to_bytes(item)
        quotient, remainder = self._hash_to_quotient_remainder(data)

        pos = quotient
        scan_count = 0

        while scan_count < self._num_buckets:
            if (
                self._occupied[pos]
                and not self._deleted[pos]
                and self._slots[pos] == remainder
            ):
                # Found it - mark as deleted
                self._deleted[pos] = True
                self._occupied[pos] = False
                self._count -= 1
                return True
            if not self._occupied[pos] and not self._shifted[pos]:
                break
            pos = (pos + 1) % self._num_buckets
            scan_count += 1
            if pos == quotient:
                break

        return False

    def contains(self, item: Any) -> bool:
        """Check if an item might be in the filter."""
        data = self._to_bytes(item)
        quotient, remainder = self._hash_to_quotient_remainder(data)
        return self._check_present(quotient, remainder)

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        return self._count

    @property
    def fill_ratio(self) -> float:
        """Fraction of occupied slots."""
        occupied = sum(1 for o in self._occupied if o)
        return occupied / self._num_buckets

    @property
    def bits_per_element(self) -> float:
        """Average bits used per element."""
        total_bits = self._num_buckets * (self._fingerprint_bits + 4)  # +4 for metadata
        return total_bits / self._count if self._count > 0 else 0

    @property
    def estimated_error_rate(self) -> float:
        """Estimated false positive rate based on load factor."""
        load = self.fill_ratio
        return min(1.0, (2 ** (-self._fingerprint_bits)) * load * 2)

    def clear(self) -> None:
        """Remove all elements."""
        self._slots = [0] * self._num_buckets
        self._occupied = [False] * self._num_buckets
        self._continuation = [False] * self._num_buckets
        self._shifted = [False] * self._num_buckets
        self._deleted = [False] * self._num_buckets
        self._count = 0

    def __repr__(self) -> str:
        return (
            f"QuotientFilter(capacity={self._capacity}, "
            f"fingerprint_bits={self._fingerprint_bits}, "
            f"num_buckets={self._num_buckets})"
        )
