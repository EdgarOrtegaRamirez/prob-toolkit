"""Bloom filter implementations.

Provides standard, counting, and scalable Bloom filters for efficient
probabilistic membership testing.
"""

import math
import struct
from typing import Any

from prob_toolkit.hashing import HashFunction, fnv1a_32, make_double_hash, murmur3_32


class BloomFilter:
    """Standard Bloom filter for probabilistic membership testing.

    A Bloom filter uses a bit array and multiple hash functions to test
    whether an element is a member of a set. False positives are possible,
    but false negatives are not.

    Space-optimal parameters:
        - For n elements and false positive rate p:
            m = -n * ln(p) / (ln(2))^2 bits
            k = (m/n) * ln(2) hash functions

    Args:
        capacity: Expected number of elements.
        error_rate: Desired false positive rate (0.0 to 1.0).
        hash_func: Primary hash function (default: murmur3_32).
        seed: Initial seed for hash function.

    Example:
        >>> bf = BloomFilter(capacity=1000, error_rate=0.01)
        >>> bf.add("hello")
        >>> bf.add("world")
        >>> "hello" in bf  # True
        >>> "goodbye" in bf  # False (usually)
        >>> bf.estimated_elements  # ~2
    """

    def __init__(
        self,
        capacity: int = 10000,
        error_rate: float = 0.001,
        hash_func: HashFunction | None = None,
        seed: int = 0,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0.0 < error_rate < 1.0):
            raise ValueError("error_rate must be between 0 and 1 (exclusive)")

        self._capacity = capacity
        self._error_rate = error_rate
        self._hash_func = hash_func or murmur3_32
        self._seed = seed
        self._count = 0

        # Optimal parameters
        self._num_hashes = self._optimal_num_hashes(capacity, error_rate)
        self._bit_size = self._optimal_bit_size(capacity, error_rate)
        self._bit_array = bytearray((self._bit_size + 7) // 8)

        # Double hash for generating multiple hash positions
        self._double_hash = make_double_hash(
            murmur3_32, fnv1a_32
        )

    @staticmethod
    def _optimal_bit_size(n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return max(1, int(math.ceil(m)))

    @staticmethod
    def _optimal_num_hashes(n: int, p: float) -> int:
        """Calculate optimal number of hash functions."""
        m = BloomFilter._optimal_bit_size(n, p)
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))

    def _get_bit(self, index: int) -> bool:
        """Get bit at position."""
        byte_index = index // 8
        bit_offset = index % 8
        return bool(self._bit_array[byte_index] & (1 << bit_offset))

    def _set_bit(self, index: int) -> None:
        """Set bit at position."""
        byte_index = index // 8
        bit_offset = index % 8
        self._bit_array[byte_index] |= 1 << bit_offset

    def _hash_positions(self, data: bytes) -> list[int]:
        """Get bit positions for the given data."""
        h = self._double_hash
        positions = []
        for i in range(self._num_hashes):
            pos = h(data, i + self._seed) % self._bit_size
            positions.append(pos)
        return positions

    def add(self, item: Any) -> None:
        """Add an item to the Bloom filter.

        Args:
            item: Item to add. Converted to bytes via str/bytes.
        """
        data = self._to_bytes(item)
        for pos in self._hash_positions(data):
            self._set_bit(pos)
        self._count += 1

    def contains(self, item: Any) -> bool:
        """Check if an item might be in the filter.

        Args:
            item: Item to check.

        Returns:
            True if item is probably in the set (may be false positive).
            False if item is definitely not in the set.
        """
        data = self._to_bytes(item)
        return all(self._get_bit(pos) for pos in self._hash_positions(data))

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        return self._count

    def _to_bytes(self, item: Any) -> bytes:
        if isinstance(item, bytes):
            return item
        elif isinstance(item, str):
            return item.encode("utf-8")
        elif isinstance(item, int):
            return struct.pack("<q", item)
        else:
            return str(item).encode("utf-8")

    @property
    def estimated_elements(self) -> int:
        """Estimate the number of elements in the filter."""
        # Count zeros in bit array
        zeros = sum(1 for byte in self._bit_array for i in range(8) if not (byte & (1 << i)))
        total = self._bit_size
        if zeros == 0:
            return self._capacity
        # Inverse of the false positive formula
        m = total
        k = self._num_hashes
        return int(-m / k * math.log(zeros / m))

    @property
    def current_error_rate(self) -> float:
        """Estimate the current false positive rate."""
        # (1 - e^(-kn/m))^k
        n = self._count
        m = self._bit_size
        k = self._num_hashes
        if m == 0 or k == 0:
            return 0.0
        exponent = -k * n / m
        return (1 - math.exp(exponent)) ** k

    @property
    def fill_ratio(self) -> float:
        """Fraction of bits set to 1."""
        set_bits = sum(1 for byte in self._bit_array for i in range(8) if byte & (1 << i))
        return set_bits / self._bit_size

    @property
    def num_hashes(self) -> int:
        """Number of hash functions used."""
        return self._num_hashes

    @property
    def bit_size(self) -> int:
        """Size of the bit array."""
        return self._bit_size

    def clear(self) -> None:
        """Remove all elements."""
        self._bit_array = bytearray((self._bit_size + 7) // 8)
        self._count = 0

    def merge(self, other: "BloomFilter") -> None:
        """Merge another Bloom filter into this one (OR their bit arrays).

        Both filters must have the same parameters.
        """
        if self._bit_size != other._bit_size:
            raise ValueError("Cannot merge Bloom filters with different sizes")
        if self._num_hashes != other._num_hashes:
            raise ValueError("Cannot merge Bloom filters with different hash counts")

        for i in range(len(self._bit_array)):
            self._bit_array[i] |= other._bit_array[i]
        self._count = max(self._count, other._count)

    def intersection(self, other: "BloomFilter") -> "BloomFilter":
        """Create a new Bloom filter that is the intersection (AND) of two filters."""
        if self._bit_size != other._bit_size:
            raise ValueError("Cannot intersect Bloom filters with different sizes")

        result = BloomFilter.__new__(BloomFilter)
        result._capacity = self._capacity
        result._error_rate = self._error_rate
        result._hash_func = self._hash_func
        result._seed = self._seed
        result._num_hashes = self._num_hashes
        result._bit_size = self._bit_size
        result._bit_array = bytearray(
            a & b for a, b in zip(self._bit_array, other._bit_array, strict=True)
        )
        result._count = min(self._count, other._count)
        result._double_hash = make_double_hash(murmur3_32, fnv1a_32)
        return result

    def to_bytes(self) -> bytes:
        """Serialize the Bloom filter to bytes."""
        header = struct.pack("<III", self._capacity, self._bit_size, self._num_hashes)
        return header + bytes(self._bit_array)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        """Deserialize a Bloom filter from bytes."""
        capacity, bit_size, num_hashes = struct.unpack("<III", data[:12])
        bit_array = bytearray(data[12:])
        bf = cls.__new__(cls)
        bf._capacity = capacity
        bf._error_rate = 0.001  # Not stored, use default
        bf._hash_func = murmur3_32
        bf._seed = 0
        bf._num_hashes = num_hashes
        bf._bit_size = bit_size
        bf._bit_array = bit_array
        bf._count = 0  # Not tracked in serialization
        bf._double_hash = make_double_hash(murmur3_32, fnv1a_32)
        return bf

    def __repr__(self) -> str:
        return (
            f"BloomFilter(capacity={self._capacity}, error_rate={self._error_rate}, "
            f"num_hashes={self._num_hashes}, bit_size={self._bit_size})"
        )


class CountingBloomFilter:
    """Bloom filter that supports deletion using counters.

    Instead of a single bit, each position uses a small counter (typically
    4 bits / a nibble) to track the number of items mapping to it. This
    allows safe deletion of items.

    Args:
        capacity: Expected number of elements.
        error_rate: Desired false positive rate.
        counter_bits: Bits per counter (default 4, supports up to 15).
    """

    def __init__(
        self,
        capacity: int = 10000,
        error_rate: float = 0.001,
        counter_bits: int = 4,
    ):
        if counter_bits not in (4, 8):
            raise ValueError("counter_bits must be 4 or 8")

        self._capacity = capacity
        self._error_rate = error_rate
        self._counter_bits = counter_bits
        self._count = 0

        self._num_hashes = BloomFilter._optimal_num_hashes(capacity, error_rate)
        self._bit_size = BloomFilter._optimal_bit_size(capacity, error_rate)
        self._max_counter = (1 << counter_bits) - 1

        # Store counters as bytes
        self._counters = bytearray(self._bit_size)
        self._double_hash = make_double_hash(murmur3_32, fnv1a_32)

    def _to_bytes(self, item: Any) -> bytes:
        if isinstance(item, bytes):
            return item
        elif isinstance(item, str):
            return item.encode("utf-8")
        elif isinstance(item, int):
            return struct.pack("<q", item)
        else:
            return str(item).encode("utf-8")

    def _hash_positions(self, data: bytes) -> list[int]:
        h = self._double_hash
        return [h(data, i) % self._bit_size for i in range(self._num_hashes)]

    def add(self, item: Any) -> None:
        """Add an item to the counting Bloom filter."""
        data = self._to_bytes(item)
        for pos in self._hash_positions(data):
            if self._counters[pos] < self._max_counter:
                self._counters[pos] += 1
        self._count += 1

    def remove(self, item: Any) -> bool:
        """Remove an item from the counting Bloom filter.

        Returns:
            True if the item was likely present and removed.
            False if the item was definitely not present.
        """
        data = self._to_bytes(item)
        positions = self._hash_positions(data)

        # Check all counters are non-zero
        if not all(self._counters[pos] > 0 for pos in positions):
            return False

        # Decrement all counters
        for pos in positions:
            self._counters[pos] -= 1
        self._count -= 1
        return True

    def contains(self, item: Any) -> bool:
        """Check if an item might be in the filter."""
        data = self._to_bytes(item)
        return all(self._counters[pos] > 0 for pos in self._hash_positions(data))

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        return self._count

    @property
    def fill_ratio(self) -> float:
        """Fraction of counters that are non-zero."""
        non_zero = sum(1 for c in self._counters if c > 0)
        return non_zero / len(self._counters)

    def clear(self) -> None:
        """Remove all elements."""
        self._counters = bytearray(self._bit_size)
        self._count = 0

    def __repr__(self) -> str:
        return (
            f"CountingBloomFilter(capacity={self._capacity}, "
            f"error_rate={self._error_rate}, counter_bits={self._counter_bits})"
        )


class ScalableBloomFilter:
    """Bloom filter that automatically grows as elements are added.

    Uses a sequence of Bloom filters with tightening error rates.
    When the current filter reaches capacity, a new, larger filter
    is created with a stricter error rate.

    Inspired by the paper "Scalable Bloom Filters" by Almeida et al.

    Args:
        initial_capacity: Initial expected number of elements.
        error_rate: Target false positive rate.
        growth_factor: Size multiplier for new filters.
        tightening_ratio: Error rate multiplier for new filters.
    """

    def __init__(
        self,
        initial_capacity: int = 1000,
        error_rate: float = 0.01,
        growth_factor: float = 2.0,
        tightening_ratio: float = 0.9,
    ):
        self._initial_capacity = initial_capacity
        self._error_rate = error_rate
        self._growth_factor = growth_factor
        self._tightening_ratio = tightening_ratio
        self._filters: list[BloomFilter] = []
        self._count = 0

        self._add_filter(error_rate)

    def _add_filter(self, error_rate: float) -> BloomFilter:
        """Add a new filter with the given error rate."""
        capacity = max(
            self._initial_capacity,
            int(self._initial_capacity * (self._growth_factor ** len(self._filters))),
        )
        f = BloomFilter(capacity=capacity, error_rate=error_rate)
        self._filters.append(f)
        return f

    def _current_filter(self) -> BloomFilter:
        """Get the current (most recent) filter."""
        return self._filters[-1]

    def add(self, item: Any) -> None:
        """Add an item to the scalable Bloom filter."""
        current = self._current_filter()

        # Check if we need a new filter
        if current._count >= current._capacity:
            new_error_rate = current._error_rate * self._tightening_ratio
            current = self._add_filter(new_error_rate)

        current.add(item)
        self._count += 1

    def contains(self, item: Any) -> bool:
        """Check if an item might be in any of the filters."""
        return any(f.contains(item) for f in self._filters)

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        return self._count

    @property
    def num_filters(self) -> int:
        """Number of internal Bloom filters."""
        return len(self._filters)

    @property
    def current_error_rate(self) -> float:
        """Current false positive rate (combined)."""
        # Combined FP rate is product of individual rates
        result = 1.0
        for f in self._filters:
            result *= f.current_error_rate
        return result

    @property
    def bit_size(self) -> int:
        """Total bits across all filters."""
        return sum(f.bit_size for f in self._filters)

    def clear(self) -> None:
        """Reset the filter."""
        self._filters.clear()
        self._count = 0
        self._add_filter(self._error_rate)

    def __repr__(self) -> str:
        return (
            f"ScalableBloomFilter(initial_capacity={self._initial_capacity}, "
            f"error_rate={self._error_rate}, num_filters={self.num_filters})"
        )
