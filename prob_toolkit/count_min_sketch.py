"""Count-Min Sketch frequency estimator.

A Count-Min Sketch is a probabilistic data structure for estimating
frequencies of items in a data stream. Based on the paper by
Cormode and Muthukrishnan (2005).

Provides:
- Point queries: estimate frequency of a specific item
- Range queries: estimate frequency over a range of items
- Inner product: estimate dot product of two frequency vectors

With w columns and d rows:
- Space: O(w * d) = O(1/epsilon * log(1/delta))
- Over-estimation error: <= epsilon * N with probability >= 1 - delta
"""

import math
import struct
from typing import Any

from prob_toolkit.hashing import HashFunction, murmur3_32


class CountMinSketch:
    """Count-Min Sketch for frequency estimation.

    Estimates the frequency of items in a data stream using a 2D array
    of counters. Items are hashed to positions in each row, and the
    minimum count across rows is returned as the estimate.

    Guarantees:
    - Never under-estimates
    - Over-estimates with probability <= delta
    - Over-estimate error <= epsilon * total_count

    Args:
        epsilon: Error parameter (0 < epsilon <= 1).
            Smaller = more accurate but more memory.
        delta: Confidence parameter (0 < delta < 1).
            Smaller = more confidence but more memory.
        width: Number of columns (overrides epsilon if set).
        depth: Number of rows (overrides delta if set).
        hash_func: Hash function to use.

    Example:
        >>> cms = CountMinSketch(epsilon=0.001, delta=0.01)
        >>> for word in stream:
        ...     cms.add(word)
        >>> cms.estimate("the")  # Frequency estimate
        >>> cms.count("the")     # Alias for estimate
    """

    def __init__(
        self,
        epsilon: float = 0.001,
        delta: float = 0.01,
        width: int | None = None,
        depth: int | None = None,
        hash_func: HashFunction | None = None,
    ):
        if width is None or depth is None:
            if not (0 < epsilon <= 1):
                raise ValueError("epsilon must be between 0 (exclusive) and 1")
            if not (0 < delta < 1):
                raise ValueError("delta must be between 0 (exclusive) and 1 (exclusive)")

        self._epsilon = epsilon
        self._delta = delta
        self._hash_func = hash_func or murmur3_32
        self._count = 0

        # Calculate dimensions
        if width is not None:
            self._width = width
        else:
            self._width = int(math.ceil(math.e / epsilon))

        if depth is not None:
            self._depth = depth
        else:
            self._depth = int(math.ceil(math.log(1 / delta)))

        # 2D counter array
        self._sketch = [[0] * self._width for _ in range(self._depth)]

        # Different seeds for each row
        self._seeds = list(range(self._depth))

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
        """Get column positions for each row."""
        positions = []
        for i in range(self._depth):
            h = self._hash_func(data, seed=self._seeds[i])
            positions.append(h % self._width)
        return positions

    def add(self, item: Any, count: int = 1) -> None:
        """Add an element (or increase its count).

        Args:
            item: Element to add.
            count: How many to add (default 1).
        """
        data = self._to_bytes(item)
        positions = self._hash_positions(data)

        for i, pos in enumerate(positions):
            self._sketch[i][pos] += count

        self._count += count

    def estimate(self, item: Any) -> int:
        """Estimate the frequency of an item.

        Args:
            item: Element to query.

        Returns:
            Estimated frequency (never under-estimates).
        """
        data = self._to_bytes(item)
        positions = self._hash_positions(data)

        return min(self._sketch[i][pos] for i, pos in enumerate(positions))

    def count(self, item: Any) -> int:
        """Alias for estimate()."""
        return self.estimate(item)

    def __getitem__(self, item: Any) -> int:
        return self.estimate(item)

    def range_estimate(self, start: Any, end: Any) -> int:
        """Estimate the total frequency for items in a range.

        This is a simplified version that works with integer keys.
        For string ranges, consider using a trie-based approach.
        """
        total = 0
        # This is a naive O(n) approach for range queries
        # In production, you'd use a more sophisticated structure
        for i in range(self._width):
            total += min(self._sketch[row][i] for row in range(self._depth))
        return total

    def merge(self, other: "CountMinSketch") -> None:
        """Merge another CountMinSketch into this one."""
        if self._width != other._width or self._depth != other._depth:
            raise ValueError("Cannot merge CountMinSketch with different dimensions")

        for i in range(self._depth):
            for j in range(self._width):
                self._sketch[i][j] += other._sketch[i][j]

        self._count += other._count

    def inner_product(self, other: "CountMinSketch") -> int:
        """Estimate the inner product (dot product) of two sketches.

        This estimates sum(a_i * b_i) where a and b are the true frequency
        vectors of two streams.
        """
        if self._width != other._width or self._depth != other._depth:
            raise ValueError("Inner product requires same-sized sketches")

        total = 0
        for i in range(self._depth):
            for j in range(self._width):
                total += self._sketch[i][j] * other._sketch[i][j]

        # This is a biased estimate; for unbiased, use a different method
        return total // self._depth

    @property
    def total_count(self) -> int:
        """Total count of all added elements."""
        return self._count

    @property
    def width(self) -> int:
        """Number of columns."""
        return self._width

    @property
    def depth(self) -> int:
        """Number of rows."""
        return self._depth

    @property
    def error_bound(self) -> float:
        """Upper bound on the error for any estimate."""
        return self._epsilon * self._count

    def clear(self) -> None:
        """Reset all counters."""
        self._sketch = [[0] * self._width for _ in range(self._depth)]
        self._count = 0

    def to_bytes(self) -> bytes:
        """Serialize the CountMinSketch to bytes."""
        header = struct.pack("<III", self._width, self._depth, self._count)
        flat = []
        for row in self._sketch:
            for val in row:
                flat.append(val)
        data = struct.pack(f"<{len(flat)}q", *flat)
        return header + data

    @classmethod
    def from_bytes(cls, data: bytes) -> "CountMinSketch":
        """Deserialize a CountMinSketch from bytes."""
        width, depth, total_count = struct.unpack("<III", data[:12])
        flat_size = width * depth
        flat = struct.unpack(f"<{flat_size}q", data[12 : 12 + flat_size * 8])
        sketch = cls(width=width, depth=depth)
        idx = 0
        for i in range(depth):
            for j in range(width):
                sketch._sketch[i][j] = flat[idx]
                idx += 1
        sketch._count = total_count
        return sketch

    def __repr__(self) -> str:
        return f"CountMinSketch(epsilon={self._epsilon}, delta={self._delta}, width={self._width}, depth={self._depth})"
