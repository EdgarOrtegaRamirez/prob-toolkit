"""HyperLogLog cardinality estimator.

HyperLogLog is a probabilistic algorithm for counting distinct elements
in a multiset (cardinality estimation) with high accuracy and minimal
memory usage. Based on the paper by Flajolet et al. (2007).

Provides ~1.04 / sqrt(m) standard error where m = 2^precision registers.
With 16KB of memory (precision=14), achieves ~0.8% error rate.
"""

import math
import struct
from typing import Any

from prob_toolkit.hashing import HashFunction, xxhash_64


class HyperLogLog:
    """HyperLogLog cardinality estimator.

    Estimates the number of distinct elements in a stream using
    a compact data structure. Works by observing the leading zeros
    in hash values of elements.

    The standard error is approximately 1.04 / sqrt(2^precision).

    Args:
        precision: Number of bits for bucket selection (4-18).
            Determines the number of registers = 2^precision.
            - precision=10: ~3.3% error, ~1KB memory
            - precision=14: ~0.8% error, ~16KB memory
            - precision=16: ~0.4% error, ~64KB memory
        hash_func: Hash function to use (must return 64-bit hash).

    Example:
        >>> hll = HyperLogLog(precision=14)
        >>> for i in range(100000):
        ...     hll.add(f"element_{i}")
        >>> print(f"Estimated: {hll.estimate()}")  # ~100000
        >>> print(f"Error: {abs(hll.estimate() - 100000) / 100000:.2%}")
    """

    def __init__(
        self,
        precision: int = 14,
        hash_func: HashFunction | None = None,
    ):
        if not (4 <= precision <= 18):
            raise ValueError("precision must be between 4 and 18")

        self._precision = precision
        self._num_buckets = 1 << precision
        self._bucket_mask = self._num_buckets - 1
        # Use 64-bit hash by default for HyperLogLog
        self._hash_func = hash_func or xxhash_64

        # Registers: store the maximum number of leading zeros + 1
        # This is the "rho" value from the paper
        self._registers = [0] * self._num_buckets

    def _to_bytes(self, item: Any) -> bytes:
        if isinstance(item, bytes):
            return item
        elif isinstance(item, str):
            return item.encode("utf-8")
        elif isinstance(item, int):
            return struct.pack("<q", item)
        else:
            return str(item).encode("utf-8")

    def _leading_zeros(self, hash_value: int) -> int:
        """Count leading zeros in the bits above the bucket selection bits.

        We have a 64-bit hash. The lower `precision` bits select the bucket.
        The remaining (64 - precision) bits are used for leading zero count.
        """
        remaining_bits = 64 - self._precision

        # Shift out the bucket selection bits (lower bits)
        shifted = hash_value >> self._precision

        # Mask to only keep remaining_bits bits
        shifted = shifted & ((1 << remaining_bits) - 1)

        if shifted == 0:
            return remaining_bits + 1  # All zeros -> max rho

        # Count leading zeros
        count = 0
        mask = 1 << (remaining_bits - 1)
        while (shifted & mask) == 0:
            count += 1
            mask >>= 1

        return count + 1  # rho = number of leading zeros + 1

    def add(self, item: Any) -> None:
        """Add an element to the HyperLogLog.

        Args:
            item: Element to add. Converted to bytes.
        """
        data = self._to_bytes(item)
        hash_value = self._hash_func(data)

        # Use lower bits for bucket selection
        bucket = hash_value & self._bucket_mask

        # Use remaining bits for leading zero count
        rho = self._leading_zeros(hash_value)

        # Update register with maximum rho seen
        if rho > self._registers[bucket]:
            self._registers[bucket] = rho

    def merge(self, other: "HyperLogLog") -> None:
        """Merge another HyperLogLog into this one (take element-wise max)."""
        if self._precision != other._precision:
            raise ValueError("Cannot merge HyperLogLog with different precision")

        for i in range(self._num_buckets):
            if other._registers[i] > self._registers[i]:
                self._registers[i] = other._registers[i]

    def estimate(self) -> float:
        """Estimate the cardinality (number of distinct elements).

        Returns:
            Estimated number of distinct elements.
        """
        m = self._num_buckets
        raw_estimate = self._raw_estimate()

        # Use different estimation methods based on register state
        zeros = sum(1 for r in self._registers if r == 0)

        # Linear counting for small cardinalities (when many zeros)
        if zeros > m * 0.7:
            if zeros == m:
                return 0.0
            return m * math.log(m / zeros)

        # Small range correction (when raw estimate is close to m)
        if raw_estimate <= 2.5 * m and zeros > 0:
            # Use linear counting + raw estimate (bias-corrected)
            linear_est = m * math.log(m / zeros)
            return linear_est

        # Large range correction (near overflow of 64-bit)
        if raw_estimate > (1 << 64) / 30.0:
            return -(2 ** 64) * math.log(1 - raw_estimate / (2 ** 64))

        return raw_estimate

    def _raw_estimate(self) -> float:
        """Compute the raw HyperLogLog estimate using harmonic mean."""
        m = self._num_buckets
        total = 0.0

        for j in range(m):
            # Sum of 2^(-M[j]) for all registers
            total += 2.0 ** (-self._registers[j])

        # Alpha_m * m^2 / sum(2^(-M[j]))
        alpha_m = self._alpha(m)
        if total == 0:
            return 0.0
        return alpha_m * m * m / total

    def _alpha(self, m: int) -> float:
        """Bias correction constant alpha_m.

        Based on the table from the HyperLogLog paper.
        """
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)

    @property
    def precision(self) -> int:
        """The precision parameter."""
        return self._precision

    @property
    def num_buckets(self) -> int:
        """Number of registers (2^precision)."""
        return self._num_buckets

    @property
    def registers(self) -> list[int]:
        """Copy of the register array."""
        return list(self._registers)

    def clear(self) -> None:
        """Reset all registers."""
        self._registers = [0] * self._num_buckets

    def to_bytes(self) -> bytes:
        """Serialize the HyperLogLog to bytes."""
        header = struct.pack("<I", self._precision)
        # Pack registers (each is 1 byte since max is 64)
        regs = bytes(self._registers)
        return header + regs

    @classmethod
    def from_bytes(cls, data: bytes) -> "HyperLogLog":
        """Deserialize a HyperLogLog from bytes."""
        precision = struct.unpack("<I", data[:4])[0]
        hll = cls(precision=precision)
        hll._registers = list(data[4:])
        return hll

    def __repr__(self) -> str:
        return (
            f"HyperLogLog(precision={self._precision}, "
            f"num_buckets={self._num_buckets}, "
            f"estimate={self.estimate():.0f})"
        )
