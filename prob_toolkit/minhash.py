"""MinHash for Jaccard similarity estimation.

MinHash is a locality-sensitive hashing technique for estimating the
Jaccard similarity between sets. Based on the paper by Broder (1997).

Jaccard similarity: J(A, B) = |A ∩ B| / |A ∪ B|

MinHash estimates this by computing the minimum hash value for each set
under multiple random hash functions. The probability that the minimum
hash values are equal for two sets equals their Jaccard similarity.
"""

import hashlib
import struct
from typing import Any

from prob_toolkit.hashing import HashFunction, murmur3_32


class MinHash:
    """MinHash for estimating Jaccard similarity between sets.

    Computes k independent hash functions to produce a "signature" for each
    set. The Jaccard similarity is estimated as the fraction of matching
    hash values across the signatures.

    Args:
        num_hashes: Number of hash functions (more = more accurate).
            Standard error = 1 / sqrt(num_hashes).
        hash_func: Base hash function to use.

    Example:
        >>> set_a = MinHash(num_hashes=128)
        >>> set_b = MinHash(num_hashes=128)
        >>> for item in ["apple", "banana", "cherry"]:
        ...     set_a.add(item)
        >>> for item in ["banana", "cherry", "date"]:
        ...     set_b.add(item)
        >>> similarity = set_a.jaccard_similarity(set_b)
        >>> print(f"Estimated Jaccard similarity: {similarity:.3f}")  # ~0.5
    """

    def __init__(
        self,
        num_hashes: int = 128,
        hash_func: HashFunction | None = None,
    ):
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")

        self._num_hashes = num_hashes
        self._hash_func = hash_func or murmur3_32

        # Generate hash function parameters using universal hashing
        # h_i(x) = (a_i * x + b_i) mod p mod MAX_HASH
        # where p is a large prime and MAX_HASH is the hash space
        self._MAX_HASH = (1 << 32) - 1
        self._prime = self._largest_prime_32()

        # Generate coefficients for each hash function
        self._coefficients = self._generate_coefficients(num_hashes)

        # Signature: minimum hash value for each hash function
        self._signature = [self._MAX_HASH] * num_hashes

    @staticmethod
    def _largest_prime_32() -> int:
        """Find the largest prime less than 2^32."""
        # Known large 32-bit primes
        return 4294967291  # 2^32 - 5

    def _generate_coefficients(self, count: int) -> list[tuple[int, int]]:
        """Generate hash function coefficients using deterministic seeding."""
        coefficients = []
        for i in range(count):
            # Use a deterministic seed based on index
            seed_bytes = struct.pack("<I", i)
            seed_hash = hashlib.sha256(seed_bytes).digest()
            a = int.from_bytes(seed_hash[:4], "little") % self._prime
            b = int.from_bytes(seed_hash[4:8], "little") % self._prime
            # Ensure a != 0
            if a == 0:
                a = 1
            coefficients.append((a, b))
        return coefficients

    def _to_bytes(self, item: Any) -> bytes:
        if isinstance(item, bytes):
            return item
        elif isinstance(item, str):
            return item.encode("utf-8")
        elif isinstance(item, int):
            return struct.pack("<q", item)
        else:
            return str(item).encode("utf-8")

    def _hash_item(self, data: bytes) -> int:
        """Get a base hash value for the item."""
        return self._hash_func(data) & self._MAX_HASH

    def add(self, item: Any) -> None:
        """Add an element to the set.

        Args:
            item: Element to add. Converted to bytes.
        """
        data = self._to_bytes(item)
        h = self._hash_item(data)

        # Apply each hash function and update signature
        for i, (a, b) in enumerate(self._coefficients):
            # Universal hash: (a * h + b) mod p
            hash_val = (a * h + b) % self._prime
            if hash_val < self._signature[i]:
                self._signature[i] = hash_val

    def add_many(self, items: list[Any]) -> None:
        """Add multiple elements at once."""
        for item in items:
            self.add(item)

    def jaccard_similarity(self, other: "MinHash") -> float:
        """Estimate the Jaccard similarity with another MinHash signature.

        Args:
            other: Another MinHash object.

        Returns:
            Estimated Jaccard similarity (0.0 to 1.0).
        """
        if self._num_hashes != other._num_hashes:
            raise ValueError("MinHash objects must have the same number of hashes")

        matches = sum(
            1
            for s, o in zip(self._signature, other._signature, strict=True)
            if s == o
        )
        return matches / self._num_hashes

    def distance(self, other: "MinHash") -> float:
        """Estimate the Jaccard distance (1 - similarity)."""
        return 1.0 - self.jaccard_similarity(other)

    @property
    def signature(self) -> list[int]:
        """The MinHash signature."""
        return list(self._signature)

    @property
    def num_hashes(self) -> int:
        """Number of hash functions."""
        return self._num_hashes

    def merge(self, other: "MinHash") -> None:
        """Merge another MinHash into this one (element-wise minimum)."""
        if self._num_hashes != other._num_hashes:
            raise ValueError("Cannot merge MinHash with different number of hashes")

        for i in range(self._num_hashes):
            if other._signature[i] < self._signature[i]:
                self._signature[i] = other._signature[i]

    def clear(self) -> None:
        """Reset the signature."""
        self._signature = [self._MAX_HASH] * self._num_hashes

    def to_bytes(self) -> bytes:
        """Serialize the MinHash to bytes."""
        header = struct.pack("<I", self._num_hashes)
        sig = struct.pack(f"<{self._num_hashes}I", *self._signature)
        return header + sig

    @classmethod
    def from_bytes(cls, data: bytes) -> "MinHash":
        """Deserialize a MinHash from bytes."""
        num_hashes = struct.unpack("<I", data[:4])[0]
        mh = cls(num_hashes=num_hashes)
        mh._signature = list(struct.unpack(f"<{num_hashes}I", data[4 : 4 + num_hashes * 4]))
        return mh

    def __repr__(self) -> str:
        return f"MinHash(num_hashes={self._num_hashes})"


class WeightedMinHash:
    """Weighted MinHash for estimating weighted Jaccard similarity.

    Useful for comparing sets where elements have associated weights
    (e.g., TF-IDF vectors, bag-of-words with term frequencies).

    Based on the paper by Li and König (2011).

    Args:
        num_hashes: Number of hash functions.
        seed: Random seed.
    """

    def __init__(self, num_hashes: int = 128, seed: int = 0):
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")

        self._num_hashes = num_hashes
        self._seed = seed
        self._MAX_HASH = (1 << 64) - 1

        # Generate parameters for weighted MinHash
        self._params = self._generate_params(num_hashes)

        # Signature
        self._signature = [(self._MAX_HASH, self._MAX_HASH)] * num_hashes

    def _generate_params(self, count: int) -> list[tuple[int, int, int]]:
        """Generate hash function parameters."""
        params = []
        for i in range(count):
            seed_bytes = struct.pack("<I", self._seed + i)
            hash_bytes = hashlib.sha256(seed_bytes).digest()
            a = int.from_bytes(hash_bytes[:8], "little") % self._MAX_HASH
            b = int.from_bytes(hash_bytes[8:16], "little") % self._MAX_HASH
            c = int.from_bytes(hash_bytes[16:24], "little") % self._MAX_HASH
            params.append((a, b, c))
        return params

    def add(self, item: Any, weight: float = 1.0) -> None:
        """Add an element with a weight.

        Args:
            item: Element to add.
            weight: Weight of the element (default 1.0).
        """
        if weight <= 0:
            return

        data = self._to_bytes(item)
        h = murmur3_32(data)

        for i, (a, b, c) in enumerate(self._params):
            # Weighted MinHash hash
            hash_val = (a * h + b) & self._MAX_HASH
            # The key insight: use weight in the hash
            weighted_key = (hash_val, hash_val ^ c)
            if weight > 0:
                # For positive weights, keep minimum
                if self._signature[i] == (self._MAX_HASH, self._MAX_HASH):
                    self._signature[i] = weighted_key
                else:
                    self._signature[i] = min(self._signature[i], weighted_key)

    def _to_bytes(self, item: Any) -> bytes:
        if isinstance(item, bytes):
            return item
        elif isinstance(item, str):
            return item.encode("utf-8")
        elif isinstance(item, int):
            return struct.pack("<q", item)
        else:
            return str(item).encode("utf-8")

    def jaccard_similarity(self, other: "WeightedMinHash") -> float:
        """Estimate weighted Jaccard similarity."""
        matches = sum(
            1 for s, o in zip(self._signature, other._signature, strict=True) if s == o
        )
        return matches / self._num_hashes

    @property
    def signature(self) -> list[tuple[int, int]]:
        """The weighted MinHash signature."""
        return list(self._signature)

    def __repr__(self) -> str:
        return f"WeightedMinHash(num_hashes={self._num_hashes})"
