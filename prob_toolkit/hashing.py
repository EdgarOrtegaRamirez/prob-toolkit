"""Hash function implementations for probabilistic data structures.

Provides fast, well-distributed hash functions suitable for use in
Bloom filters, Cuckoo filters, and other probabilistic structures.
"""

import struct
from collections.abc import Callable
from typing import Protocol


class HashFunction(Protocol):
    """Protocol for hash functions usable in probabilistic data structures."""

    def __call__(self, data: bytes, seed: int = 0) -> int:
        """Hash data bytes and return an unsigned integer."""
        ...


def murmur3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3 32-bit hash function.

    Excellent distribution and speed. Used widely in databases and
    network applications (Redis, Memcached, Cassandra).

    Args:
        data: Bytes to hash.
        seed: Hash seed for independent hash functions.

    Returns:
        32-bit unsigned integer hash.
    """
    length = len(data)
    nblocks = length // 4

    h1 = seed
    c1 = 0xCCCCCCCC
    c2 = 0x1B873593

    # body
    for block_start in range(0, nblocks * 4, 4):
        k1 = struct.unpack("<I", data[block_start : block_start + 4])[0]

        k1 *= c1
        k1 = (k1 << 15) | (k1 >> 17)
        k1 *= c2

        h1 ^= k1
        h1 = (h1 << 13) | (h1 >> 19)
        h1 = h1 * 5 + 0xE6546B64

    # tail
    tail_index = nblocks * 4
    k1 = 0
    tail_size = length & 3

    if tail_size >= 3:
        k1 ^= data[tail_index + 2] << 16
    if tail_size >= 2:
        k1 ^= data[tail_index + 1] << 8
    if tail_size >= 1:
        k1 ^= data[tail_index]
        k1 *= c1
        k1 = (k1 << 15) | (k1 >> 17)
        k1 *= c2
        h1 ^= k1

    # finalization
    h1 ^= length
    h1 ^= h1 >> 16
    h1 *= 0x85EBCA6B
    h1 ^= h1 >> 13
    h1 *= 0xC2B2AE35
    h1 ^= h1 >> 16

    return h1 & 0xFFFFFFFF


def fnv1a_32(data: bytes, seed: int = 0) -> int:
    """FNV-1a 32-bit hash function.

    Simple, fast, with good distribution. Widely used in hash tables.

    Args:
        data: Bytes to hash.
        seed: Hash seed (XOR'd with FNV offset basis).

    Returns:
        32-bit unsigned integer hash.
    """
    h = 0x811C9DC5 ^ seed  # FNV offset basis
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime
    return h


def xxhash_64(data: bytes, seed: int = 0) -> int:
    """xxHash 64-bit hash function (pure Python implementation).

    Very fast with good distribution. Excellent for large data.

    Args:
        data: Bytes to hash.
        seed: Hash seed.

    Returns:
        64-bit unsigned integer hash.
    """
    if len(data) == 0:
        return seed ^ 0x165667B19D3719E9

    PRIME64_1 = 0x9E3779B185EBCA87
    PRIME64_2 = 0xC2B2AE3D27D4EB4F
    PRIME64_3 = 0x165667B19D3719E9
    PRIME64_4 = 0x85EBCA77C2B2AE63
    PRIME64_5 = 0x27D4EB2F165667C5

    h64 = PRIME64_5 + len(data)
    h64 += seed

    # Process 32-byte chunks
    if len(data) >= 32:
        limit = len(data) - 32
        v1 = h64 + PRIME64_1 + PRIME64_2
        v2 = h64 + PRIME64_2
        v3 = h64
        v4 = h64 - PRIME64_1

        idx = 0
        while idx <= limit:
            k1 = struct.unpack("<Q", data[idx : idx + 8])[0]
            k2 = struct.unpack("<Q", data[idx + 8 : idx + 16])[0]
            k3 = struct.unpack("<Q", data[idx + 16 : idx + 24])[0]
            k4 = struct.unpack("<Q", data[idx + 24 : idx + 32])[0]

            v1 = _xxhash_round(v1, k1)
            v2 = _xxhash_round(v2, k2)
            v3 = _xxhash_round(v3, k3)
            v4 = _xxhash_round(v4, k4)

            idx += 32

        h64 = _rotl64(v1, 1) + _rotl64(v2, 7) + _rotl64(v3, 12) + _rotl64(v4, 18)

        # Merge rounds
        h64 ^= _xxhash_merge_round(h64, v1)
        h64 = h64 * PRIME64_3 + PRIME64_4
        h64 ^= _xxhash_merge_round(h64, v2)
        h64 = h64 * PRIME64_3 + PRIME64_4
        h64 ^= _xxhash_merge_round(h64, v3)
        h64 = h64 * PRIME64_3 + PRIME64_4
        h64 ^= _xxhash_merge_round(h64, v4)
        h64 = h64 * PRIME64_3 + PRIME64_4
    else:
        h64 += seed

    # Process remaining bytes
    idx = (len(data) // 32) * 32
    while idx + 8 <= len(data):
        k1 = struct.unpack("<Q", data[idx : idx + 8])[0]
        k1 *= PRIME64_2
        k1 = _rotl64(k1, 31)
        k1 *= PRIME64_1
        h64 ^= k1
        h64 = _rotl64(h64, 27) * PRIME64_1 + PRIME64_4
        idx += 8

    if idx + 4 <= len(data):
        h64 ^= (struct.unpack("<I", data[idx : idx + 4])[0]) * PRIME64_1
        h64 = _rotl64(h64, 23) * PRIME64_2 + PRIME64_3
        idx += 4

    while idx < len(data):
        h64 ^= data[idx] * PRIME64_5
        h64 = _rotl64(h64, 11) * PRIME64_1
        idx += 1

    # Avalanche
    h64 ^= h64 >> 33
    h64 *= PRIME64_2
    h64 ^= h64 >> 29
    h64 *= PRIME64_3
    h64 ^= h64 >> 32

    return h64 & 0xFFFFFFFFFFFFFFFF


def _xxhash_round(acc: int, k: int) -> int:
    _PRIME64_1 = 0x9E3779B185EBCA87
    _PRIME64_2 = 0xC2B2AE3D27D4EB4F
    acc += k * _PRIME64_2
    acc = _rotl64(acc, 31)
    acc *= _PRIME64_1
    return acc & 0xFFFFFFFFFFFFFFFF


def _xxhash_merge_round(acc: int, val: int) -> int:
    _PRIME64_1 = 0x9E3779B185EBCA87
    _PRIME64_2 = 0xC2B2AE3D27D4EB4F
    val *= _PRIME64_1
    val = _rotl64(val, 31)
    val *= _PRIME64_2
    acc ^= val
    return (acc * _PRIME64_1 + _PRIME64_2) & 0xFFFFFFFFFFFFFFFF


def _rotl64(x: int, r: int) -> int:
    """Rotate left a 64-bit integer."""
    return ((x << r) | (x >> (64 - r))) & 0xFFFFFFFFFFFFFFFF


def make_double_hash(h1: HashFunction, h2: HashFunction) -> Callable[[bytes, int], int]:
    """Create a double-hash function for Bloom filter probing.

    Uses the formula: g(x) = h1(x) + i * h2(x)
    This gives independent hash positions with only 2 hash computations.

    Args:
        h1: First hash function.
        h2: Second hash function.

    Returns:
        A function that takes (data, i) and returns the i-th hash position.
    """

    def double_hash(data: bytes, i: int) -> int:
        return h1(data) + i * h2(data)

    return double_hash


# Default hash function
DEFAULT_HASH = murmur3_32
