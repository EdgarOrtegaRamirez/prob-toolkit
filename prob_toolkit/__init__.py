"""ProbToolkit: Modern Probabilistic Data Structures Library.

A comprehensive collection of probabilistic data structures for efficient
membership testing, frequency estimation, cardinality estimation, and
similarity detection.
"""

__version__ = "0.1.0"

from prob_toolkit.bloom import BloomFilter, CountingBloomFilter, ScalableBloomFilter
from prob_toolkit.count_min_sketch import CountMinSketch
from prob_toolkit.cuckoo import CuckooFilter
from prob_toolkit.hashing import HashFunction, fnv1a_32, murmur3_32, xxhash_64
from prob_toolkit.hyperloglog import HyperLogLog
from prob_toolkit.minhash import MinHash
from prob_toolkit.quotient import QuotientFilter

__all__ = [
    "BloomFilter",
    "CountingBloomFilter",
    "ScalableBloomFilter",
    "CuckooFilter",
    "HyperLogLog",
    "CountMinSketch",
    "MinHash",
    "QuotientFilter",
    "HashFunction",
    "murmur3_32",
    "fnv1a_32",
    "xxhash_64",
]
