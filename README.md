# ProbToolkit

A modern Python library for probabilistic data structures.

Probabilistic data structures trade perfect accuracy for massive space and time efficiency. They're used in databases, networking, big data systems, and anywhere you need fast approximate answers with guaranteed error bounds.

## Features

### Data Structures

| Structure | Use Case | Space | Accuracy |
|-----------|----------|-------|----------|
| **Bloom Filter** | Membership testing | O(n) bits | Configurable FP rate |
| **Counting Bloom Filter** | Membership + deletion | O(n × counter_bits) bits | Configurable FP rate |
| **Scalable Bloom Filter** | Growing sets | Auto-scales | Tightening FP rate |
| **Cuckoo Filter** | Membership + deletion | O(n × fp_bits) bits | Configurable FP rate |
| **HyperLogLog** | Cardinality estimation | O(2^p) bytes | ~1.04/√m standard error |
| **Count-Min Sketch** | Frequency estimation | O(w × d) counters | ≤ εN error with prob ≥ 1-δ |
| **MinHash** | Set similarity | O(k) values | 1/√k standard error |
| **Quotient Filter** | Cache-friendly membership | O(n × fp_bits) bits | Configurable FP rate |

### Hash Functions

- **MurmurHash3** — Fast, excellent distribution (Redis, Cassandra)
- **FNV-1a** — Simple, widely used (hash tables)
- **xxHash** — Very fast, 64-bit (data processing)

## Quick Start

### Installation

```bash
pip install prob-toolkit
```

### As a Library

```python
from prob_toolkit import BloomFilter, HyperLogLog, MinHash

# Bloom Filter — membership testing with O(1) lookups
bf = BloomFilter(capacity=1_000_000, error_rate=0.01)
bf.add("user_12345")
bf.add("user_67890")
print("user_12345" in bf)  # True
print("user_99999" in bf)  # False (usually)
print(f"False positive rate: {bf.current_error_rate:.4%}")

# HyperLogLog — cardinality estimation with ~0.8% error
hll = HyperLogLog(precision=14)
for user_id in millions_of_users:
    hll.add(user_id)
print(f"Unique users: {hll.estimate():,.0f}")  # ~1,000,000 with 0.8% error

# MinHash — Jaccard similarity between sets
set_a = MinHash(num_hashes=128)
set_b = MinHash(num_hashes=128)
for item in dataset_a:
    set_a.add(item)
for item in dataset_b:
    set_b.add(item)
similarity = set_a.jaccard_similarity(set_b)
print(f"Dataset similarity: {similarity:.3f}")

# Count-Min Sketch — frequency estimation in streams
from prob_toolkit import CountMinSketch
cms = CountMinSketch(epsilon=0.001, delta=0.01)
for word in text_stream:
    cms.add(word)
print(f"'the' appears ~{cms.estimate('the'):,.0f} times")

# Cuckoo Filter — membership testing with deletion support
from prob_toolkit import CuckooFilter
cf = CuckooFilter(capacity=10_000, error_rate=0.01)
cf.add("session_abc123")
cf.add("session_def456")
cf.remove("session_abc123")  # Supported!
print("session_abc123" in cf)  # False
```

### As a CLI Tool

```bash
# Bloom filter test
prob-toolkit bloom test --count 100000

# Cuckoo filter test
prob-toolkit cuckoo test --count 100000

# HyperLogLog test
prob-toolkit hll test --count 1000000

# Count-Min Sketch test
prob-toolkit cms test

# MinHash test
prob-toolkit minhash test

# Run all benchmarks
prob-toolkit benchmark --count 1000000
```

## Data Structure Details

### Bloom Filter

A Bloom filter uses a bit array and k hash functions to test membership. False positives are possible, but false negatives are not.

**When to use:**
- Database query optimization (skip keys not in disk)
- Network routers (packet routing)
- Malware detection (known bad URLs)
- Cache deduplication

**Space formula:** `m = -n × ln(p) / (ln(2))²` bits for n elements, false positive rate p.

### Cuckoo Filter

A Cuckoo filter stores fingerprints in a hash table with cuckoo hashing. Supports deletion and has better space efficiency than Bloom filters at low-to-moderate FP rates.

**When to use:**
- Same as Bloom filter, but when you need deletion
- Network packet deduplication
- Set membership with updates

**Advantage over Bloom:** Supports deletion, better cache locality, smaller at moderate FP rates.

### HyperLogLog

HyperLogLog estimates cardinality by counting leading zeros in hash values. With precision=14, it uses only 16KB and achieves ~0.8% standard error.

**When to use:**
- Counting unique visitors
- Database query optimization (estimate result size)
- Network flow monitoring
- Set cardinality without storing elements

### Count-Min Sketch

A 2D array of counters for frequency estimation. Items hash to positions in each row; the minimum across rows is the estimate. Never underestimates.

**When to use:**
- Stream processing (word frequency)
- Network monitoring (flow size estimation)
- Database query optimization (selectivity estimation)

**Guarantee:** Error ≤ ε×N with probability ≥ 1-δ.

### MinHash

Estimates Jaccard similarity between sets by computing minimum hash values under multiple hash functions.

**When to use:**
- Near-duplicate detection
- Document similarity
- Recommendation systems
- Data deduplication

**Accuracy:** Standard error = 1/√k where k is the number of hash functions.

### Quotient Filter

A cache-friendly alternative to Bloom filters that stores fingerprints using quotienting. Better cache performance than standard hash tables.

**When to use:**
- Same as Bloom filter, but when cache performance matters
- Embedded systems with limited memory

## Architecture

```
prob_toolkit/
├── __init__.py          # Public API
├── bloom.py             # Bloom, Counting, Scalable Bloom filters
├── cuckoo.py            # Cuckoo filter
├── hyperloglog.py       # HyperLogLog cardinality estimator
├── count_min_sketch.py  # Count-Min Sketch frequency estimator
├── minhash.py           # MinHash similarity estimator
├── quotient.py          # Quotient filter
├── hashing.py           # Hash function implementations
└── cli.py               # Command-line interface
```

## Performance

Typical performance on modern hardware (Apple M1 / Intel i7):

| Operation | Bloom | Cuckoo | HLL | CMS | MinHash |
|-----------|-------|--------|-----|-----|---------|
| Add (ops/sec) | 2-5M | 1-3M | 3-6M | 3-6M | 2-4M |
| Query (ops/sec) | 3-6M | 2-4M | — | 3-6M | — |
| Memory/1M elements | ~1.2MB | ~1MB | 16KB | ~1.2MB | 0.5KB |

## Hash Functions

All hash functions are pure Python with no external dependencies:

- **MurmurHash3** — Used as the default hash function. Excellent distribution and speed.
- **FNV-1a** — Used as the secondary hash for double hashing in Bloom filters.
- **xxHash** — 64-bit variant for high-throughput applications.

## Contributing

Contributions welcome! Please ensure:
1. All tests pass (`pytest`)
2. Type hints are maintained
3. New data structures include benchmarks
4. Documentation is updated

## License

MIT
