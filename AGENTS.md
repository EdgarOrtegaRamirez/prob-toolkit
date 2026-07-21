# AGENTS.md — Notes for AI Agents

## Project Overview
ProbToolkit is a Python library for probabilistic data structures. All implementations are pure Python with no external dependencies.

## Key Files
- `prob_toolkit/hashing.py` — Hash functions (MurmurHash3, FNV-1a, xxHash)
- `prob_toolkit/bloom.py` — Bloom, Counting, Scalable Bloom filters
- `prob_toolkit/cuckoo.py` — Cuckoo filter with deletion support
- `prob_toolkit/hyperloglog.py` — HyperLogLog cardinality estimator
- `prob_toolkit/count_min_sketch.py` — Count-Min Sketch frequency estimator
- `prob_toolkit/minhash.py` — MinHash + WeightedMinHash for similarity
- `prob_toolkit/quotient.py` — Quotient filter (cache-friendly)
- `prob_toolkit/cli.py` — Command-line interface

## Running Tests
```bash
cd /root/workspace/prob-toolkit
uv venv && uv pip install --editable ".[dev]"
uv run pytest -v
```

## Linting
```bash
uv run ruff check prob_toolkit/ tests/
```

## Building
The package uses hatchling as the build backend. No compilation needed — pure Python.

## Key Algorithms
- **Bloom Filter**: Optimal parameters computed from capacity and error rate
- **Cuckoo Filter**: XOR-based alternate bucket computation for O(1) insertion
- **HyperLogLog**: Harmonic mean estimator with small/large range corrections
- **Count-Min Sketch**: Conservative counting with width = ceil(e/ε), depth = ceil(ln(1/δ))
- **MinHash**: Universal hashing with LCG coefficients for independent hash functions
- **Quotient Filter**: Clustered storage with metadata bits (occupied, continuation, shifted)
