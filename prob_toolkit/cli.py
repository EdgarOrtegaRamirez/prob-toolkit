#!/usr/bin/env python3
"""ProbToolkit CLI: Command-line interface for probabilistic data structures.

Usage:
    python -m prob_toolkit <command> [options]

Commands:
    bloom       Bloom filter operations
    cuckoo      Cuckoo filter operations
    hll         HyperLogLog cardinality estimation
    cms         Count-Min Sketch frequency estimation
    minhash     MinHash similarity estimation
    benchmark   Run performance benchmarks
"""

import argparse
import sys
import time


def cmd_bloom(args: argparse.Namespace) -> None:
    """Bloom filter operations."""
    from prob_toolkit.bloom import BloomFilter, CountingBloomFilter, ScalableBloomFilter

    bf_type = args.type or "standard"
    capacity = args.capacity or 10000
    error_rate = args.error_rate or 0.001

    if bf_type == "standard":
        bf = BloomFilter(capacity=capacity, error_rate=error_rate)
    elif bf_type == "counting":
        bf = CountingBloomFilter(capacity=capacity, error_rate=error_rate)
    elif bf_type == "scalable":
        bf = ScalableBloomFilter(initial_capacity=capacity, error_rate=error_rate)
    else:
        print(f"Unknown Bloom filter type: {bf_type}", file=sys.stderr)
        sys.exit(1)

    if args.action == "add":
        for item in args.items:
            bf.add(item)
        print(f"Added {len(args.items)} items")
        print(f"  Fill ratio: {bf.fill_ratio:.2%}")
        if hasattr(bf, "current_error_rate"):
            print(f"  Est. error rate: {bf.current_error_rate:.4%}")

    elif args.action == "check":
        for item in args.items:
            result = "PROBABLY YES" if item in bf else "DEFINITELY NO"
            print(f"  {item}: {result}")

    elif args.action == "stats":
        print(f"Bloom Filter ({bf_type})")
        print(f"  Capacity: {capacity}")
        print(f"  Error rate: {error_rate}")
        print(f"  Items added: {len(bf)}")
        print(f"  Fill ratio: {bf.fill_ratio:.2%}")
        if hasattr(bf, "num_hashes"):
            print(f"  Hash functions: {bf.num_hashes}")
        if hasattr(bf, "bit_size"):
            print(f"  Bit array size: {bf.bit_size} bits ({bf.bit_size // 8} bytes)")
        if hasattr(bf, "current_error_rate"):
            print(f"  Est. error rate: {bf.current_error_rate:.4%}")
        if hasattr(bf, "num_filters"):
            print(f"  Internal filters: {bf.num_filters}")

    elif args.action == "test":
        # Run a quick test with random data
        import random
        import string

        n = args.count or 10000
        test_size = n // 10

        print(f"Testing {bf_type} Bloom filter with {n} items...")
        start = time.perf_counter()

        # Add items
        items = ["".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(n)]
        for item in items:
            bf.add(item)

        add_time = time.perf_counter() - start
        print(f"  Add time: {add_time:.3f}s ({n / add_time:.0f} items/sec)")

        # Check existing items
        start = time.perf_counter()
        true_positives = sum(1 for item in items[:test_size] if item in bf)
        tp_time = time.perf_counter() - start

        # Check non-existing items
        non_items = ["".join(random.choices(string.ascii_uppercase, k=8)) for _ in range(test_size)]
        start = time.perf_counter()
        false_positives = sum(1 for item in non_items if item in bf)
        fp_time = time.perf_counter() - start

        print(f"  Check time: {tp_time + fp_time:.3f}s")
        print(f"  True positives: {true_positives}/{test_size} ({true_positives / test_size:.2%})")
        print(f"  False positives: {false_positives}/{test_size} ({false_positives / test_size:.4%})")

    else:
        print(f"Unknown action: {args.action}", file=sys.stderr)
        sys.exit(1)


def cmd_cuckoo(args: argparse.Namespace) -> None:
    """Cuckoo filter operations."""
    from prob_toolkit.cuckoo import CuckooFilter

    capacity = args.capacity or 10000
    error_rate = args.error_rate or 0.01
    cf = CuckooFilter(capacity=capacity, error_rate=error_rate)

    if args.action == "add":
        for item in args.items:
            success = cf.add(item)
            if not success:
                print(f"  Warning: Failed to add {item} (filter too full)")
        print("Added items")
        print(f"  Total items: {len(cf)}")
        print(f"  Fill ratio: {cf.fill_ratio:.2%}")
        print(f"  Bits/element: {cf.bits_per_element:.1f}")

    elif args.action == "check":
        for item in args.items:
            result = "PROBABLY YES" if item in cf else "DEFINITELY NO"
            print(f"  {item}: {result}")

    elif args.action == "remove":
        for item in args.items:
            success = cf.remove(item)
            print(f"  {item}: {'Removed' if success else 'Not found'}")
        print(f"  Total items: {len(cf)}")

    elif args.action == "stats":
        print("Cuckoo Filter")
        print(f"  Capacity: {capacity}")
        print(f"  Error rate: {error_rate}")
        print(f"  Items: {len(cf)}")
        print(f"  Fill ratio: {cf.fill_ratio:.2%}")
        print(f"  Bits/element: {cf.bits_per_element:.1f}")

    elif args.action == "test":
        import random
        import string

        n = args.count or 10000
        test_size = n // 10

        print(f"Testing Cuckoo filter with {n} items...")
        start = time.perf_counter()

        items = ["".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(n)]
        added = sum(1 for item in items if cf.add(item))
        add_time = time.perf_counter() - start
        print(f"  Add time: {add_time:.3f}s ({added / add_time:.0f} items/sec)")
        print(f"  Successfully added: {added}/{n}")

        start = time.perf_counter()
        true_positives = sum(1 for item in items[:test_size] if item in cf)
        tp_time = time.perf_counter() - start

        non_items = ["".join(random.choices(string.ascii_uppercase, k=8)) for _ in range(test_size)]
        start = time.perf_counter()
        false_positives = sum(1 for item in non_items if item in cf)
        fp_time = time.perf_counter() - start

        print(f"  Check time: {tp_time + fp_time:.3f}s")
        print(f"  True positives: {true_positives}/{test_size} ({true_positives / test_size:.2%})")
        print(f"  False positives: {false_positives}/{test_size} ({false_positives / test_size:.4%})")


def cmd_hll(args: argparse.Namespace) -> None:
    """HyperLogLog operations."""
    from prob_toolkit.hyperloglog import HyperLogLog

    precision = args.precision or 14
    hll = HyperLogLog(precision=precision)

    if args.action == "add":
        for item in args.items:
            hll.add(item)
        print(f"Added {len(args.items)} items")
        print(f"  Estimated cardinality: {hll.estimate():.0f}")

    elif args.action == "stats":
        print("HyperLogLog")
        print(f"  Precision: {precision}")
        print(f"  Registers: {hll.num_buckets}")
        print(f"  Est. cardinality: {hll.estimate():.0f}")
        print(f"  Est. error: {1.04 / (hll.num_buckets**0.5):.2%}")

    elif args.action == "test":
        import random
        import string

        n = args.count or 100000

        print(f"Testing HyperLogLog with {n} unique items...")
        start = time.perf_counter()

        items = ["".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(n)]
        for item in items:
            hll.add(item)

        add_time = time.perf_counter() - start
        estimate = hll.estimate()
        error = abs(estimate - n) / n

        print(f"  Add time: {add_time:.3f}s ({n / add_time:.0f} items/sec)")
        print(f"  True cardinality: {n}")
        print(f"  Estimated cardinality: {estimate:.0f}")
        print(f"  Error: {error:.2%}")

        # Test with duplicates
        print(f"\nTesting with {n} items (50% duplicates)...")
        hll2 = HyperLogLog(precision=precision)
        for _ in range(n):
            if random.random() < 0.5:
                hll2.add(random.choice(items))
            else:
                hll2.add("".join(random.choices(string.ascii_lowercase, k=8)))

        estimate2 = hll2.estimate()
        print(f"  Estimated cardinality: {estimate2:.0f}")

    else:
        print(f"Unknown action: {args.action}", file=sys.stderr)
        sys.exit(1)


def cmd_cms(args: argparse.Namespace) -> None:
    """Count-Min Sketch operations."""
    from prob_toolkit.count_min_sketch import CountMinSketch

    epsilon = args.epsilon or 0.001
    delta = args.delta or 0.01
    cms = CountMinSketch(epsilon=epsilon, delta=delta)

    if args.action == "add":
        for item in args.items:
            cms.add(item)
        print(f"Added {len(args.items)} items")
        print(f"  Total count: {cms.total_count}")

    elif args.action == "estimate":
        for item in args.items:
            est = cms.estimate(item)
            print(f"  {item}: {est}")

    elif args.action == "stats":
        print("Count-Min Sketch")
        print(f"  Epsilon: {epsilon}")
        print(f"  Delta: {delta}")
        print(f"  Width: {cms.width}")
        print(f"  Depth: {cms.depth}")
        print(f"  Total count: {cms.total_count}")
        print(f"  Error bound: {cms.error_bound:.0f}")

    elif args.action == "test":
        import random
        import string

        n = 10000
        num_unique = 100

        print(f"Testing Count-Min Sketch with {n} items ({num_unique} unique)...")
        start = time.perf_counter()

        items = ["".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(num_unique)]
        # Create a stream with Zipf-like distribution
        stream = [random.choice(items) for _ in range(n)]
        for item in stream:
            cms.add(item)

        add_time = time.perf_counter() - start
        print(f"  Add time: {add_time:.3f}s ({n / add_time:.0f} items/sec)")

        # Check estimates
        print("  True frequencies vs estimates:")
        true_counts = {}
        for item in stream:
            true_counts[item] = true_counts.get(item, 0) + 1

        sorted_items = sorted(true_counts.items(), key=lambda x: -x[1])[:5]
        for item, true_count in sorted_items:
            est = cms.estimate(item)
            print(f"    {item}: true={true_count}, est={est}")


def cmd_minhash(args: argparse.Namespace) -> None:
    """MinHash operations."""
    from prob_toolkit.minhash import MinHash

    num_hashes = args.num_hashes or 128

    if args.action == "similarity":
        if len(args.items) < 2:
            print("Need at least 2 sets for similarity", file=sys.stderr)
            sys.exit(1)

        # Parse items as "set_a:item1,item2,...|set_b:item3,item4,..."
        sets = []
        for set_spec in args.items:
            items = set_spec.split(",")
            mh = MinHash(num_hashes=num_hashes)
            for item in items:
                mh.add(item)
            sets.append(mh)

        # Compute pairwise similarities
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                sim = sets[i].jaccard_similarity(sets[j])
                print(f"  Set {i + 1} vs Set {j + 1}: Jaccard similarity = {sim:.3f}")

    elif args.action == "stats":
        print("MinHash")
        print(f"  Num hashes: {num_hashes}")
        print(f"  Est. error: {1.0 / (num_hashes**0.5):.3f}")

    elif args.action == "test":
        import random
        import string

        set_size = 1000
        overlap = 500  # 50% overlap

        print(f"Testing MinHash with {set_size} items per set ({overlap} shared)...")
        start = time.perf_counter()

        # Generate two sets with known overlap
        all_items = ["".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(set_size * 2 - overlap)]
        set_a_items = all_items[:set_size]
        set_b_items = all_items[set_size - overlap :]

        mh_a = MinHash(num_hashes=num_hashes)
        mh_b = MinHash(num_hashes=num_hashes)
        for item in set_a_items:
            mh_a.add(item)
        for item in set_b_items:
            mh_b.add(item)

        add_time = time.perf_counter() - start
        estimated_sim = mh_a.jaccard_similarity(mh_b)
        true_sim = overlap / (2 * set_size - overlap)

        print(f"  Build time: {add_time:.3f}s")
        print(f"  True Jaccard similarity: {true_sim:.3f}")
        print(f"  Estimated Jaccard similarity: {estimated_sim:.3f}")
        print(f"  Error: {abs(estimated_sim - true_sim):.3f}")

    else:
        print(f"Unknown action: {args.action}", file=sys.stderr)
        sys.exit(1)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run performance benchmarks."""
    import random
    import string

    n = args.count or 100000
    items = ["".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(n)]

    print(f"Benchmarking with {n:,} items...")
    print("=" * 60)

    results = {}

    # Bloom Filter
    from prob_toolkit.bloom import BloomFilter

    bf = BloomFilter(capacity=n, error_rate=0.01)
    start = time.perf_counter()
    for item in items:
        bf.add(item)
    bf_add_time = time.perf_counter() - start

    start = time.perf_counter()
    for item in items[:1000]:
        _ = item in bf
    bf_check_time = time.perf_counter() - start

    results["BloomFilter"] = {
        "add_time": bf_add_time,
        "check_time": bf_check_time,
        "add_rate": n / bf_add_time,
        "check_rate": 1000 / bf_check_time,
        "memory_kb": len(bf._bit_array) / 1024,
    }

    # Cuckoo Filter
    from prob_toolkit.cuckoo import CuckooFilter

    cf = CuckooFilter(capacity=n, error_rate=0.01)
    start = time.perf_counter()
    added = 0
    for item in items:
        if cf.add(item):
            added += 1
    cf_add_time = time.perf_counter() - start

    start = time.perf_counter()
    for item in items[:1000]:
        _ = item in cf
    cf_check_time = time.perf_counter() - start

    results["CuckooFilter"] = {
        "add_time": cf_add_time,
        "check_time": cf_check_time,
        "add_rate": added / cf_add_time,
        "check_rate": 1000 / cf_check_time,
        "memory_kb": len(cf._buckets) * 4 / 1024,
    }

    # HyperLogLog
    from prob_toolkit.hyperloglog import HyperLogLog

    hll = HyperLogLog(precision=14)
    start = time.perf_counter()
    for item in items:
        hll.add(item)
    hll_add_time = time.perf_counter() - start

    results["HyperLogLog"] = {
        "add_time": hll_add_time,
        "add_rate": n / hll_add_time,
        "memory_kb": 2**14 / 1024,
    }

    # Count-Min Sketch
    from prob_toolkit.count_min_sketch import CountMinSketch

    cms = CountMinSketch(epsilon=0.001, delta=0.01)
    start = time.perf_counter()
    for item in items:
        cms.add(item)
    cms_add_time = time.perf_counter() - start

    start = time.perf_counter()
    for item in items[:1000]:
        cms.estimate(item)
    cms_check_time = time.perf_counter() - start

    results["CountMinSketch"] = {
        "add_time": cms_add_time,
        "check_time": cms_check_time,
        "add_rate": n / cms_add_time,
        "check_rate": 1000 / cms_check_time,
        "memory_kb": cms.width * cms.depth * 8 / 1024,
    }

    # MinHash
    from prob_toolkit.minhash import MinHash

    mh = MinHash(num_hashes=128)
    start = time.perf_counter()
    for item in items:
        mh.add(item)
    mh_add_time = time.perf_counter() - start

    results["MinHash"] = {
        "add_time": mh_add_time,
        "add_rate": n / mh_add_time,
        "memory_kb": 128 * 4 / 1024,
    }

    # Print results
    print(f"{'Data Structure':<20} {'Add (ms)':<12} {'Add Rate':<15} {'Query (ms)':<12} {'Memory (KB)':<12}")
    print("-" * 60)
    for name, r in results.items():
        add_ms = r["add_time"] * 1000
        query_ms = r.get("check_time", 0) * 1000
        print(f"{name:<20} {add_ms:<12.1f} {r['add_rate']:>10,.0f}/s  {query_ms:<12.1f} {r['memory_kb']:<12.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prob-toolkit",
        description="Probabilistic Data Structures Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Bloom filter
    bloom_parser = subparsers.add_parser("bloom", help="Bloom filter operations")
    bloom_parser.add_argument("action", choices=["add", "check", "stats", "test"])
    bloom_parser.add_argument("--type", choices=["standard", "counting", "scalable"], default="standard")
    bloom_parser.add_argument("--capacity", type=int, default=10000)
    bloom_parser.add_argument("--error-rate", type=float, default=0.001)
    bloom_parser.add_argument("--count", type=int, default=10000)
    bloom_parser.add_argument("items", nargs="*", help="Items to add/check")

    # Cuckoo filter
    cuckoo_parser = subparsers.add_parser("cuckoo", help="Cuckoo filter operations")
    cuckoo_parser.add_argument("action", choices=["add", "check", "remove", "stats", "test"])
    cuckoo_parser.add_argument("--capacity", type=int, default=10000)
    cuckoo_parser.add_argument("--error-rate", type=float, default=0.01)
    cuckoo_parser.add_argument("--count", type=int, default=10000)
    cuckoo_parser.add_argument("items", nargs="*", help="Items to add/check/remove")

    # HyperLogLog
    hll_parser = subparsers.add_parser("hll", help="HyperLogLog cardinality estimation")
    hll_parser.add_argument("action", choices=["add", "stats", "test"])
    hll_parser.add_argument("--precision", type=int, default=14)
    hll_parser.add_argument("--count", type=int, default=100000)
    hll_parser.add_argument("items", nargs="*", help="Items to add")

    # Count-Min Sketch
    cms_parser = subparsers.add_parser("cms", help="Count-Min Sketch frequency estimation")
    cms_parser.add_argument("action", choices=["add", "estimate", "stats", "test"])
    cms_parser.add_argument("--epsilon", type=float, default=0.001)
    cms_parser.add_argument("--delta", type=float, default=0.01)
    cms_parser.add_argument("items", nargs="*", help="Items to add/estimate")

    # MinHash
    minhash_parser = subparsers.add_parser("minhash", help="MinHash similarity estimation")
    minhash_parser.add_argument("action", choices=["similarity", "stats", "test"])
    minhash_parser.add_argument("--num-hashes", type=int, default=128)
    minhash_parser.add_argument("items", nargs="*", help="Sets (comma-separated items)")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--count", type=int, default=100000)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "bloom": cmd_bloom,
        "cuckoo": cmd_cuckoo,
        "hll": cmd_hll,
        "cms": cmd_cms,
        "minhash": cmd_minhash,
        "benchmark": cmd_benchmark,
    }

    try:
        commands[args.command](args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
