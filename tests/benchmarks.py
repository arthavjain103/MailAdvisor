"""Benchmark Redis semantic cache lookup performance."""

import os
import statistics
import time
import sys 

import numpy as np
import redis
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache import (
    KEY_PREFIX,
    _embed,
    _make_key,
    clear_cache,
    get_cached_response,
    store_in_cache,
)

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 15

CACHE_SIZES = [100, 1000, 5000]
ITERATIONS = 20


def measure_latency(func, iterations=ITERATIONS):
    """Return latency measurements in milliseconds."""
    timings = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        timings.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": statistics.mean(timings),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def populate_cache(client, size, rng):
    """Populate Redis with synthetic embeddings for lookup benchmarks."""
    clear_cache(redis_client=client)

    # Get the real embedding dimension used by the cache.
    query_vector = _embed("benchmark dimension query")
    dimension = query_vector.size

    for i in range(size - 1):
        query = f"synthetic benchmark query {i}"
        key = _make_key(query)

        vector = rng.normal(size=dimension).astype(np.float32)
        norm = np.linalg.norm(vector)

        if norm:
            vector /= norm

        client.hset(key, "query", query)
        client.hset(key, "embedding", vector.tobytes())
        client.hset(key, "response", f"Benchmark response {i}")
        client.hset(key, "metadata", "{}")
        


    # Add one real cache entry for exact-hit measurements.
    store_in_cache(
        query="What is retrieval augmented generation?",
        response="RAG combines retrieval and generation.",
        metadata={"source": "benchmark"},
        redis_client=client,
    )


def main():
    if os.getenv("CACHE_BENCHMARK") != "1":
        print("Set CACHE_BENCHMARK=1 to run the benchmark.")
        return

    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=False,
    )

    try:
        client.ping()
    except redis.RedisError as exc:
        print(f"Redis is unavailable: {exc}")
        return

    rng = np.random.default_rng(42)

    exact_query = "What is retrieval augmented generation?"

    # Ensure the model is loaded before collecting timings.
    _embed("benchmark warmup query")

    print("\nRedis Semantic Cache Benchmark")
    print(f"Redis DB: {REDIS_DB}")
    print(f"Iterations per measurement: {ITERATIONS}")
    print("-" * 75)
    print(
        f"{'Entries':>8} | {'Exact avg':>12} | "
        f"{'Miss avg':>12} | {'Miss p95':>12}"
    )
    print("-" * 75)

    try:
        for size in CACHE_SIZES:
            populate_cache(client, size, rng)

            # Warm up Python/model and Redis connections.
            get_cached_response(exact_query, redis_client=client)

            exact_result = measure_latency(
                lambda: get_cached_response(
                    exact_query,
                    redis_client=client,
                )
            )

            miss_counter = iter(range(ITERATIONS))

            def cache_miss():
                # Unique query every time: avoids accidental exact hits.
                query_id = next(miss_counter)
                result = get_cached_response(
                    f"unrelated benchmark question number {query_id}",
                    redis_client=client,
                )
                if result is not None:
                    raise AssertionError(
                        "Expected a cache miss; check similarity threshold."
                    )

            miss_result = measure_latency(cache_miss)

            print(
                f"{size:>8} | "
                f"{exact_result['avg_ms']:>9.3f} ms | "
                f"{miss_result['avg_ms']:>9.3f} ms | "
                f"{miss_result['p95_ms']:>9.3f} ms"
            )

    finally:
        clear_cache(redis_client=client)
        client.close()

    print("-" * 75)
    print("Benchmark completed.")


if __name__ == "__main__":
    main()