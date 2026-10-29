"""Semantic cache on Redis: reuse answers for similar queries."""

import hashlib
import json
import time
from functools import lru_cache

import numpy as np
import redis
from ingestion.embedder import get_embedding_model as get_shared_embedding_model

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
SIMILARITY_THRESHOLD = 0.89  # e5 scores run high, so keep this strict
CACHE_TTL_SECONDS = 3600
KEY_PREFIX = "semantic_cache:query:"


_embedding_model = None
_redis_client = None

def _make_key(query):
    normalized = " ".join(query.strip().lower().split())
    return KEY_PREFIX + hashlib.sha256(normalized.encode()).hexdigest()[:16]




def get_embedding_model():
     """Reuse the embedding model shared with semantic search."""
     global _embedding_model

     if _embedding_model is None:
        print("[CACHE] Getting shared embedding model...")
        _embedding_model = get_shared_embedding_model()

     return _embedding_model


@lru_cache(maxsize=128)
def _embed(query):
    """Embed a query (memoized, so lookup + store embed only once)."""
    vec = np.asarray(
        get_embedding_model().embed_query(f"query: {query.strip()}"),  # e5 prefix
        dtype=np.float32,
    )
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def get_redis_connection():
    """Return a shared Redis client."""
    global _redis_client
    if _redis_client is None:
        pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        _redis_client = redis.Redis(connection_pool=pool)
    return _redis_client


def _json_default(obj):
    """Make numpy types JSON serializable."""
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def store_in_cache(query, response, metadata=None, redis_client=None):
    """Save query, embedding, response and metadata. Returns the key or None."""
    client = redis_client if redis_client is not None else get_redis_connection()
    key = _make_key(query)

    try:
        pipe = client.pipeline()
        # Use individual hset calls instead of mapping (compatible with all redis-py versions)
        pipe.hset(key, "query", query)
        pipe.hset(key, "embedding", _embed(query).tobytes())  # raw float32 bytes
        pipe.hset(key, "response", response)
        pipe.hset(key, "metadata", json.dumps(metadata or {}, default=_json_default))
        pipe.expire(key, CACHE_TTL_SECONDS)
        pipe.execute()
        print(f"[CACHE] Stored -> {key} (TTL {CACHE_TTL_SECONDS}s)")
        return key
    except redis.RedisError as e:
        print(f"[CACHE] Store failed: {e}")
        return None
    
def _build_result(client, key, similarity):
    data = client.hgetall(key)
    if not data:  # expire ho gaya
        return None
    try:
        metadata = json.loads(data.get(b"metadata", b"{}").decode("utf-8"))
    except json.JSONDecodeError:
        metadata = {}
    return {
        "response": data[b"response"].decode("utf-8"),
        "similarity": similarity,
        "stored_query": data[b"query"].decode("utf-8"),
        "metadata": metadata,
        "cache_key": key.decode("utf-8") if isinstance(key, bytes) else key,
        "match_type": "exact" if similarity == 1.0 else "semantic",
    }

def lookup_exact(redis_client, query):
    key = _make_key(query)
    if not redis_client.exists(key):
        return None
    return _build_result(redis_client, key, 1.0)


def lookup_in_cache(redis_client, query):
    """Return the best cached match above the threshold, else None."""
    q_vec = _embed(query)

    keys = list(redis_client.scan_iter(match=KEY_PREFIX + "*", count=200))
    print(f"[CACHE] Entries in cache: {len(keys)}")
    if not keys:
        return None

    # Fetch all embeddings in one round trip
    pipe = redis_client.pipeline()
    for k in keys:
        pipe.hget(k, "embedding")
    raw = pipe.execute()

    valid_keys, vectors = [], []
    for k, blob in zip(keys, raw):
        if blob is None:  # expired between scan and fetch
            continue
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.size == q_vec.size:  # skip entries from a different model
            valid_keys.append(k)
            vectors.append(vec)

    if not vectors:
        return None

    # Vectors are normalized, so dot product == cosine similarity
    sims = np.vstack(vectors) @ q_vec
    best = int(np.argmax(sims))
    best_sim = float(sims[best])
    print(f"[CACHE] Best similarity: {best_sim:.4f} (threshold {SIMILARITY_THRESHOLD})")

    if best_sim < SIMILARITY_THRESHOLD:
        return None

    key = valid_keys[best]
    return _build_result(redis_client, key, best_sim)


def get_cached_response(query, redis_client=None):
    """Return cached result dict or None. Never raises on Redis errors."""
    client = redis_client if redis_client is not None else get_redis_connection()
    start = time.perf_counter()
    try:
         result = lookup_exact(client, query)          # 1. exact
         if result:
            print(f"[CACHE] EXACT HIT in {time.perf_counter() - start:.4f}s")
            return result

         result = lookup_in_cache(client, query)       # 2. semantic
    except redis.RedisError as e:
        print(f"[CACHE] Redis error, skipping cache: {e}")
        return None

    status = "HIT" if result else "MISS"
    print(f"[CACHE] {status} in {time.perf_counter() - start:.3f}s")
    return result


def clear_cache(redis_client=None):
    """Delete all semantic cache entries."""
    client = redis_client if redis_client is not None else get_redis_connection()
    keys = list(client.scan_iter(match=KEY_PREFIX + "*", count=200))
    if keys:
        client.delete(*keys)
    print(f"[CACHE] Cleared {len(keys)} entries")