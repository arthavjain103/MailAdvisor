import os

import pytest
import redis

from cache import store_in_cache, get_cached_response, clear_cache


pytestmark = pytest.mark.skipif(
    os.getenv("CACHE_TEST_REDIS") != "1",
    reason="Set CACHE_TEST_REDIS=1 to run Redis integration tests",
)


@pytest.fixture
def redis_client():
    client = redis.Redis(
        host="localhost",
        port=6379,
        db=15,
    )

    client.ping()
    clear_cache(redis_client=client)

    yield client

    clear_cache(redis_client=client)
    client.close()


def test_store_and_retrieve_response(redis_client):
    query = "What is retrieval augmented generation?"
    response = "RAG combines retrieval with generation."
    metadata = {"source": "test.pdf"}

    store_in_cache(
        query=query,
        response=response,
        metadata=metadata,
        redis_client=redis_client,
    )

    result = get_cached_response(
        query=query,
        redis_client=redis_client,
    )

    assert result is not None
    assert result["response"] == response
    assert result["match_type"] == "exact"
    assert result["metadata"] == metadata


def test_cache_miss(redis_client):
    result = get_cached_response(
        query="A completely unrelated test query",
        redis_client=redis_client,
    )

    assert result is None


def test_clear_cache(redis_client):
    query = "Explain semantic caching"
    response = "Semantic caching reuses answers for similar queries."

    store_in_cache(
        query=query,
        response=response,
        redis_client=redis_client,
    )

    assert get_cached_response(query, redis_client) is not None

    clear_cache(redis_client=redis_client)

    assert get_cached_response(query, redis_client) is None