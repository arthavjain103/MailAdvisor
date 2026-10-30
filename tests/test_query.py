
import json
import os
from pathlib import Path

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


@pytest.fixture
def cache_queries():
    file_path = Path(__file__).parent / "data" / "queries.json"

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_domain_queries_load(cache_queries):
    assert len(cache_queries) > 0

    for item in cache_queries:
        assert "query" in item
        assert "category" in item


def test_original_queries_can_be_cached(redis_client, cache_queries):
    for item in cache_queries:
        query = item["query"]
        response = f"Test response for category: {item['category']}"

        store_in_cache(
            query=query,
            response=response,
            metadata={"category": item["category"]},
            redis_client=redis_client,
        )

        result = get_cached_response(
            query=query,
            redis_client=redis_client,
        )

        assert result is not None, f"Cache miss for query: {query}"
        assert result["match_type"] == "exact"
        assert result["response"] == response

def test_exact_cache_hit(redis_client):
    query = "How do I configure SPF records for my domain?"
    response = "Add a valid SPF TXT record to your DNS."

    store_in_cache(
        query=query,
        response=response,
        redis_client=redis_client,
    )

    result = get_cached_response(
        query=query,
        redis_client=redis_client,
    )

    assert result is not None
    assert result["match_type"] == "exact"
    assert result["response"] == response


def test_semantic_cache_hit(redis_client):
    original_query = "Why are my emails going to spam?"
    similar_query = "Why do my emails land in the spam folder?"
    response = "Check your SPF, DKIM, DMARC records and sender reputation."

    store_in_cache(
        query=original_query,
        response=response,
        redis_client=redis_client,
    )

    result = get_cached_response(
        query=similar_query,
        redis_client=redis_client,
    )

    assert result is not None, (
        "Expected a semantic cache hit, but no cached response was found."
    )
    assert result["match_type"] == "semantic"
    assert result["response"] == response


def test_unrelated_query_cache_miss(redis_client):
    store_in_cache(
        query="How do I configure SPF records for my domain?",
        response="Configure the correct SPF TXT record in DNS.",
        redis_client=redis_client,
    )

    result = get_cached_response(
        query="What is the capital of Japan?",
        redis_client=redis_client,
    )

    assert result is None


def test_different_email_topics_do_not_match(redis_client):
    store_in_cache(
        query="How do I configure SPF records for my domain?",
        response="Configure the SPF TXT record in your DNS.",
        redis_client=redis_client,
    )

    result = get_cached_response(
        query="Why is my DKIM signature failing?",
        redis_client=redis_client,
    )

    assert result is None