from unittest.mock import MagicMock

from cache import _make_key, lookup_exact


def test_make_key_normalizes_query():
    key1 = _make_key("  What is RAG?  ")
    key2 = _make_key("what is rag?")

    assert key1 == key2


def test_exact_cache_hit():
    client = MagicMock()

    client.exists.return_value = True
    client.hgetall.return_value = {
        b"response": b"RAG combines retrieval and generation.",
        b"query": b"What is RAG?",
        b"metadata": b'{"source": "test.pdf"}',
    }

    result = lookup_exact(client, "What is RAG?")

    assert result is not None
    assert result["response"] == "RAG combines retrieval and generation."
    assert result["similarity"] == 1.0
    assert result["match_type"] == "exact"
    assert result["metadata"] == {"source": "test.pdf"}


def test_exact_cache_miss():
    client = MagicMock()
    client.exists.return_value = False

    result = lookup_exact(client, "Unknown query")

    assert result is None
    client.hgetall.assert_not_called()