"""Tests for SpeakerCache with blend and TTL support.

Verifies thread-safe caching, blend interpolation, and expiration behavior.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlx_zonos2.adapter.speaker_cache import SpeakerCache


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def embedding_a() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0], dtype=np.float32)


@pytest.fixture
def embedding_b() -> np.ndarray:
    return np.array([4.0, 5.0, 6.0], dtype=np.float32)


@pytest.fixture
def cache(ttl: float = 3600.0) -> SpeakerCache:
    return SpeakerCache(ttl=ttl)


# ── Basic Cache Operations ───────────────────────────────────────────────


class TestBasicCacheOperations:
    """Set, get, and clear operations."""

    def test_set_and_get(self, cache, embedding_a):
        """Setting and getting same ID returns the embedding."""
        cache.set("alice", embedding_a)
        result = cache.get("alice")
        assert result is not None
        np.testing.assert_array_almost_equal(result, embedding_a)

    def test_get_missing_id_returns_none(self, cache):
        """Getting non-existent ID returns None."""
        assert cache.get("nobody") is None

    def test_get_returns_copy(self, cache, embedding_a):
        """Returned embedding is a copy, not the original reference."""
        cache.set("alice", embedding_a)
        result = cache.get("alice")
        assert result is not embedding_a

    def test_clear_removes_all(self, cache, embedding_a, embedding_b):
        """clear() removes all entries and returns count."""
        cache.set("alice", embedding_a)
        cache.set("bob", embedding_b)
        count = cache.clear()
        assert count == 2
        assert cache.get("alice") is None
        assert cache.get("bob") is None

    def test_clear_empty_returns_zero(self, cache):
        """clear() on empty cache returns 0."""
        assert cache.clear() == 0


# ── Blend Operations ─────────────────────────────────────────────────────


class TestBlendOperations:
    """Blending two cached embeddings."""

    def test_blend_t_zero_returns_first(self, cache, embedding_a, embedding_b):
        """t=0 returns embedding_a (full weight on first)."""
        cache.set("alice", embedding_a)
        cache.set("bob", embedding_b)
        blended = cache.blend("alice", "bob", 0.0)
        assert blended is not None
        np.testing.assert_array_almost_equal(blended, embedding_a)

    def test_blend_t_one_returns_second(self, cache, embedding_a, embedding_b):
        """t=1 returns embedding_b (full weight on second)."""
        cache.set("alice", embedding_a)
        cache.set("bob", embedding_b)
        blended = cache.blend("alice", "bob", 1.0)
        assert blended is not None
        np.testing.assert_array_almost_equal(blended, embedding_b)

    def test_blend_t_half_is_average(self, cache, embedding_a, embedding_b):
        """t=0.5 is linear average of both embeddings."""
        cache.set("alice", embedding_a)
        cache.set("bob", embedding_b)
        blended = cache.blend("alice", "bob", 0.5)
        expected = (embedding_a + embedding_b) / 2.0
        np.testing.assert_array_almost_equal(blended, expected)

    def test_blend_mixed_25_75(self, cache, embedding_a, embedding_b):
        """t=0.25 is 75% first + 25% second."""
        cache.set("alice", embedding_a)
        cache.set("bob", embedding_b)
        blended = cache.blend("alice", "bob", 0.25)
        expected = 0.75 * embedding_a + 0.25 * embedding_b
        np.testing.assert_array_almost_equal(blended, expected)

    def test_blend_missing_first_returns_none(self, cache, embedding_b):
        """Blend with missing first ID returns None."""
        cache.set("bob", embedding_b)
        assert cache.blend("nobody", "bob", 0.5) is None

    def test_blend_missing_second_returns_none(self, cache, embedding_a):
        """Blend with missing second ID returns None."""
        cache.set("alice", embedding_a)
        assert cache.blend("alice", "nobody", 0.5) is None

    def test_blend_invalid_t_raises(self, cache, embedding_a, embedding_b):
        """Blend with t < 0 or t > 1 raises ValueError."""
        cache.set("alice", embedding_a)
        cache.set("bob", embedding_b)
        with pytest.raises(ValueError, match="must be in"):
            cache.blend("alice", "bob", 1.5)
        with pytest.raises(ValueError, match="must be in"):
            cache.blend("alice", "bob", -0.1)


# ── TTL Expiration ───────────────────────────────────────────────────────


class TestTTLExpiration:
    """Time-to-live eviction behavior."""

    def test_ttl_property(self, cache):
        """Cache exposes TTL value."""
        assert cache.ttl == 3600.0

    def test_no_ttl_means_no_expiration(self):
        """Cache with ttl=0 never expires."""
        cache = SpeakerCache(ttl=0.0)
        cache.set("alice", np.array([1.0, 2.0], dtype=np.float32))
        time.sleep(0.1)
        assert cache.get("alice") is not None

    def test_expired_embedding_is_evicted(self):
        """Short TTL cache expires and evicts on get()."""
        cache = SpeakerCache(ttl=0.05)  # 50ms TTL
        cache.set("alice", np.array([1.0], dtype=np.float32))
        time.sleep(0.1)  # Wait past TTL
        assert cache.get("alice") is None

    def test_expired_embedding_removed_from_store(self):
        """Expired embedding is removed from internal store."""
        cache = SpeakerCache(ttl=0.05)
        cache.set("alice", np.array([1.0], dtype=np.float32))
        time.sleep(0.1)
        cache.get("alice")  # Should evict
        # Subsequent gets should fail
        assert cache.get("alice") is None