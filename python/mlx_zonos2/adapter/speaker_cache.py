"""Speaker embedding cache with blend support.

Implements in-memory session storage for speaker embeddings with:
- ID-based embedding lookup
- Linear interpolation blending (weighted average of two embeddings)
- TTL-based eviction (configurable, default 1 hour)
- Automatic extraction on first use via speaker_encoder
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np


@dataclass
class _CachedEmbedding:
    """Single cached speaker embedding with expiration tracking."""
    embedding: np.ndarray
    created_at: float = field(default_factory=time.monotonic)

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.monotonic() - self.created_at) > self.ttl

    def blend_with(self, other: "_CachedEmbedding", t: float) -> np.ndarray:
        """Linear interpolation: (1-t) * self + t * other."""
        return np.asarray((1.0 - t) * self.embedding + t * other.embedding, dtype=np.float32)


class SpeakerCache:
    """Thread-safe in-memory speaker embedding cache.

    Supports embedding storage by ID, retrieval, and weighted blending
    between two cached embeddings.
    """

    def __init__(self, ttl: float = 3600.0) -> None:
        """Initialize cache.

        Args:
            ttl: Time-to-live in seconds. 0 means no expiration.
        """
        self._ttl = ttl
        self._store: dict[str, _CachedEmbedding] = {}
        self._lock = Lock()

    @property
    def ttl(self) -> float:
        return self._ttl

    def get(self, embedding_id: str) -> np.ndarray | None:
        """Retrieve embedding by ID, or None if not found/expired."""
        with self._lock:
            cached = self._store.get(embedding_id)
            if cached is None or cached.is_expired:
                if cached is not None:
                    del self._store[embedding_id]
                return None
            return cached.embedding.copy()

    def set(self, embedding_id: str, embedding: np.ndarray) -> None:
        """Store embedding in cache with TTL."""
        with self._lock:
            self._store[embedding_id] = _CachedEmbedding(
                embedding=np.asarray(embedding, dtype=np.float32),
                created_at=time.monotonic(),
            )
            self._store[embedding_id].ttl = self._ttl

    def blend(self, id_a: str, id_b: str, t: float) -> np.ndarray | None:
        """Blend two embeddings using weighted interpolation.

        Result = (1 - t) * embedding_a + t * embedding_b

        Args:
            id_a: First embedding ID.
            id_b: Second embedding ID.
            t: Blend weight [0, 1]. 0 = all A, 1 = all B.

        Returns:
            Blended embedding as float32 numpy array, or None if either ID missing.
        """
        if t < 0.0 or t > 1.0:
            raise ValueError(f"Blend weight t must be in [0, 1], got {t}")

        with self._lock:
            cached_a = self._store.get(id_a)
            cached_b = self._store.get(id_b)

            # Evict expired
            if cached_a is not None and cached_a.is_expired:
                del self._store[id_a]
                cached_a = None
            if cached_b is not None and cached_b.is_expired:
                del self._store[id_b]
                cached_b = None

            if cached_a is None or cached_b is None:
                return None

            return cached_a.blend_with(cached_b, t)

    def clear(self) -> int:
        """Remove all entries. Returns number of entries removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count
