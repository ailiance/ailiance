# tests/router/test_router_cache.py
"""Tests for the L1 LRU cache on DomainRouter.route().

These tests construct a DomainRouter without going through _load() so
they don't need the real Jina v3 model (~5GB). The encoder + MLP are
replaced by lightweight stubs that count invocations.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from src.router.classifier import (
    DomainRouter,
    RouterConfig,
    _CACHE_MAXSIZE,
)


class _FakeRouter(DomainRouter):
    """DomainRouter that bypasses model loading."""

    def __init__(self, domains: list[str] | None = None) -> None:
        # Skip super().__init__ — no weights_dir, no Jina, no MLP.
        self._cfg = RouterConfig()
        self._encoder = None
        self._mlp = None
        self._domains = domains or [f"d{i}" for i in range(40)]
        self.compute_calls = 0
        # L2 semantic cache state — initialised by the real router but
        # skipped here. prewarm() reads _l2_embs.shape, so default to
        # None so the "L2 disabled" branch fires.
        self._l2_embs = None
        self._l2_hashes: list[str] = []
        self._cached_route_by_hash = lru_cache(maxsize=_CACHE_MAXSIZE)(
            self._route_by_hash
        )

    def _compute_route(self, query: str) -> list[tuple[str, float]]:
        # Deterministic fake: top score depends on query length.
        self.compute_calls += 1
        score = min(0.99, 0.20 + 0.001 * len(query))
        return [(self._domains[0], score), (self._domains[1], score / 2)]


def test_cache_hit_returns_same_result_as_miss():
    r = _FakeRouter()
    out1 = r.route("hello world")
    out2 = r.route("hello world")
    assert out1 == out2
    assert r.compute_calls == 1  # second call served from cache


def test_cache_hit_increments_hit_counter():
    r = _FakeRouter()
    assert r.cache_info()["hits"] == 0
    assert r.cache_info()["misses"] == 0

    r.route("foo")
    info = r.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 0
    assert info["currsize"] == 1

    r.route("foo")
    info = r.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 1
    assert info["currsize"] == 1

    r.route("bar")
    info = r.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 2
    assert info["currsize"] == 2


def test_prewarm_populates_cache():
    r = _FakeRouter()
    prompts = ["alpha", "beta", "gamma", "delta"]
    n = r.prewarm(prompts)
    assert n == len(prompts)

    info = r.cache_info()
    assert info["currsize"] == len(prompts)
    assert info["misses"] == len(prompts)
    assert info["hits"] == 0

    # Subsequent route() calls on prewarmed prompts hit the cache.
    r.route("alpha")
    r.route("beta")
    info = r.cache_info()
    assert info["hits"] == 2


def test_cache_clear_resets_state():
    r = _FakeRouter()
    r.route("x")
    r.route("y")
    assert r.cache_info()["currsize"] == 2
    r.cache_clear()
    info = r.cache_info()
    assert info["currsize"] == 0
    assert info["hits"] == 0
    assert info["misses"] == 0


def test_cache_maxsize_matches_config():
    r = _FakeRouter()
    assert r.cache_info()["maxsize"] == _CACHE_MAXSIZE


def test_distinct_queries_produce_distinct_cache_entries():
    r = _FakeRouter()
    r.route("short")
    r.route("a much longer query than the first one")
    info = r.cache_info()
    assert info["currsize"] == 2
    assert info["misses"] == 2
