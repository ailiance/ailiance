"""Swap-pool routing: distinct base models reach the swap server."""
from src.gateway.server import (
    ALIAS_MODEL_REWRITES,
    MODEL_FORCE_MAP,
    WORKER_URLS,
)

SWAP_PORT = 9350

SWAP_ALIASES = [
    "ailiance-llama", "ailiance-mixtral", "ailiance-mixtral-8x22b",
    "ailiance-qwen-235b", "ailiance-flagship", "ailiance-qwen36",
    "ailiance-devstral-base", "ailiance-mistral-small",
]
# Note: EuroLLM is stopped in Phase 1 to free RAM but has no public gateway
# alias (absent from /v1/models), so it is not in SWAP_ALIASES.


def test_swap_port_is_registered():
    assert SWAP_PORT in WORKER_URLS


def test_base_aliases_route_to_swap_port():
    for alias in SWAP_ALIASES:
        assert MODEL_FORCE_MAP.get(alias) == SWAP_PORT, alias


def test_swap_aliases_have_a_model_rewrite():
    for alias in SWAP_ALIASES:
        assert alias in ALIAS_MODEL_REWRITES, alias
        assert ALIAS_MODEL_REWRITES[alias].get("model"), alias
