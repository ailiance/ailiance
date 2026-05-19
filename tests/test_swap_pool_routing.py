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
