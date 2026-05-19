"""ailiance-html / ailiance-ml-training fall back to Gemma.

The devstral-html-css and devstral-ml-training LoRA adapters are
degenerate — verified 2026-05-19 by a direct worker call on :9330:
both emit repetition-loop garbage ("ursprüng ursprüng ...") instead
of usable output. Until the adapters are retrained, these two aliases
route to the Gemma worker (:9304) rather than the broken adapter.
"""
from src.gateway.server import ALIAS_MODEL_REWRITES, MODEL_FORCE_MAP

GEMMA_PORT = 9304
BROKEN_ADAPTER_ALIASES = ["ailiance-html", "ailiance-ml-training"]


def test_broken_devstral_aliases_route_to_gemma():
    for alias in BROKEN_ADAPTER_ALIASES:
        assert MODEL_FORCE_MAP.get(alias) == GEMMA_PORT, alias


def test_broken_devstral_aliases_have_no_devstral_rewrite():
    # With no per-alias rewrite, the :9304 port-level override
    # (eu-kiki-gemma) applies. A leftover devstral adapter name would
    # be forwarded to the Gemma worker, which would 404 on it.
    for alias in BROKEN_ADAPTER_ALIASES:
        assert alias not in ALIAS_MODEL_REWRITES, alias


def test_working_devstral_aliases_still_route_to_multilora():
    # The three healthy adapters keep their :9330 multi-LoRA routing.
    for alias in ("ailiance-python", "ailiance-cpp", "ailiance-rust-emb"):
        assert MODEL_FORCE_MAP.get(alias) == 9330, alias
