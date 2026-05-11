# tests/test_router.py
import pytest


def test_domain_map_lookup():
    from src.router.domain_map import DOMAIN_TO_WORKER, get_worker_for_domain

    assert get_worker_for_domain("python") == 9302
    assert get_worker_for_domain("electronics-hw") == 9301
    assert get_worker_for_domain("chat-fr") == 9303
    assert get_worker_for_domain("unknown-domain") is None


def test_domain_map_completeness():
    from src.router.domain_map import DOMAIN_TO_WORKER, ALL_DOMAINS

    # 40 classifier-predicted domains (apertus + devstral + eurollm + qwen)
    # + 5 GEMMA fallback/utility domains (general, quick, summarize,
    # classification, tldr). The MLP head still emits 40 logits — the
    # 5 extras are routing-only post-fallback targets. Bumps here MUST
    # be reviewed against RouterConfig.num_domains (still 40).
    assert len(ALL_DOMAINS) == 45
    for domain in ALL_DOMAINS:
        assert domain in DOMAIN_TO_WORKER, f"Missing mapping for {domain}"


def test_classifier_config():
    from src.router.classifier import RouterConfig

    cfg = RouterConfig()
    assert cfg.embedding_model == "jinaai/jina-embeddings-v3"
    assert cfg.embedding_dim == 1024
    assert cfg.hidden_dim == 512
    assert cfg.num_domains == 40


def test_mascarade_overrides_apertus():
    """The 10 mascarade-specialized domains must route to Tower (:8004),
    overriding the default Apertus mapping. Otherwise the auto-router
    silently sends KiCad/SPICE/STM32 prompts to the 128B generalist
    instead of the domain-fine-tuned LoRA."""
    from src.router.domain_map import (
        APERTUS_DOMAINS,
        APERTUS_PORT,
        MASCARADE_DOMAINS,
        MASCARADE_PORT,
        get_worker_for_domain,
    )

    assert MASCARADE_PORT == 8004
    assert MASCARADE_DOMAINS <= APERTUS_DOMAINS  # subset = override semantics

    for d in MASCARADE_DOMAINS:
        assert get_worker_for_domain(d) == MASCARADE_PORT, (
            f"{d!r} must route to Mascarade Tower tunnel, "
            f"got {get_worker_for_domain(d)}"
        )

    # Apertus domains NOT in mascarade keep Apertus
    apertus_only = APERTUS_DOMAINS - MASCARADE_DOMAINS
    assert apertus_only, "sanity: at least one pure-Apertus domain remains"
    for d in apertus_only:
        assert get_worker_for_domain(d) == APERTUS_PORT


def test_kicad_pcb_alias_routes_to_mascarade():
    """Regression: the bare 'kicad-pcb' surface form (emitted by the
    Jina v3 classifier head) must canonicalize to 'kicad' AND end up on
    Mascarade Tower, not Apertus Studio."""
    from src.router.domain_map import MASCARADE_PORT, get_worker_for_domain

    assert get_worker_for_domain("kicad-pcb") == MASCARADE_PORT


def test_confidence_gating_falls_back_to_apertus():
    """Below the confidence threshold, mascarade domains fall back to
    Apertus. This protects against false-positive specialist routing
    on ambiguous prompts where the bigger generalist is safer."""
    from src.router.domain_map import (
        APERTUS_PORT,
        MASCARADE_MIN_CONFIDENCE,
        MASCARADE_PORT,
        get_worker_for_domain_with_confidence,
    )

    # High confidence → Mascarade
    assert (
        get_worker_for_domain_with_confidence("kicad", 0.996)
        == MASCARADE_PORT
    )
    # Below threshold → Apertus fallback
    assert (
        get_worker_for_domain_with_confidence("kicad", 0.50)
        == APERTUS_PORT
    )
    # Exactly at threshold → Mascarade (>= semantics)
    assert (
        get_worker_for_domain_with_confidence(
            "kicad", MASCARADE_MIN_CONFIDENCE
        )
        == MASCARADE_PORT
    )
    # Non-mascarade domain ignores threshold
    assert (
        get_worker_for_domain_with_confidence("python", 0.01) == 9302
    )
    # Empty/None safe
    assert get_worker_for_domain_with_confidence(None, 0.99) is None
