# tests/test_router.py
import pytest


def test_domain_map_lookup():
    from src.router.domain_map import DOMAIN_TO_WORKER, get_worker_for_domain

    assert get_worker_for_domain("python") == 9302
    assert get_worker_for_domain("electronics-hw") == 9301
    # chat-fr routes to EuroLLM (:9303) — Studio worker is up since 2026-05-11 08:50
    assert get_worker_for_domain("chat-fr") == 9303
    assert get_worker_for_domain("unknown-domain") is None


def test_domain_map_completeness():
    from src.router.domain_map import DOMAIN_TO_WORKER, ALL_DOMAINS

    # 40 classifier-predicted domains (apertus + devstral + eurollm + qwen)
    # + 5 GEMMA fallback/utility domains (general, quick, summarize,
    # classification, tldr). The MLP head still emits 40 logits — the
    # 5 extras are routing-only post-fallback targets. Bumps here MUST
    # be reviewed against RouterConfig.num_domains (still 40).
    # + 2 eu-kiki P1 KiCad generation domains added 2026-05-11 (kicad-dsl,
    # kicad-pcb): bench Phase 6 champion, routed to macm1 :8502.
    assert len(ALL_DOMAINS) == 47
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


def test_kicad_pcb_routes_to_eukiki():
    """'kicad-pcb' is now a first-class eu-kiki domain routed to :8502
    (macm1 Gemma-4 E4B + curriculum LoRA). It was previously aliased to
    'kicad' → Mascarade :8004. Bench Phase 6 (commit 46801af) shows eu-kiki
    wins P1 generation by +42 pts — kicad-pcb alias removed, direct mapping
    added to AILIANCE_MACM1_DOMAINS."""
    from src.router.domain_map import AILIANCE_MACM1_PORT, get_worker_for_domain

    assert get_worker_for_domain("kicad-pcb") == AILIANCE_MACM1_PORT


def test_kicad_dsl_routes_to_eukiki():
    """'kicad-dsl' is a new P1 generation label routed to :8502
    (macm1 Gemma-4 E4B + curriculum LoRA). Bench Phase 6 (commit 46801af)
    shows eu-kiki wins P1 generation by +55 pts vs base Gemma-E4B."""
    from src.router.domain_map import AILIANCE_MACM1_PORT, get_worker_for_domain

    assert get_worker_for_domain("kicad-dsl") == AILIANCE_MACM1_PORT


def test_eukiki_domains_all_route_to_8502():
    """All AILIANCE_MACM1_DOMAINS must resolve to :8502 — regression guard."""
    from src.router.domain_map import AILIANCE_MACM1_DOMAINS, AILIANCE_MACM1_PORT, get_worker_for_domain

    assert AILIANCE_MACM1_PORT == 8502
    for d in AILIANCE_MACM1_DOMAINS:
        assert get_worker_for_domain(d) == AILIANCE_MACM1_PORT, (
            f"{d!r} must route to eu-kiki macm1 :8502, "
            f"got {get_worker_for_domain(d)}"
        )


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



def test_eurollm_domains_route_to_eurollm_when_live():
    """EuroLLM :9303 (Studio) was restored 2026-05-11 08:50 CEST after
    the morning bench freed it. The 4 EUROLLM_DOMAINS (chat-fr,
    traduction-tech, redaction-multilingue, localisation-doc) should now
    route to EuroLLM 22B again instead of falling back to Gemma 4B.

    Flip EUROLLM_LIVE=False in domain_map.py if :9303 dies again — the
    sister assertion below will fail loudly and remind you to invert
    this test back to the GEMMA_PORT expectation."""
    from src.router.domain_map import (
        EUROLLM_DOMAINS,
        EUROLLM_LIVE,
        EUROLLM_PORT,
        get_worker_for_domain,
    )

    assert EUROLLM_LIVE is True, (
        "EUROLLM_LIVE=False means Studio :9303 is down again — "
        "this test must then be inverted to expect GEMMA_PORT."
    )
    for d in EUROLLM_DOMAINS:
        assert get_worker_for_domain(d) == EUROLLM_PORT, (
            f"{d!r} must route to EuroLLM when live, "
            f"got {get_worker_for_domain(d)}"
        )
