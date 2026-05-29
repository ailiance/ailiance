# tests/test_router.py
import pytest


def test_domain_map_lookup():
    from src.router.domain_map import DOMAIN_TO_WORKER, OMLX_PORT, get_worker_for_domain

    # omlx consolidation (2026-05-29): all domains in DOMAIN_TO_OMLX_MODEL route
    # to the single omlx :8500 server (OMLX_PORT). Per-port workers are legacy.
    assert get_worker_for_domain("python") == OMLX_PORT
    assert get_worker_for_domain("electronics-hw") == OMLX_PORT
    assert get_worker_for_domain("chat-fr") == OMLX_PORT
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
    assert cfg.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.embedding_dim == 384
    assert cfg.hidden_dim == 256
    assert cfg.num_domains == 47


def test_mascarade_overrides_apertus():
    """The mascarade-specialized domains route to the omlx server (:8500)
    after the omlx consolidation (2026-05-29). Previously routed to
    Tower :8004 (Ollama Q4_K_M), then Studio MLX :9340 (MASCARADE_PORT).
    Now all domains in DOMAIN_TO_OMLX_MODEL — including mascarade hardware
    domains — resolve to OMLX_PORT via the last-write-wins consolidation loop."""
    from src.router.domain_map import (
        APERTUS_DOMAINS,
        MASCARADE_DOMAINS,
        MASCARADE_PORT,
        OMLX_PORT,
        get_worker_for_domain,
    )

    assert MASCARADE_PORT == 9340  # Studio MLX bf16 (was Tower Ollama :8004)
    assert MASCARADE_DOMAINS <= APERTUS_DOMAINS  # subset = override semantics

    # After omlx consolidation, mascarade domains are in DOMAIN_TO_OMLX_MODEL
    # and thus route to OMLX_PORT (last-write-wins over MASCARADE_PORT).
    for d in MASCARADE_DOMAINS:
        assert get_worker_for_domain(d) == OMLX_PORT, (
            f"{d!r} must route to omlx :8500 after consolidation, "
            f"got {get_worker_for_domain(d)}"
        )


def test_kicad_pcb_routes_to_eukiki():
    """'kicad-pcb' routes to omlx :8500 (OMLX_PORT) after the omlx
    consolidation (2026-05-29). Previously routed to macm1 :8502
    (AILIANCE_MACM1_PORT) as a eu-kiki P1 champion (commit 46801af,
    +42 pts). Now served via the omlx multi-model server with model
    'gemma-4-e4b-eukiki-fused' (DOMAIN_TO_OMLX_MODEL)."""
    from src.router.domain_map import OMLX_PORT, get_worker_for_domain

    assert get_worker_for_domain("kicad-pcb") == OMLX_PORT


def test_kicad_dsl_routes_to_eukiki():
    """'kicad-dsl' routes to omlx :8500 (OMLX_PORT) after the omlx
    consolidation (2026-05-29). Previously routed to macm1 :8502
    (AILIANCE_MACM1_PORT) as a eu-kiki P1 champion (commit 46801af,
    +55 pts). Now served via the omlx multi-model server with model
    'gemma-4-e4b-eukiki-fused' (DOMAIN_TO_OMLX_MODEL)."""
    from src.router.domain_map import OMLX_PORT, get_worker_for_domain

    assert get_worker_for_domain("kicad-dsl") == OMLX_PORT


def test_eukiki_domains_all_route_to_8502():
    """AILIANCE_MACM1_DOMAINS (kicad-dsl, kicad-pcb) now route to omlx :8500
    after the omlx consolidation (2026-05-29) — the omlx server handles these
    via DOMAIN_TO_OMLX_MODEL. AILIANCE_MACM1_PORT (:8502) constant is preserved
    for potential rollback; AILIANCE_MACM1_DOMAINS defines the label set."""
    from src.router.domain_map import (
        AILIANCE_MACM1_DOMAINS,
        AILIANCE_MACM1_PORT,
        DOMAIN_TO_OMLX_MODEL,
        OMLX_PORT,
        get_worker_for_domain,
    )

    assert AILIANCE_MACM1_PORT == 8502
    for d in AILIANCE_MACM1_DOMAINS:
        expected = OMLX_PORT if d in DOMAIN_TO_OMLX_MODEL else AILIANCE_MACM1_PORT
        assert get_worker_for_domain(d) == expected, (
            f"{d!r} must route to omlx :8500 after consolidation, "
            f"got {get_worker_for_domain(d)}"
        )


def test_confidence_gating_falls_back_to_apertus():
    """Below the confidence threshold, mascarade domains fall back to
    Apertus. This protects against false-positive specialist routing
    on ambiguous prompts where the bigger generalist is safer.

    omlx consolidation (2026-05-29): high-confidence mascarade domains
    now route to OMLX_PORT (8500) instead of MASCARADE_PORT (9340),
    because the omlx loop is last-write-wins in DOMAIN_TO_WORKER.
    The low-confidence → APERTUS_PORT fallback is unchanged."""
    from src.router.domain_map import (
        APERTUS_PORT,
        MASCARADE_MIN_CONFIDENCE,
        MASCARADE_PORT,
        OMLX_PORT,
        get_worker_for_domain_with_confidence,
    )

    # High confidence → omlx (consolidation: kicad in DOMAIN_TO_OMLX_MODEL → OMLX_PORT)
    assert (
        get_worker_for_domain_with_confidence("kicad", 0.996)
        == OMLX_PORT
    )
    # Below threshold → Apertus fallback (confidence gate still active)
    assert (
        get_worker_for_domain_with_confidence("kicad", 0.50)
        == APERTUS_PORT
    )
    # Exactly at threshold → omlx (>= semantics; above MASCARADE_MIN_CONFIDENCE)
    assert (
        get_worker_for_domain_with_confidence(
            "kicad", MASCARADE_MIN_CONFIDENCE
        )
        == OMLX_PORT
    )
    # Non-mascarade domain ignores threshold; python → omlx after consolidation
    assert (
        get_worker_for_domain_with_confidence("python", 0.01) == OMLX_PORT
    )
    # Empty/None safe
    assert get_worker_for_domain_with_confidence(None, 0.99) is None



def test_eurollm_domains_route_to_eurollm_when_live():
    """omlx consolidation (2026-05-29): EUROLLM_LIVE flag removed from
    domain_map.py. The 4 EUROLLM_DOMAINS (chat-fr, traduction-tech,
    redaction-multilingue, localisation-doc) are now in DOMAIN_TO_OMLX_MODEL
    (model: EuroLLM-22B-Instruct-2512) and route to OMLX_PORT (8500).
    The per-port EUROLLM_PORT (:9303) constant is preserved but no longer
    wired into DOMAIN_TO_WORKER."""
    from src.router.domain_map import (
        EUROLLM_DOMAINS,
        OMLX_PORT,
        get_worker_for_domain,
    )

    for d in EUROLLM_DOMAINS:
        assert get_worker_for_domain(d) == OMLX_PORT, (
            f"{d!r} must route to omlx :8500 after consolidation, "
            f"got {get_worker_for_domain(d)}"
        )
