# tests/test_router.py
import pytest


def test_domain_map_lookup():
    from src.router.domain_map import (
        DOMAIN_TO_QWEN36,
        OMLX_PORT,
        QWEN36_PORT,
        get_worker_for_domain,
    )

    # qwen36 hybrid routing (f375c12): domains in DOMAIN_TO_QWEN36 route to
    # QWEN36_PORT (9360); domains in DOMAIN_TO_OMLX_MODEL but NOT in
    # DOMAIN_TO_QWEN36 stay on OMLX_PORT (8500).
    # python is NOT in DOMAIN_TO_QWEN36 → stays omlx :8500.
    assert get_worker_for_domain("python") == OMLX_PORT
    # electronics-hw IS in DOMAIN_TO_QWEN36 → routes to :9360.
    assert get_worker_for_domain("electronics-hw") == QWEN36_PORT
    # chat-fr IS in DOMAIN_TO_QWEN36 → routes to :9360.
    assert get_worker_for_domain("chat-fr") == QWEN36_PORT
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
    """qwen36 hybrid routing (f375c12): mascarade hardware domains that are
    also in DOMAIN_TO_QWEN36 now route to QWEN36_PORT (9360) — the qwen36
    loop is last-write-wins in DOMAIN_TO_WORKER, after omlx consolidation.
    MASCARADE_DOMAINS not in DOMAIN_TO_QWEN36 stay on OMLX_PORT (none currently).
    The confidence-gate still applies: below MASCARADE_MIN_CONFIDENCE the
    fallback is APERTUS_PORT regardless of the final high-confidence target."""
    from src.router.domain_map import (
        APERTUS_DOMAINS,
        DOMAIN_TO_QWEN36,
        MASCARADE_DOMAINS,
        MASCARADE_PORT,
        OMLX_PORT,
        QWEN36_PORT,
        get_worker_for_domain,
    )

    assert MASCARADE_PORT == 9340  # Studio MLX bf16 (was Tower Ollama :8004)
    assert MASCARADE_DOMAINS <= APERTUS_DOMAINS  # subset = override semantics

    # After qwen36 hybrid routing, mascarade domains in DOMAIN_TO_QWEN36
    # route to QWEN36_PORT; those not in DOMAIN_TO_QWEN36 stay on OMLX_PORT.
    for d in MASCARADE_DOMAINS:
        expected = QWEN36_PORT if d in DOMAIN_TO_QWEN36 else OMLX_PORT
        assert get_worker_for_domain(d) == expected, (
            f"{d!r} expected {expected}, got {get_worker_for_domain(d)}"
        )


def test_kicad_pcb_routes_to_eukiki():
    """'kicad-pcb' routes to omlx :8500 (OMLX_PORT). It is intentionally
    excluded from DOMAIN_TO_QWEN36 (broken output noted in f375c12 comment).
    Served via the omlx multi-model server with model 'gemma-4-e4b-eukiki-fused'
    (DOMAIN_TO_OMLX_MODEL). Previously routed to macm1 :8502 (AILIANCE_MACM1_PORT)
    as a eu-kiki P1 champion (commit 46801af, +42 pts)."""
    from src.router.domain_map import OMLX_PORT, get_worker_for_domain

    assert get_worker_for_domain("kicad-pcb") == OMLX_PORT


def test_kicad_dsl_routes_to_eukiki():
    """'kicad-dsl' routes to qwen36 :9360 (QWEN36_PORT) after the qwen36
    hybrid routing change (f375c12). It is in DOMAIN_TO_QWEN36 with adapter
    'qwen36-kicad-dsl'. Previously routed to macm1 :8502 (AILIANCE_MACM1_PORT)
    as a eu-kiki P1 champion (commit 46801af, +55 pts), then omlx :8500."""
    from src.router.domain_map import QWEN36_PORT, get_worker_for_domain

    assert get_worker_for_domain("kicad-dsl") == QWEN36_PORT


def test_eukiki_domains_all_route_to_8502():
    """AILIANCE_MACM1_DOMAINS (kicad-dsl, kicad-pcb): after qwen36 hybrid
    routing (f375c12), kicad-dsl is in DOMAIN_TO_QWEN36 → QWEN36_PORT (9360);
    kicad-pcb is NOT in DOMAIN_TO_QWEN36 (excluded, broken output) → OMLX_PORT
    (8500) via DOMAIN_TO_OMLX_MODEL. AILIANCE_MACM1_PORT (:8502) is preserved
    for rollback; AILIANCE_MACM1_DOMAINS defines the label set."""
    from src.router.domain_map import (
        AILIANCE_MACM1_DOMAINS,
        AILIANCE_MACM1_PORT,
        DOMAIN_TO_OMLX_MODEL,
        DOMAIN_TO_QWEN36,
        OMLX_PORT,
        QWEN36_PORT,
        get_worker_for_domain,
    )

    assert AILIANCE_MACM1_PORT == 8502
    for d in AILIANCE_MACM1_DOMAINS:
        if d in DOMAIN_TO_QWEN36:
            expected = QWEN36_PORT
        elif d in DOMAIN_TO_OMLX_MODEL:
            expected = OMLX_PORT
        else:
            expected = AILIANCE_MACM1_PORT
        assert get_worker_for_domain(d) == expected, (
            f"{d!r} expected {expected}, got {get_worker_for_domain(d)}"
        )


def test_confidence_gating_falls_back_to_apertus():
    """Below the confidence threshold, mascarade domains fall back to
    Apertus. This protects against false-positive specialist routing
    on ambiguous prompts where the bigger generalist is safer.

    qwen36 hybrid routing (f375c12): high-confidence mascarade domains
    now route to QWEN36_PORT (9360) instead of OMLX_PORT (8500), because
    the qwen36 loop is last-write-wins in DOMAIN_TO_WORKER.
    The low-confidence → APERTUS_PORT fallback is unchanged."""
    from src.router.domain_map import (
        APERTUS_PORT,
        MASCARADE_MIN_CONFIDENCE,
        MASCARADE_PORT,
        OMLX_PORT,
        QWEN36_PORT,
        get_worker_for_domain_with_confidence,
    )

    # High confidence + kicad in DOMAIN_TO_QWEN36 → qwen36 :9360
    assert (
        get_worker_for_domain_with_confidence("kicad", 0.996)
        == QWEN36_PORT
    )
    # Below threshold → Apertus fallback (confidence gate still active)
    assert (
        get_worker_for_domain_with_confidence("kicad", 0.50)
        == APERTUS_PORT
    )
    # Exactly at threshold → qwen36 (>= semantics; kicad in DOMAIN_TO_QWEN36)
    assert (
        get_worker_for_domain_with_confidence(
            "kicad", MASCARADE_MIN_CONFIDENCE
        )
        == QWEN36_PORT
    )
    # Non-mascarade domain ignores threshold; python NOT in DOMAIN_TO_QWEN36
    # → omlx :8500 after consolidation
    assert (
        get_worker_for_domain_with_confidence("python", 0.01) == OMLX_PORT
    )
    # Empty/None safe
    assert get_worker_for_domain_with_confidence(None, 0.99) is None



def test_eurollm_domains_route_to_eurollm_when_live():
    """qwen36 hybrid routing (f375c12): the 4 EUROLLM_DOMAINS (chat-fr,
    traduction-tech, redaction-multilingue, localisation-doc) are now in
    DOMAIN_TO_QWEN36 and route to QWEN36_PORT (9360). Previously routed to
    OMLX_PORT via DOMAIN_TO_OMLX_MODEL (EuroLLM-22B-Instruct-2512).
    The per-port EUROLLM_PORT (:9303) constant is preserved but not wired."""
    from src.router.domain_map import (
        EUROLLM_DOMAINS,
        QWEN36_PORT,
        get_worker_for_domain,
    )

    for d in EUROLLM_DOMAINS:
        assert get_worker_for_domain(d) == QWEN36_PORT, (
            f"{d!r} must route to qwen36 :9360, "
            f"got {get_worker_for_domain(d)}"
        )
