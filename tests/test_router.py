# tests/test_router.py
import pytest


def test_domain_map_lookup():
    from src.router.domain_map import (
        DOMAIN_TO_QWEN36,
        OMLX_PORT,
        QWEN36_PORT,
        QWEN36_PORT_B,
        QWEN36_DOMAINS_B,
        get_worker_for_domain,
    )

    # Two-instance qwen36 split (79f4136): hardware/EDA/math domains stay on
    # QWEN36_PORT (9360); code/web/devops/ml/language domains in
    # QWEN36_DOMAINS_B route to QWEN36_PORT_B (9361). Domains in
    # DOMAIN_TO_OMLX_MODEL but NOT in DOMAIN_TO_QWEN36 stay on OMLX_PORT (8500).
    # python is NOT in DOMAIN_TO_QWEN36 → stays omlx :8500.
    assert get_worker_for_domain("python") == OMLX_PORT
    # electronics-hw is a hardware domain → instance A :9360.
    assert "electronics-hw" not in QWEN36_DOMAINS_B
    assert get_worker_for_domain("electronics-hw") == QWEN36_PORT
    # chat-fr is a language domain → instance B :9361.
    assert "chat-fr" in QWEN36_DOMAINS_B
    assert get_worker_for_domain("chat-fr") == QWEN36_PORT_B
    assert get_worker_for_domain("unknown-domain") is None


def test_domain_map_completeness():
    from src.router.domain_map import DOMAIN_TO_WORKER, ALL_DOMAINS

    # 47 domains total — the canonical set the MLP head emits logits over,
    # derived from DOMAIN_TO_OMLX_MODEL keys. Bumps here MUST be reviewed
    # against RouterConfig.num_domains (47) and the trained classifier
    # (retraining required when the domain set changes).
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


def test_hardware_specialist_domains_route_to_qwen36():
    """The former mascarade hardware specialists (kicad, stm32, emc, embedded,
    platformio, freecad, dsp, iot, power) are all in DOMAIN_TO_QWEN36 and route
    to the qwen36 hardware instance :9360 (QWEN36_PORT) — none are in the
    code/language split QWEN36_DOMAINS_B (:9361)."""
    from src.router.domain_map import (
        DOMAIN_TO_QWEN36,
        QWEN36_DOMAINS_B,
        QWEN36_PORT,
        get_worker_for_domain,
    )

    hardware = {"kicad", "stm32", "emc", "embedded",
                "platformio", "freecad", "dsp", "iot", "power"}
    for d in hardware:
        assert d in DOMAIN_TO_QWEN36, f"{d!r} should be qwen36-routed"
        assert d not in QWEN36_DOMAINS_B, f"{d!r} is hardware → instance A"
        assert get_worker_for_domain(d) == QWEN36_PORT, (
            f"{d!r} expected {QWEN36_PORT}, got {get_worker_for_domain(d)}"
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


def test_kicad_generation_domains_route():
    """The two KiCad generation labels split across backends: kicad-dsl is in
    DOMAIN_TO_QWEN36 → qwen36 :9360 (QWEN36_PORT); kicad-pcb is intentionally
    excluded (broken qwen36 output) and stays on omlx :8500 (OMLX_PORT) via
    DOMAIN_TO_OMLX_MODEL 'gemma-4-e4b-eukiki-fused'."""
    from src.router.domain_map import (
        DOMAIN_TO_QWEN36,
        OMLX_PORT,
        QWEN36_PORT,
        get_worker_for_domain,
    )

    assert get_worker_for_domain("kicad-dsl") == QWEN36_PORT
    assert "kicad-dsl" in DOMAIN_TO_QWEN36
    assert get_worker_for_domain("kicad-pcb") == OMLX_PORT
    assert "kicad-pcb" not in DOMAIN_TO_QWEN36


def test_language_domains_route_to_qwen36_instance_b():
    """The 4 EU-language domains (chat-fr, traduction-tech,
    redaction-multilingue, localisation-doc) are in QWEN36_DOMAINS_B and route
    to the qwen36 language instance :9361 (QWEN36_PORT_B)."""
    from src.router.domain_map import (
        QWEN36_DOMAINS_B,
        QWEN36_PORT_B,
        get_worker_for_domain,
    )

    language = {"chat-fr", "traduction-tech",
                "redaction-multilingue", "localisation-doc"}
    for d in language:
        assert d in QWEN36_DOMAINS_B, f"{d!r} should be instance B"
        assert get_worker_for_domain(d) == QWEN36_PORT_B, (
            f"{d!r} expected {QWEN36_PORT_B}, got {get_worker_for_domain(d)}"
        )
