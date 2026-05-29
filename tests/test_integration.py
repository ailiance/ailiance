# tests/test_integration.py
"""Smoke tests for the full stack (no model loading)."""


def test_all_domains_mapped():
    from src.router.domain_map import ALL_DOMAINS, DOMAIN_TO_WORKER

    for domain in ALL_DOMAINS:
        assert domain in DOMAIN_TO_WORKER, f"Unmapped domain: {domain}"


def test_worker_configs_cover_all_domains():
    # Post-omlx-consolidation (482f877) the auto-router maps every domain
    # to a fused specialist on the single omlx :8500 server via
    # DOMAIN_TO_OMLX_MODEL; the legacy per-port worker YAMLs
    # (apertus/devstral/eurollm/gemma4) are retired and no longer the
    # routing source of truth. Assert against the live omlx model map.
    from src.router.domain_map import ALL_DOMAINS, DOMAIN_TO_OMLX_MODEL

    for domain in ALL_DOMAINS:
        assert domain in DOMAIN_TO_OMLX_MODEL, (
            f"Domain {domain} not mapped to an omlx specialist model"
        )


def test_gateway_and_worker_import():
    import pytest

    pytest.importorskip("mlx")  # worker/mlx_models pull mlx; skip on CI/linux
    from src.gateway.server import make_gateway_app
    from src.worker.server import make_worker_app
    from src.router.classifier import RouterConfig
    from src.mlx_models.xielu import XIELU
    from src.mlx_models.apertus import ApertusModel, ApertusConfig
