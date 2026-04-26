# tests/test_integration.py
"""Smoke tests for the full stack (no model loading)."""


def test_all_domains_mapped():
    from src.router.domain_map import ALL_DOMAINS, DOMAIN_TO_WORKER

    for domain in ALL_DOMAINS:
        assert domain in DOMAIN_TO_WORKER, f"Unmapped domain: {domain}"


def test_worker_configs_cover_all_domains():
    import yaml
    from pathlib import Path
    from src.router.domain_map import ALL_DOMAINS

    config_domains = set()
    for cfg_file in ["configs/apertus.yaml", "configs/devstral.yaml", "configs/eurollm.yaml"]:
        p = Path(cfg_file)
        if p.exists():
            data = yaml.safe_load(p.read_text())
            config_domains.update(data.get("domains", []))

    for domain in ALL_DOMAINS:
        assert domain in config_domains, f"Domain {domain} not in any worker config"


def test_gateway_and_worker_import():
    from src.gateway.server import make_gateway_app
    from src.worker.server import make_worker_app
    from src.router.classifier import RouterConfig
    from src.mlx_models.xielu import XIELU
    from src.mlx_models.apertus import ApertusModel, ApertusConfig
