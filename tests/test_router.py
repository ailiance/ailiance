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

    # 40 classifier-predicted domains (apertus + devstral + eurollm)
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
