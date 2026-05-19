from src.gateway.training.domains import (
    CAMPAIGN_DOMAINS,
    HARDWARE_DOMAINS,
    DONE_DOMAINS,
    LORA_RANK,
    LORA_SCALE,
    LORA_DROPOUT,
)


def test_campaign_has_28_domains_hardware_first():
    assert len(CAMPAIGN_DOMAINS) == 28
    assert len(set(CAMPAIGN_DOMAINS)) == 28  # no duplicates
    assert CAMPAIGN_DOMAINS[:9] == HARDWARE_DOMAINS
    assert CAMPAIGN_DOMAINS[0] == "kicad-dsl"


def test_done_domains_excluded():
    assert DONE_DOMAINS == frozenset({"chat-fr", "cpp", "docker-devops"})
    assert not (set(CAMPAIGN_DOMAINS) & DONE_DOMAINS)


def test_verified_hyperparameters():
    assert LORA_RANK == 16
    assert LORA_SCALE == 32.0
    assert LORA_DROPOUT == 0.01
