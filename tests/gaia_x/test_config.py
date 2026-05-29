import os
import pytest
from gateway.gaia_x.config import GaiaXConfig


def test_from_env_defaults(monkeypatch):
    for k in list(os.environ):
        if k.startswith("GAIA_X_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GAIA_X_DOMAIN", "ailiance.fr")
    monkeypatch.setenv("GAIA_X_LEGAL_NAME", "Ailiance")
    monkeypatch.setenv("GAIA_X_VAT_ID", "FR12345678901")
    cfg = GaiaXConfig.from_env()
    assert cfg.domain == "ailiance.fr"
    assert cfg.did == "did:web:ailiance.fr"
    assert cfg.base_url == "https://ailiance.fr"
    assert cfg.well_known_url == "https://ailiance.fr/.well-known"
    assert cfg.legal_name == "Ailiance"
    assert cfg.vat_id == "FR12345678901"
    assert "lab.gaia-x.eu" in cfg.compliance_base_url
    assert "lab.gaia-x.eu" in cfg.notary_base_url


def test_from_env_production_override(monkeypatch):
    monkeypatch.setenv("GAIA_X_DOMAIN", "ailiance.fr")
    monkeypatch.setenv("GAIA_X_LEGAL_NAME", "Ailiance")
    monkeypatch.setenv("GAIA_X_VAT_ID", "FR12345678901")
    monkeypatch.setenv("GAIA_X_COMPLIANCE_BASE_URL", "https://compliance.ovh.gaia-x.eu/v1")
    cfg = GaiaXConfig.from_env()
    assert cfg.compliance_base_url == "https://compliance.ovh.gaia-x.eu/v1"


def test_did_document_url(monkeypatch):
    monkeypatch.setenv("GAIA_X_DOMAIN", "ailiance.fr")
    monkeypatch.setenv("GAIA_X_LEGAL_NAME", "Ailiance")
    monkeypatch.setenv("GAIA_X_VAT_ID", "FR12345678901")
    cfg = GaiaXConfig.from_env()
    assert cfg.did_document_url == "https://ailiance.fr/.well-known/did.json"
    assert cfg.verification_method_id == "did:web:ailiance.fr#JWK2020-RSA"
