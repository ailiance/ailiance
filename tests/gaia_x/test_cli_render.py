import json
from pathlib import Path

import pytest

from gateway.gaia_x.config import GaiaXConfig
from gateway.gaia_x.cli import render_artifacts

# render signs VCs -> canonical_digest fetches the W3C JSON-LD context (network).
pytestmark = pytest.mark.network


def test_render_writes_all_artifacts(tmp_path: Path):
    cfg = GaiaXConfig(domain="ailiance.fr", legal_name="Ailiance",
                      vat_id="FR12345678901",
                      x5u_url="https://ailiance.fr/.well-known/x509CertificateChain.pem")
    out = tmp_path / "well-known"
    key = tmp_path / "k.pem"
    render_artifacts(cfg, out_dir=out, key_path=key, issuance_date="2026-05-29T00:00:00.000Z")
    for name in ["did.json", "participant.json", "gx-terms-and-conditions.json",
                 "service-offering.json"]:
        assert (out / name).exists(), name
    part = json.loads((out / "participant.json").read_text())
    assert part["proof"]["type"] == "JsonWebSignature2020"
    did = json.loads((out / "did.json").read_text())
    assert "proof" not in did
    assert did["id"] == "did:web:ailiance.fr"
