import pytest
from pathlib import Path
from src.gateway.gaia_x.keys import ensure_key
from src.gateway.gaia_x.signing import sign_credential, verify_credential

pytestmark = pytest.mark.network


def _vc():
    return {
        "@context": ["https://www.w3.org/2018/credentials/v1"],
        "type": "VerifiableCredential",
        "issuer": "did:web:ailiance.fr",
        "issuanceDate": "2026-05-29T00:00:00.000Z",
        "credentialSubject": {"id": "did:web:ailiance.fr#s"},
    }


def test_sign_adds_jws_proof(tmp_path: Path, cfg):
    key_path = tmp_path / "k.pem"
    ensure_key(key_path)
    signed = sign_credential(_vc(), key_path, cfg)
    proof = signed["proof"]
    assert proof["type"] == "JsonWebSignature2020"
    assert proof["proofPurpose"] == "assertionMethod"
    assert proof["verificationMethod"] == cfg.verification_method_id
    assert proof["jws"].count(".") == 2  # detached JWS: header..signature
    assert proof["jws"].split(".")[1] == ""  # payload detached (empty middle)


def test_verify_roundtrip(tmp_path: Path, cfg):
    key_path = tmp_path / "k.pem"
    ensure_key(key_path)
    signed = sign_credential(_vc(), key_path, cfg)
    assert verify_credential(signed, key_path) is True


def test_verify_fails_on_tamper(tmp_path: Path, cfg):
    key_path = tmp_path / "k.pem"
    ensure_key(key_path)
    signed = sign_credential(_vc(), key_path, cfg)
    signed["credentialSubject"]["id"] = "did:web:evil.example#s"
    assert verify_credential(signed, key_path) is False
