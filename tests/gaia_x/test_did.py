from pathlib import Path
from gateway.gaia_x.keys import ensure_key, public_jwk
from gateway.gaia_x.did import build_did_document


def test_did_document_shape(tmp_path: Path, cfg):
    key_path = tmp_path / "k.pem"
    ensure_key(key_path)
    jwk = public_jwk(key_path, cfg)
    doc = build_did_document(cfg, jwk)
    assert doc["id"] == "did:web:ailiance.fr"
    assert doc["@context"] == ["https://www.w3.org/ns/did/v1"]
    vm = doc["verificationMethod"][0]
    assert vm["id"] == "did:web:ailiance.fr#JWK2020-RSA"
    assert vm["type"] == "JsonWebKey2020"
    assert vm["controller"] == "did:web:ailiance.fr"
    assert vm["publicKeyJwk"]["kty"] == "RSA"
    assert vm["publicKeyJwk"]["x5u"].endswith("x509CertificateChain.pem")
    assert doc["assertionMethod"] == ["did:web:ailiance.fr#JWK2020-RSA"]
