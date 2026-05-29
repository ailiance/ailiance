from pathlib import Path
from gateway.gaia_x.keys import ensure_key, public_jwk


def test_ensure_key_creates_rsa_pem(tmp_path: Path):
    key_path = tmp_path / "gaia-x-signing.pem"
    assert not key_path.exists()
    ensure_key(key_path)
    assert key_path.exists()
    pem = key_path.read_text()
    assert "BEGIN PRIVATE KEY" in pem
    # idempotent: second call does not overwrite
    before = key_path.read_bytes()
    ensure_key(key_path)
    assert key_path.read_bytes() == before


def test_public_jwk_shape(tmp_path: Path, cfg):
    key_path = tmp_path / "k.pem"
    ensure_key(key_path)
    jwk = public_jwk(key_path, cfg)
    assert jwk["kty"] == "RSA"
    assert jwk["alg"] == "PS256"
    assert jwk["e"]
    assert jwk["n"]
    assert jwk["x5u"] == cfg.x5u_url
    assert "d" not in jwk  # public only, no private material
