"""RSA signing key management and public JWK export for Gaia-X."""
from __future__ import annotations

import base64
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from gateway.gaia_x.config import GaiaXConfig


def ensure_key(path: Path) -> None:
    """Create a 2048-bit RSA private key at *path* if it does not exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.write_bytes(pem)
    path.chmod(0o600)


def _load_private(path: Path):
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def _b64url_uint(value: int) -> str:
    length = (value.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(value.to_bytes(length, "big")).rstrip(b"=").decode()


def public_jwk(path: Path, cfg: GaiaXConfig) -> dict:
    """Return the public RSA key as a Gaia-X JsonWebKey2020 publicKeyJwk."""
    pub = _load_private(path).public_key()
    numbers = pub.public_numbers()
    return {
        "kty": "RSA",
        "n": _b64url_uint(numbers.n),
        "e": _b64url_uint(numbers.e),
        "alg": "PS256",
        "x5u": cfg.x5u_url,
    }
