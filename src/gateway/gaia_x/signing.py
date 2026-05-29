"""JsonWebSignature2020 detached-JWS proofs over Gaia-X credentials.

Recipe: strip proof -> URDNA2015 normalize -> SHA-256 -> detached JWS (PS256)
over the digest. The digest (raw 32 bytes) is the JWS payload; the payload is
removed from the serialized token (detached form: header..signature).

Verification: reattach the base64url-encoded digest to rebuild the compact
token, then call jwcrypto's JWS.deserialize which checks the PS256 signature.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from jwcrypto import jwk, jws

from gateway.gaia_x.canonicalize import canonical_digest
from gateway.gaia_x.config import GaiaXConfig

_PROTECTED = {"alg": "PS256"}


def _signing_key(key_path: Path) -> jwk.JWK:
    return jwk.JWK.from_pem(key_path.read_bytes())


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _detached_jws(digest: bytes, key: jwk.JWK) -> str:
    """Sign *digest* with PS256 and return a detached compact JWS (header..sig)."""
    token = jws.JWS(digest)
    token.add_signature(key, alg="PS256", protected=json.dumps(_PROTECTED))
    token.detach_payload()
    return token.serialize(compact=True)


def sign_credential(credential: dict, key_path: Path, cfg: GaiaXConfig) -> dict:
    """Return *credential* with a JsonWebSignature2020 proof appended."""
    digest = canonical_digest(credential)
    key = _signing_key(key_path)
    signed = dict(credential)
    signed["proof"] = {
        "type": "JsonWebSignature2020",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "proofPurpose": "assertionMethod",
        "verificationMethod": cfg.verification_method_id,
        "jws": _detached_jws(digest, key),
    }
    return signed


def verify_credential(credential: dict, key_path: Path) -> bool:
    """Verify a JsonWebSignature2020 proof against the local key.

    Local integrity check only — production verification is done by the GXDCH.
    Returns False (never raises) on any verification failure.
    """
    proof = credential.get("proof")
    if not proof or "jws" not in proof:
        return False
    jws_token = proof["jws"]
    parts = jws_token.split(".")
    if len(parts) != 3 or parts[1] != "":
        return False
    header, _, signature = parts

    # canonical_digest strips proof internally, so passing the full credential is fine
    digest = canonical_digest(credential)

    # Reattach the base64url-encoded digest to rebuild a verifiable compact token
    reattached = f"{header}.{_b64url(digest)}.{signature}"
    key = _signing_key(key_path)
    verifier = jws.JWS()
    try:
        verifier.deserialize(reattached, key=key, alg="PS256")
        return True
    except Exception:
        return False
