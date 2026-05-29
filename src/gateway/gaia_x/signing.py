"""JsonWebSignature2020 detached-JWS proofs over Gaia-X credentials.

Recipe: strip proof -> URDNA2015 normalize -> SHA-256 -> detached JWS (PS256)
over the digest. The digest (raw 32 bytes) is the JWS payload; the payload is
removed from the serialized token (RFC 7797 unencoded-detached form:
header..signature, with b64:false and crit:["b64"] in the protected header).

This matches the JsonWebSignature2020 spec as expected by the Gaia-X GXDCH:
the protected header carries {"alg":"PS256","b64":false,"crit":["b64"]}
so the payload is NOT base64url-encoded before signing (RFC 7797 §4).

Verification: use jwcrypto's detached_payload parameter to supply the raw
digest bytes directly — do NOT re-encode as base64url (that would break the
b64:false contract).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jwcrypto import jwk, jws

from gateway.gaia_x.canonicalize import canonical_digest
from gateway.gaia_x.config import GaiaXConfig

# RFC 7797 §3 unencoded payload; required by JsonWebSignature2020 / GXDCH
_PROTECTED = {"alg": "PS256", "b64": False, "crit": ["b64"]}


def _signing_key(key_path: Path) -> jwk.JWK:
    return jwk.JWK.from_pem(key_path.read_bytes())


def _detached_jws(digest: bytes, key: jwk.JWK) -> str:
    """Sign *digest* with PS256 (b64:false) and return detached JWS (header..sig).

    Per RFC 7797 §4 the signing input is:
        ASCII(BASE64URL(protected_header)) || '.' || raw_payload_bytes
    jwcrypto handles this internally when b64:false is in the protected header.
    After signing, detach_payload() removes the middle segment so the serialized
    form is ``header..signature``.
    """
    token = jws.JWS(digest)
    token.add_signature(key, alg="PS256", protected=json.dumps(_PROTECTED))
    token.detach_payload()
    full = token.serialize(compact=True)          # "header..signature"
    header, _mid, signature = full.split(".")
    return f"{header}..{signature}"


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
    Uses RFC 7797 detached-payload verification (b64:false): the raw digest
    bytes are passed directly via detached_payload, not re-encoded as base64url.
    Returns False (never raises) on any verification failure.
    """
    proof = credential.get("proof")
    if not proof or "jws" not in proof:
        return False
    jws_token = proof["jws"]
    parts = jws_token.split(".")
    if len(parts) != 3 or parts[1] != "":
        return False

    # canonical_digest strips proof internally, so passing the full credential is fine
    digest = canonical_digest(credential)

    key = _signing_key(key_path)
    verifier = jws.JWS()
    try:
        verifier.deserialize(jws_token)
        verifier.verify(key, alg="PS256", detached_payload=digest)
        return True
    except Exception:
        return False
