"""did:web DID document builder."""
from __future__ import annotations

from src.gateway.gaia_x.config import GaiaXConfig


def build_did_document(cfg: GaiaXConfig, public_jwk: dict) -> dict:
    """Build the did:web DID document served at /.well-known/did.json."""
    vm_id = cfg.verification_method_id
    return {
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": cfg.did,
        "verificationMethod": [
            {
                "id": vm_id,
                "type": "JsonWebKey2020",
                "controller": cfg.did,
                "publicKeyJwk": public_jwk,
            }
        ],
        "assertionMethod": [vm_id],
    }
