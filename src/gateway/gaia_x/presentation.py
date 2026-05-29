"""Assemble a VerifiablePresentation from signed credentials."""
from __future__ import annotations

from src.gateway.gaia_x.config import VC_CONTEXT


def build_presentation(credentials: list[dict]) -> dict:
    if not credentials:
        raise ValueError("a presentation needs at least one credential")
    return {
        "@context": [VC_CONTEXT],
        "type": "VerifiablePresentation",
        "verifiableCredential": list(credentials),
    }
