"""URDNA2015 canonicalization + SHA-256 digest for Gaia-X credentials.

A cached JSON-LD document loader makes normalization deterministic and
offline-capable. New @context URLs must be added to ``_CACHED_CONTEXTS``.
"""
from __future__ import annotations

import copy
import hashlib
from functools import lru_cache

from pyld import jsonld

# Minimal cached contexts. At integration time, fetch each URL once and
# paste the JSON here, or point _document_loader at an on-disk cache dir.
# Keeping them inline guarantees tests never hit the network.
_CACHED_CONTEXTS: dict[str, dict] = {
    # Populated at integration time from the real context bodies.
}


@lru_cache(maxsize=1)
def _base_loader():
    return jsonld.requests_document_loader()


def _document_loader(url, options=None):
    if url in _CACHED_CONTEXTS:
        return {
            "contextUrl": None,
            "documentUrl": url,
            "document": _CACHED_CONTEXTS[url],
        }
    return _base_loader()(url, options or {})


def normalize(doc: dict) -> str:
    """Return URDNA2015-normalized N-Quads for *doc*."""
    return jsonld.normalize(
        doc,
        {
            "algorithm": "URDNA2015",
            "format": "application/n-quads",
            "documentLoader": _document_loader,
        },
    )


def canonical_digest(doc: dict) -> bytes:
    """SHA-256 over the URDNA2015 N-Quads of *doc*, with any proof removed."""
    payload = copy.deepcopy(doc)
    payload.pop("proof", None)
    return hashlib.sha256(normalize(payload).encode("utf-8")).digest()
