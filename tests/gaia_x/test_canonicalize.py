import hashlib

import pytest

from gateway.gaia_x.canonicalize import canonical_digest, normalize

# These tests fetch the W3C VC JSON-LD context over the network because the
# offline context cache (_CACHED_CONTEXTS) is deliberately deferred. Marked
# so offline CI can skip them with `-m 'not network'`.
pytestmark = pytest.mark.network


def test_normalize_is_deterministic():
    doc = {
        "@context": ["https://www.w3.org/2018/credentials/v1"],
        "type": "VerifiableCredential",
        "issuer": "did:web:ailiance.fr",
        "issuanceDate": "2026-05-29T00:00:00.000Z",
        "credentialSubject": {"id": "did:web:ailiance.fr#s"},
    }
    a = normalize(doc)
    b = normalize(dict(reversed(list(doc.items()))))  # key order must not matter
    assert a == b
    assert a.endswith("\n") or a == ""  # N-Quads


def test_canonical_digest_ignores_proof():
    doc = {
        "@context": ["https://www.w3.org/2018/credentials/v1"],
        "type": "VerifiableCredential",
        "issuer": "did:web:ailiance.fr",
        "issuanceDate": "2026-05-29T00:00:00.000Z",
        "credentialSubject": {"id": "did:web:ailiance.fr#s"},
    }
    d1 = canonical_digest(doc)
    doc_with_proof = dict(doc, proof={"type": "JsonWebSignature2020", "jws": "x"})
    d2 = canonical_digest(doc_with_proof)
    assert d1 == d2  # proof is stripped before hashing
    assert len(d1) == 32  # sha256 raw bytes
