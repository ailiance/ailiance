from src.gateway.gaia_x.credentials import (
    build_legal_participant,
    build_terms_and_conditions,
    build_service_offering,
    GAIA_X_TERMS_TEXT,
)


def test_legal_participant(cfg):
    vc = build_legal_participant(cfg, issuance_date="2026-05-29T00:00:00.000Z")
    assert vc["issuer"] == "did:web:ailiance.fr"
    assert vc["type"] == "VerifiableCredential"
    cs = vc["credentialSubject"]
    assert cs["type"] == "gx:LegalParticipant"
    assert cs["gx:legalName"] == "Ailiance"
    assert cs["gx:legalRegistrationNumber"]["id"].endswith(
        "legal-registration-number.json#subject"
    )
    assert cs["gx:headquarterAddress"]["gx:countrySubdivisionCode"] == "FR-75"
    assert "https://www.w3.org/2018/credentials/v1" in vc["@context"]


def test_terms_and_conditions(cfg):
    vc = build_terms_and_conditions(cfg, issuance_date="2026-05-29T00:00:00.000Z")
    cs = vc["credentialSubject"]
    assert cs["type"] == "gx:GaiaXTermsAndConditions"
    assert cs["gx:termsAndConditions"] == GAIA_X_TERMS_TEXT


def test_service_offering(cfg):
    vc = build_service_offering(cfg, issuance_date="2026-05-29T00:00:00.000Z")
    cs = vc["credentialSubject"]
    assert cs["type"] == "gx:ServiceOffering"
    assert cs["gx:providedBy"]["id"].endswith("participant.json#subject")
    assert cs["gx:dataAccountExport"]["gx:requestType"] == "API"
    assert cs["gx:dataAccountExport"]["gx:formatType"] == "application/json"
