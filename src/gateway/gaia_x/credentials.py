"""Gaia-X Verifiable Credential builders (unsigned)."""
from __future__ import annotations

from src.gateway.gaia_x.config import GaiaXConfig, GX_CONTEXT, JWS_CONTEXT, VC_CONTEXT

# Fixed Gaia-X-mandated T&C text. Its SHA-256 must match the registry
# canonical text; copy the exact string from the registry at integration.
GAIA_X_TERMS_TEXT = (
    "The PARTICIPANT signing the Self-Description agrees as follows:\n"
    "- to update its descriptions about any changes, be it technical, "
    "organisational, or legal - especially but not limited to contractual "
    "in regards to the indicated attributes present in the descriptions.\n\n"
    "The keypair used to sign Verifiable Credentials will be revoked where "
    "Gaia-X Association becomes aware of any inaccurate statements in regards "
    "to the claims which result in a non-compliance with the Trust Framework "
    "and policy rules defined in the Policy Rules and Labelling Document."
)

_BASE_CONTEXT = [VC_CONTEXT, JWS_CONTEXT, GX_CONTEXT]


def _envelope(
    cfg: GaiaXConfig, file_name: str, issuance_date: str, subject: dict
) -> dict:
    return {
        "@context": list(_BASE_CONTEXT),
        "type": "VerifiableCredential",
        "id": cfg.well_known(file_name),
        "issuer": cfg.did,
        "issuanceDate": issuance_date,
        "credentialSubject": subject,
    }


def build_legal_participant(cfg: GaiaXConfig, issuance_date: str) -> dict:
    subject = {
        "id": cfg.well_known("participant.json") + "#subject",
        "type": "gx:LegalParticipant",
        "gx:legalName": cfg.legal_name,
        "gx:legalRegistrationNumber": {
            "id": cfg.well_known("legal-registration-number.json") + "#subject"
        },
        "gx:headquarterAddress": {
            "gx:countrySubdivisionCode": cfg.country_subdivision_code
        },
        "gx:legalAddress": {
            "gx:countrySubdivisionCode": cfg.country_subdivision_code
        },
    }
    return _envelope(cfg, "participant.json", issuance_date, subject)


def build_terms_and_conditions(cfg: GaiaXConfig, issuance_date: str) -> dict:
    subject = {
        "id": cfg.well_known("gx-terms-and-conditions.json") + "#subject",
        "type": "gx:GaiaXTermsAndConditions",
        "gx:termsAndConditions": GAIA_X_TERMS_TEXT,
    }
    return _envelope(cfg, "gx-terms-and-conditions.json", issuance_date, subject)


def build_service_offering(cfg: GaiaXConfig, issuance_date: str) -> dict:
    subject = {
        "id": cfg.well_known("service-offering.json") + "#subject",
        "type": "gx:ServiceOffering",
        "gx:providedBy": {"id": cfg.well_known("participant.json") + "#subject"},
        "gx:policy": [""],
        "gx:termsAndConditions": [
            {
                "gx:URL": f"{cfg.base_url}/tandc",
                "gx:hash": "",  # filled by CLI with sha256 of T&C doc (later task)
            }
        ],
        "gx:dataAccountExport": {
            "gx:requestType": "API",
            "gx:accessType": "digital",
            "gx:formatType": "application/json",
        },
    }
    return _envelope(cfg, "service-offering.json", issuance_date, subject)
