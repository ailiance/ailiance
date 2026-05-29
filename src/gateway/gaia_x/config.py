"""Configuration for Gaia-X credential issuance."""
from __future__ import annotations

import os
from dataclasses import dataclass

_LAB_COMPLIANCE = "https://compliance.lab.gaia-x.eu/v1/api"
_LAB_NOTARY = "https://registrationnumber.notary.lab.gaia-x.eu/v1"
_LAB_REGISTRY = "https://registry.lab.gaia-x.eu/v1/api"
_LAB_CATALOGUE = "https://catalogue.lab.gaia-x.eu"

# gx: ontology JSON-LD context (production trusted-shape registry)
GX_CONTEXT = (
    "https://registry.lab.gaia-x.eu/v1/api/trusted-shape-registry/"
    "v1/shapes/jsonld/trustframework#"
)
VC_CONTEXT = "https://www.w3.org/2018/credentials/v1"
JWS_CONTEXT = "https://w3id.org/security/suites/jws-2020/v1"


@dataclass(frozen=True)
class GaiaXConfig:
    domain: str
    legal_name: str
    vat_id: str
    country_subdivision_code: str = "FR-75"
    compliance_base_url: str = _LAB_COMPLIANCE
    notary_base_url: str = _LAB_NOTARY
    registry_base_url: str = _LAB_REGISTRY
    catalogue_base_url: str = _LAB_CATALOGUE
    x5u_url: str = ""  # URL to the published X.509 chain; set in M3

    @property
    def did(self) -> str:
        return f"did:web:{self.domain}"

    @property
    def base_url(self) -> str:
        return f"https://{self.domain}"

    @property
    def well_known_url(self) -> str:
        return f"{self.base_url}/.well-known"

    @property
    def did_document_url(self) -> str:
        return f"{self.well_known_url}/did.json"

    @property
    def verification_method_id(self) -> str:
        return f"{self.did}#JWK2020-RSA"

    def well_known(self, filename: str) -> str:
        return f"{self.well_known_url}/{filename}"

    @classmethod
    def from_env(cls) -> "GaiaXConfig":
        domain = os.environ["GAIA_X_DOMAIN"]
        return cls(
            domain=domain,
            legal_name=os.environ["GAIA_X_LEGAL_NAME"],
            vat_id=os.environ["GAIA_X_VAT_ID"],
            country_subdivision_code=os.environ.get(
                "GAIA_X_COUNTRY_SUBDIVISION", "FR-75"
            ),
            compliance_base_url=os.environ.get(
                "GAIA_X_COMPLIANCE_BASE_URL", _LAB_COMPLIANCE
            ),
            notary_base_url=os.environ.get("GAIA_X_NOTARY_BASE_URL", _LAB_NOTARY),
            registry_base_url=os.environ.get("GAIA_X_REGISTRY_BASE_URL", _LAB_REGISTRY),
            catalogue_base_url=os.environ.get(
                "GAIA_X_CATALOGUE_BASE_URL", _LAB_CATALOGUE
            ),
            x5u_url=os.environ.get(
                "GAIA_X_X5U_URL",
                f"https://{domain}/.well-known/x509CertificateChain.pem",
            ),
        )
