import pytest
from gateway.gaia_x.config import GaiaXConfig


@pytest.fixture
def cfg() -> GaiaXConfig:
    return GaiaXConfig(
        domain="ailiance.fr",
        legal_name="Ailiance",
        vat_id="FR12345678901",
        x5u_url="https://ailiance.fr/.well-known/x509CertificateChain.pem",
    )
