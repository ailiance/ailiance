from src.gateway.gaia_x.presentation import build_presentation


def test_presentation_wraps_credentials():
    vc1 = {"type": "VerifiableCredential", "id": "a"}
    vc2 = {"type": "VerifiableCredential", "id": "b"}
    vp = build_presentation([vc1, vc2])
    assert vp["type"] == "VerifiablePresentation"
    assert "https://www.w3.org/2018/credentials/v1" in vp["@context"]
    assert vp["verifiableCredential"] == [vc1, vc2]


def test_presentation_rejects_empty():
    import pytest
    with pytest.raises(ValueError):
        build_presentation([])
