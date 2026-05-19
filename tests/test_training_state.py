from src.gateway.training.state import CampaignState, load_state, save_state


def test_default_state_is_idle():
    s = CampaignState()
    assert s.status == "IDLE"
    assert s.is_active is False
    assert s.current_domain is None


def test_current_domain_tracks_index():
    s = CampaignState(status="TRAINING", domains=["a", "b", "c"], domain_index=1)
    assert s.current_domain == "b"
    assert s.is_active is True


def test_roundtrip_persistence(tmp_path):
    path = tmp_path / "campaign_state.json"
    s = CampaignState(status="TRAINING", domains=["kicad-dsl"], batch_pid=4242,
                      verdicts={"kicad-dsl": "OK"}, unloaded_ports=[9301, 9303])
    save_state(path, s)
    loaded = load_state(path)
    assert loaded == s


def test_load_missing_returns_idle(tmp_path):
    assert load_state(tmp_path / "absent.json").status == "IDLE"


def test_load_corrupt_returns_idle(tmp_path):
    path = tmp_path / "campaign_state.json"
    path.write_text("{not valid json")
    assert load_state(path).status == "IDLE"
