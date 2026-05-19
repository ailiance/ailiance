import pytest

from src.gateway.training.orchestrator import TrainingOrchestrator, build_training_503
from src.gateway.training.state import CampaignState


class FakeOps:
    def __init__(self):
        self.reloaded = False
        self.free = 400.0
        self.logs = {}

    async def venv_ok(self):
        return True

    async def free_memory_gb(self):
        return self.free

    async def deploy_scripts(self, _):
        pass

    async def unload_workers(self):
        return [9301, 9303]

    async def reload_workers(self, ports):
        self.reloaded = True
        return []

    async def spawn_domain(self, domain):
        return 4242

    async def pid_alive(self, pid):
        return False  # batch already finished

    async def read_domain_log(self, domain):
        return self.logs.get(domain, "")


@pytest.mark.asyncio
async def test_preflight_fails_on_low_memory(tmp_path):
    ops = FakeOps()
    ops.free = 100.0
    orch = TrainingOrchestrator(ops, tmp_path / "s.json")
    ok = await orch._preflight()
    assert ok is False
    assert orch.state.status == "FAILED"
    assert "320" in orch.state.error


@pytest.mark.asyncio
async def test_gate_records_verdict(tmp_path):
    ops = FakeOps()
    ops.logs["kicad-dsl"] = "### DOMAIN COMPLETE kicad-dsl final_val_loss=0.42\n"
    orch = TrainingOrchestrator(ops, tmp_path / "s.json")
    orch.state = CampaignState(status="GATING", domains=["kicad-dsl"])
    await orch._gate_domain("kicad-dsl")
    assert orch.state.verdicts["kicad-dsl"] == "OK"


def test_build_503_body_has_progress():
    state = CampaignState(status="TRAINING", domains=["kicad-dsl", "kicad-pcb"],
                          domain_index=1, phase=2, iter=340, iter_total=800)
    body = build_training_503(state, "ailiance-mistral-medium")
    assert body["error"]["code"] == "training_in_progress"
    assert body["training"]["current_domain"] == "kicad-pcb"
    assert body["training"]["domain_index"] == 2
    assert body["training"]["domains_total"] == 2
