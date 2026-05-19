import pytest

from src.gateway.training.orchestrator import TrainingOrchestrator, build_training_503
from src.gateway.training.state import CampaignState, save_state


class FakeOps:
    def __init__(self):
        self.reloaded = False
        self.free = 400.0
        self.logs = {}
        self.spawn_calls = 0
        self.alive_limit = 0      # pid_alive returns True for this many calls
        self._alive_calls = 0

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
        self.spawn_calls += 1
        return 4242

    async def pid_alive(self, pid):
        if self._alive_calls < self.alive_limit:
            self._alive_calls += 1
            return True
        return False

    async def read_domain_log(self, domain):
        return self.logs.get(domain, "")


@pytest.mark.asyncio
async def test_campaign_fails_when_unload_frees_too_little(tmp_path, monkeypatch):
    monkeypatch.setattr("src.gateway.training.orchestrator.MEM_SETTLE_POLL_S", 0.0)
    ops = FakeOps()
    ops.free = 100.0  # after unload, still far below the 320 GB requirement
    orch = TrainingOrchestrator(ops, tmp_path / "s.json")
    orch.state = CampaignState(status="PREFLIGHT", domains=["a"])
    await orch._run_campaign()
    assert orch.state.status == "FAILED"
    assert "320" in orch.state.error
    assert ops.reloaded is True  # workers reloaded on failure


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


@pytest.mark.asyncio
async def test_run_campaign_completes(tmp_path, monkeypatch):
    monkeypatch.setattr("src.gateway.training.orchestrator.MEM_SETTLE_POLL_S", 0.0)
    ops = FakeOps()
    ops.logs = {
        "a": "### DOMAIN COMPLETE a final_val_loss=0.40\n",
        "b": "### DOMAIN COMPLETE b final_val_loss=0.50\n",
    }
    orch = TrainingOrchestrator(ops, tmp_path / "s.json")
    orch.state = CampaignState(status="PREFLIGHT", domains=["a", "b"])
    await orch._run_campaign()
    assert orch.state.status == "DONE"
    assert orch.state.verdicts == {"a": "OK", "b": "OK"}
    assert ops.reloaded is True


@pytest.mark.asyncio
async def test_abort_stops_the_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr("src.gateway.training.orchestrator.MEM_SETTLE_POLL_S", 0.0)
    ops = FakeOps()
    orch = TrainingOrchestrator(ops, tmp_path / "s.json")
    orch.state = CampaignState(status="PREFLIGHT", domains=["a", "b"],
                               abort_requested=True)
    await orch._run_campaign()
    assert orch.state.status == "ABORTED"
    assert orch.state.domain_index == 0   # no domain was trained
    assert ops.reloaded is True


@pytest.mark.asyncio
async def test_reattach_resumes_active_campaign(tmp_path):
    ops = FakeOps()
    ops.logs = {"a": "### DOMAIN COMPLETE a final_val_loss=0.40\n"}
    state_path = tmp_path / "s.json"
    save_state(state_path, CampaignState(status="TRAINING", domains=["a"],
                                         batch_pid=999, unloaded_ports=[9301]))
    orch = TrainingOrchestrator(ops, state_path)
    resumed = await orch.reattach()
    assert resumed is True
    await orch._task  # let the resumed campaign run to completion
    assert orch.state.status == "DONE"
    assert orch.state.verdicts == {"a": "OK"}


@pytest.mark.asyncio
async def test_reattach_noop_when_idle(tmp_path):
    orch = TrainingOrchestrator(FakeOps(), tmp_path / "absent.json")
    assert await orch.reattach() is False


@pytest.mark.asyncio
async def test_reattach_polls_live_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.gateway.training.orchestrator.POLL_INTERVAL_S", 0.0)
    ops = FakeOps()
    ops.alive_limit = 2  # the resumed batch is still alive for 2 poll checks
    ops.logs = {"a": "### DOMAIN COMPLETE a final_val_loss=0.40\n"}
    state_path = tmp_path / "s.json"
    save_state(state_path, CampaignState(status="TRAINING", domains=["a"],
                                         batch_pid=999, unloaded_ports=[9301]))
    orch = TrainingOrchestrator(ops, state_path)
    assert await orch.reattach() is True
    await orch._task
    assert ops.spawn_calls == 0  # re-attached to the live batch, no re-spawn
    assert orch.state.status == "DONE"


@pytest.mark.asyncio
async def test_reattach_from_gating_skips_retrain(tmp_path):
    ops = FakeOps()
    ops.logs = {"a": "### DOMAIN COMPLETE a final_val_loss=0.40\n"}
    state_path = tmp_path / "s.json"
    save_state(state_path, CampaignState(status="GATING", domains=["a"],
                                         batch_pid=999, unloaded_ports=[9301]))
    orch = TrainingOrchestrator(ops, state_path)
    assert await orch.reattach() is True
    await orch._task
    assert ops.spawn_calls == 0  # training already finished — not re-trained
    assert orch.state.status == "DONE"
    assert orch.state.verdicts == {"a": "OK"}


@pytest.mark.asyncio
async def test_progress_sets_iter_total(tmp_path):
    orch = TrainingOrchestrator(FakeOps(), tmp_path / "s.json")
    orch.state = CampaignState(status="TRAINING", domains=["a"])
    orch._progress(2, 340)
    assert orch.state.iter == 340
    assert orch.state.iter_total == 800
    orch._progress(1, 50)
    assert orch.state.iter_total == 500


@pytest.mark.asyncio
async def test_reattach_from_reloading_skips_to_reload(tmp_path):
    ops = FakeOps()
    state_path = tmp_path / "s.json"
    save_state(state_path, CampaignState(status="RELOADING", domains=["a", "b"],
                                         domain_index=2, unloaded_ports=[9301]))
    orch = TrainingOrchestrator(ops, state_path)
    assert await orch.reattach() is True
    await orch._task
    assert ops.spawn_calls == 0  # no domain trained — went straight to reload
    assert ops.reloaded is True
    assert orch.state.status == "DONE"


@pytest.mark.asyncio
async def test_gate_records_failed_oom(tmp_path):
    ops = FakeOps()
    ops.logs["a"] = "### PHASE 1/3 domain=a seq=512\n### DOMAIN FAILED_OOM a phase=1\n"
    orch = TrainingOrchestrator(ops, tmp_path / "s.json")
    orch.state = CampaignState(status="GATING", domains=["a"])
    await orch._gate_domain("a")
    assert orch.state.verdicts["a"] == "FAILED_OOM"


@pytest.mark.asyncio
async def test_settled_free_memory_waits_for_rise_to_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.gateway.training.orchestrator.MEM_SETTLE_POLL_S", 0.0)
    ops = FakeOps()
    # free memory climbs 100 -> 200 -> 305 -> 306 then flat (buffers reclaimed)
    readings = iter([100.0, 200.0, 305.0, 306.0, 306.0])

    async def rising():
        return next(readings)
    ops.free_memory_gb = rising
    orch = TrainingOrchestrator(ops, tmp_path / "s.json")
    settled = await orch._settled_free_memory()
    assert settled == 306.0  # stopped once the rise flattened (306 <= 305+2)


@pytest.mark.asyncio
async def test_unload_gates_on_settled_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.gateway.training.orchestrator.MEM_SETTLE_POLL_S", 0.0)
    ops = FakeOps()
    # immediate reading 302 is below the 320 gate, but it settles at 360
    readings = iter([302.0, 340.0, 360.0, 360.0])

    async def rising():
        return next(readings)
    ops.free_memory_gb = rising
    orch = TrainingOrchestrator(ops, tmp_path / "s.json")
    orch.state = CampaignState(status="UNLOADING", domains=["a"])
    await orch._unload()  # must NOT raise — settled memory 360 >= 320
    assert orch.state.status == "UNLOADING"
