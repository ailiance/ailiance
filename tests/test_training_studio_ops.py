import asyncio
from unittest.mock import patch

import pytest

from src.gateway.training.studio_ops import StudioOps


class _FakeProc:
    def __init__(self, stdout=b"", stderr=b"", returncode=0):
        self._out, self._err, self.returncode = stdout, stderr, returncode

    async def communicate(self):
        return self._out, self._err

    def kill(self):
        pass


@pytest.mark.asyncio
async def test_free_memory_parses_top_output():
    out = b"PhysMem: 412G used (3G wired), 99G unused.\n"
    with patch("asyncio.create_subprocess_exec", return_value=_FakeProc(stdout=out)):
        ops = StudioOps()
        assert await ops.free_memory_gb() == 99.0


@pytest.mark.asyncio
async def test_run_builds_ssh_command():
    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        return _FakeProc(stdout=b"ok\n")

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        ops = StudioOps(ssh_target="clems@studio")
        res = await ops.run("echo ok")
    assert captured["args"][0] == "ssh"
    assert "clems@studio" in captured["args"]
    assert res.stdout.strip() == "ok"


@pytest.mark.asyncio
async def test_pid_alive_true_on_exit_zero():
    with patch("asyncio.create_subprocess_exec", return_value=_FakeProc(returncode=0)):
        assert await StudioOps().pid_alive(123) is True


@pytest.mark.asyncio
async def test_run_timeout_kills_process():
    waited = {"v": False}

    class _SlowProc:
        returncode = -9
        async def communicate(self):
            await asyncio.sleep(10)
        def kill(self):
            pass
        async def wait(self):
            waited["v"] = True

    with patch("asyncio.create_subprocess_exec", return_value=_SlowProc()):
        with pytest.raises(asyncio.TimeoutError):
            await StudioOps().run("sleep 100", timeout=0.05)
    assert waited["v"] is True


@pytest.mark.asyncio
async def test_reload_workers_uses_long_timeout():
    captured = {}

    async def fake_run(self, command, timeout=60.0):
        captured["command"] = command
        captured["timeout"] = timeout
        from src.gateway.training.studio_ops import SSHResult
        return SSHResult(0, "RELOADED 9301\nRELOAD_FAILED 9303\n", "")

    with patch.object(StudioOps, "run", new=fake_run):
        failed = await StudioOps().reload_workers([9301, 9303])
    assert captured["timeout"] >= 2400.0
    assert "reload" in captured["command"]
    assert failed == [9303]
