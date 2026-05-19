import asyncio
from unittest.mock import patch

import pytest

from src.gateway.training.studio_ops import SSHResult, StudioOps


class _FakeProc:
    def __init__(self, stdout=b"", stderr=b"", returncode=0):
        self._out, self._err, self.returncode = stdout, stderr, returncode

    async def communicate(self):
        return self._out, self._err

    def kill(self):
        pass


def _make_fake_run(stdout):
    async def fake_run(self, command, timeout=60.0):
        return SSHResult(0, stdout, "")
    return fake_run


_VM_STAT_SAMPLE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                            16973156.
Pages active:                           3664794.
Pages inactive:                        12152962.
Pages speculative:                        98729.
Pages throttled:                               0.
Pages wired down:                        474239.
Pages purgeable:                            3145.
Pages stored in compressor:               107320.
File-backed pages:                     12136776.
Anonymous pages:                        3779709.
"""


def test_parse_vm_stat_available_gb():
    from src.gateway.training.studio_ops import parse_vm_stat_available_gb
    gb = parse_vm_stat_available_gb(_VM_STAT_SAMPLE)
    # free 16973156 + inactive 12152962 + speculative 98729 + purgeable 3145
    # = 29227992 pages * 16384 / 1024^3 ~= 445.9 GB
    assert 445.0 < gb < 447.0


def test_parse_vm_stat_unparseable_returns_zero():
    from src.gateway.training.studio_ops import parse_vm_stat_available_gb
    assert parse_vm_stat_available_gb("garbage output") == 0.0


@pytest.mark.asyncio
async def test_free_memory_gb_uses_vm_stat():
    from src.gateway.training.studio_ops import StudioOps
    with patch.object(
        StudioOps, "run",
        new=_make_fake_run(_VM_STAT_SAMPLE),
    ):
        gb = await StudioOps().free_memory_gb()
    assert 445.0 < gb < 447.0


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
async def test_reload_workers_timeout_scales_with_ports():
    captured = {}

    async def fake_run(self, command, timeout=60.0):
        captured["command"] = command
        captured["timeout"] = timeout
        from src.gateway.training.studio_ops import SSHResult
        return SSHResult(0, "RELOADED 9301\nRELOAD_FAILED 9303\n", "")

    with patch.object(StudioOps, "run", new=fake_run):
        failed = await StudioOps().reload_workers([9301, 9303])
    # 120 + 320*2 = 760s for 2 ports; must exceed the 9-worker worst case too
    assert captured["timeout"] == 120.0 + 320.0 * 2
    assert "reload" in captured["command"]
    assert failed == [9303]


@pytest.mark.asyncio
async def test_reload_timeout_covers_full_fleet():
    captured = {}

    async def fake_run(self, command, timeout=60.0):
        captured["timeout"] = timeout
        from src.gateway.training.studio_ops import SSHResult
        return SSHResult(0, "", "")

    from src.gateway.training.studio_ops import UNLOAD_PORTS
    with patch.object(StudioOps, "run", new=fake_run):
        await StudioOps().reload_workers(list(UNLOAD_PORTS))
    # 9 workers x 300s sequential worst case = 2700s; timeout must exceed it
    assert captured["timeout"] > 2700.0
