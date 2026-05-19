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
