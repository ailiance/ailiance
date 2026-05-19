"""All SSH operations against the Mac Studio training host.

The only module in the training package that performs I/O against Studio.
electron-server reaches Studio directly (the documented bastion path).
"""
from __future__ import annotations

import asyncio
import os
import re
import shlex
from dataclasses import dataclass

STUDIO_SSH = os.environ.get("AILIANCE_STUDIO_SSH", "clems@100.116.92.12")
REMOTE_SCRIPT_DIR = "/Users/clems/.ailiance-training"
REMOTE_LOG_DIR = "/Users/clems/KIKI-Mac_tunner/logs"

# Worker ports unloaded during training (frees ~280 GB).
UNLOAD_PORTS: tuple[int, ...] = (9301, 9303, 9324, 8500, 9323, 9327, 9325, 9328, 9329)
# Kept resident so the gateway stays partially routable.
MINIMAL_ROUTABLE_PORTS: frozenset[int] = frozenset({9326, 8501})

_UNUSED_RE = re.compile(r"([0-9.]+)G unused")


@dataclass
class SSHResult:
    returncode: int
    stdout: str
    stderr: str


class StudioOps:
    def __init__(self, ssh_target: str = STUDIO_SSH) -> None:
        self._target = ssh_target

    async def run(self, command: str, timeout: float = 60.0) -> SSHResult:
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            self._target, command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        return SSHResult(proc.returncode or 0,
                         out.decode(errors="replace"),
                         err.decode(errors="replace"))

    async def free_memory_gb(self) -> float:
        res = await self.run("top -l1 -s0 | awk '/PhysMem/{print}'")
        m = _UNUSED_RE.search(res.stdout)
        return float(m.group(1)) if m else 0.0

    async def venv_ok(self) -> bool:
        res = await self.run(
            "/Users/clems/KIKI-Mac_tunner/.venv/bin/python "
            "-c 'import mlx.core' && echo VENV_OK"
        )
        return "VENV_OK" in res.stdout

    async def deploy_scripts(self, local_dir: str) -> None:
        """scp the two Studio scripts into REMOTE_SCRIPT_DIR."""
        await self.run(f"mkdir -p {shlex.quote(REMOTE_SCRIPT_DIR)}")
        proc = await asyncio.create_subprocess_exec(
            "scp", f"{local_dir}/medium35_workers.sh",
            f"{local_dir}/medium35_train_domain.sh",
            f"{self._target}:{REMOTE_SCRIPT_DIR}/",
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"scp to Studio failed (exit {proc.returncode})")

    async def unload_workers(self) -> list[int]:
        await self.run(f"bash {REMOTE_SCRIPT_DIR}/medium35_workers.sh unload",
                       timeout=120.0)
        return list(UNLOAD_PORTS)

    async def reload_workers(self, ports: list[int]) -> list[int]:
        """Reload workers; return the ports that failed the HTTP healthcheck."""
        res = await self.run(
            f"bash {REMOTE_SCRIPT_DIR}/medium35_workers.sh reload", timeout=300.0
        )
        failed = []
        for line in res.stdout.splitlines():
            if line.startswith("RELOAD_FAILED "):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    failed.append(int(parts[1]))
        return failed

    async def spawn_domain(self, domain: str) -> int:
        """Launch the detached single-domain trainer; return its PID."""
        res = await self.run(
            f"bash {REMOTE_SCRIPT_DIR}/medium35_train_domain.sh spawn "
            f"{shlex.quote(domain)}"
        )
        return int(res.stdout.strip())

    async def pid_alive(self, pid: int) -> bool:
        res = await self.run(f"kill -0 {pid} 2>/dev/null && echo ALIVE")
        return res.returncode == 0 or "ALIVE" in res.stdout

    async def read_domain_log(self, domain: str) -> str:
        res = await self.run(f"cat {REMOTE_LOG_DIR}/medium35-{shlex.quote(domain)}.log "
                              "2>/dev/null || true")
        return res.stdout
