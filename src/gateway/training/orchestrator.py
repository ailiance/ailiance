"""medium35 campaign state machine — drives StudioOps over SSH."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from src.gateway.training import domains as D
from src.gateway.training.progress import classify_val_loss, parse_domain_log
from src.gateway.training.state import CampaignState, load_state, save_state

log = logging.getLogger(__name__)

POST_UNLOAD_MIN_FREE_GB = 320.0
POLL_INTERVAL_S = 30.0
MEM_SETTLE_POLL_S = 15.0       # interval between free-memory samples after unload
MEM_SETTLE_MAX_SAMPLES = 8     # give up to ~2 min for buffers to be reclaimed
MAX_DOMAIN_SECONDS = 2 * D.HOURS_PER_DOMAIN * 3600  # stuck-PID guard
SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts" / "studio")


class StudioOpsProtocol(Protocol):
    """The StudioOps surface the orchestrator depends on."""

    async def venv_ok(self) -> bool: ...
    async def free_memory_gb(self) -> float: ...
    async def deploy_scripts(self, local_dir: str) -> None: ...
    async def unload_workers(self) -> list[int]: ...
    async def reload_workers(self, ports: list[int]) -> list[int]: ...
    async def spawn_domain(self, domain: str) -> int: ...
    async def pid_alive(self, pid: int) -> bool: ...
    async def read_domain_log(self, domain: str) -> str: ...


AVAILABLE_DURING_TRAINING: tuple[str, ...] = (
    "ailiance-mistral-small", "ailiance-qwen",
)


def build_training_503(state: CampaignState, alias: str) -> dict:
    """OpenAI-compatible 503 body with the live training step.

    The human-readable `error.message` includes the ETA and the fallback
    aliases so that a CLI client surfacing only the message still gives the
    user actionable information; the structured `training` block and
    `available_models` field carry the same data for machine consumers.
    """
    total = len(state.domains)
    remaining = max(0, total - state.domain_index)
    eta_hours = remaining * D.HOURS_PER_DOMAIN
    eta = f"~{eta_hours} h" if eta_hours < 48 else f"~{eta_hours // 24} jours"
    fallbacks = ", ".join(AVAILABLE_DURING_TRAINING)
    return {
        "error": {
            "message": (
                f"Modèle '{alias}' temporairement indisponible. "
                f"Campagne d'entraînement {state.campaign} en cours, "
                f"{eta} restants. Modèles disponibles : {fallbacks}."
            ),
            "type": "model_unavailable_training",
            "code": "training_in_progress",
        },
        "training": {
            "campaign": state.campaign,
            "current_domain": state.current_domain,
            "domain_index": min(state.domain_index + 1, total),
            "domains_total": total,
            "phase": state.phase,
            "phase_total": 3,
            "iter": state.iter,
            "iter_total": state.iter_total,
            "eta_campaign": eta,
            "last_verdicts": state.verdicts,
        },
        "available_models": list(AVAILABLE_DURING_TRAINING),
    }


class TrainingOrchestrator:
    def __init__(self, ops: StudioOpsProtocol, state_path: Path) -> None:
        self._ops = ops
        self._state_path = Path(state_path)
        self.state: CampaignState = load_state(self._state_path)
        self._task: asyncio.Task | None = None

    def _save(self) -> None:
        save_state(self._state_path, self.state)

    def _set(self, status: str, **fields) -> None:
        self.state.status = status
        for k, v in fields.items():
            setattr(self.state, k, v)
        self._save()

    def _progress(self, phase: int, iter_: int) -> None:
        """Update training progress WITHOUT touching status — so a pending
        abort (tracked separately) is never clobbered by a poll tick."""
        self.state.phase = phase
        self.state.iter = iter_
        self.state.iter_total = D.PHASE_ITERS.get(phase, 0)
        self._save()

    async def start(self, domains: list[str] | None = None) -> None:
        if self.state.is_active:
            raise RuntimeError(f"campaign already active: {self.state.status}")
        self.state = CampaignState(
            status="PREFLIGHT",
            domains=list(domains or D.CAMPAIGN_DOMAINS),
            started_at=datetime.now(UTC).isoformat(),
        )
        self._save()
        self._task = asyncio.create_task(self._run_campaign())
        self._task.add_done_callback(self._on_task_done)

    @staticmethod
    def _on_task_done(task: asyncio.Task) -> None:
        """Surface an exception that escaped _run_campaign — never lose it."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("training campaign task crashed", exc_info=exc)

    async def abort(self) -> None:
        """Request a graceful stop. The domain currently training finishes,
        then the campaign loop stops and workers are reloaded. The detached
        Studio batch is never killed."""
        self.state.abort_requested = True
        self._save()

    async def reattach(self) -> bool:
        """On gateway boot: resume a non-terminal campaign. Returns True if a
        campaign was resumed. Skips preflight/unload when the workers are
        already unloaded; the in-progress domain re-attaches to its live batch
        pid, or restarts (resume-safe via phaseN_done sentinels) if it died."""
        if not self.state.is_active:
            return False
        if self._task is not None and not self._task.done():
            return False  # a campaign task is already running
        self._task = asyncio.create_task(self._run_campaign(resume=True))
        self._task.add_done_callback(self._on_task_done)
        return True

    def status(self) -> dict:
        return {
            "status": self.state.status,
            "campaign": self.state.campaign,
            "current_domain": self.state.current_domain,
            "domain_index": self.state.domain_index,
            "domains_total": len(self.state.domains),
            "phase": self.state.phase,
            "iter": self.state.iter,
            "abort_requested": self.state.abort_requested,
            "verdicts": self.state.verdicts,
            "reload_failed": self.state.reload_failed,
            "error": self.state.error,
        }

    async def _preflight(self) -> bool:
        if not await self._ops.venv_ok():
            self._set("FAILED", error="Studio venv invalide (import mlx.core)")
            return False
        await self._ops.deploy_scripts(SCRIPTS_DIR)
        return True

    async def _unload(self) -> None:
        self._set("UNLOADING")
        self.state.unloaded_ports = await self._ops.unload_workers()
        self._save()
        free = await self._settled_free_memory()
        if free < POST_UNLOAD_MIN_FREE_GB:
            raise RuntimeError(
                f"après déchargement, mémoire libre {free:.0f} GB "
                f"< {POST_UNLOAD_MIN_FREE_GB:.0f} GB requis"
            )

    async def _settled_free_memory(self) -> float:
        """Poll free memory until it stops rising. A killed worker's macOS
        unified-memory buffers take tens of seconds to be reclaimed, so a
        reading taken immediately after unload undercounts free memory."""
        free = await self._ops.free_memory_gb()
        for _ in range(MEM_SETTLE_MAX_SAMPLES):
            await asyncio.sleep(MEM_SETTLE_POLL_S)
            nxt = await self._ops.free_memory_gb()
            if nxt <= free + 2.0:  # stabilised — no meaningful further gain
                return nxt
            free = nxt
        return free

    async def _train_domain(self, domain: str, resume_pid: int | None = None) -> None:
        if resume_pid is not None and await self._ops.pid_alive(resume_pid):
            pid = resume_pid  # batch survived the gateway restart — re-attach
            self._set("TRAINING")  # correct a stale resumed status
        else:
            self._set("TRAINING", phase=0, iter=0)
            pid = await self._ops.spawn_domain(domain)
            self.state.batch_pid = pid
            self._save()
        deadline = time.monotonic() + MAX_DOMAIN_SECONDS
        while await self._ops.pid_alive(pid):
            if time.monotonic() > deadline:
                log.warning("domain %s exceeded max duration, abandoning poll",
                            domain)
                break
            text = await self._ops.read_domain_log(domain)
            prog = parse_domain_log(text, domain)
            self._progress(prog.phase, prog.iter)
            await asyncio.sleep(POLL_INTERVAL_S)

    async def _gate_domain(self, domain: str) -> None:
        self._set("GATING")
        text = await self._ops.read_domain_log(domain)
        prog = parse_domain_log(text, domain)
        if prog.complete and prog.final_val_loss is not None:
            verdict = classify_val_loss(prog.final_val_loss)
        else:
            verdict = "FAILED_OOM" if "FAILED_OOM" in text else "INCOMPLETE"
        self.state.verdicts[domain] = verdict
        self._save()

    async def _reload(self) -> None:
        if not self.state.unloaded_ports:
            return
        failed = await self._ops.reload_workers(self.state.unloaded_ports)
        self.state.reload_failed = failed
        self.state.unloaded_ports = []
        self._save()

    async def _domain_loop(self, resume_status: str) -> None:
        """Run train+gate for each remaining domain. On a resume:
        - resume_status TRAINING: the first domain re-attaches to its batch pid;
        - resume_status GATING: the first domain's training already finished,
          so it is gated only (not re-trained)."""
        first = resume_status in ("TRAINING", "GATING")
        skip_train_first = resume_status == "GATING"
        while self.state.domain_index < len(self.state.domains):
            if self.state.abort_requested:
                break
            domain = self.state.domains[self.state.domain_index]
            if not (first and skip_train_first):
                await self._train_domain(
                    domain,
                    resume_pid=self.state.batch_pid if first else None,
                )
            first = False
            await self._gate_domain(domain)
            self.state.domain_index += 1
            self._save()

    async def _run_campaign(self, resume: bool = False) -> None:
        try:
            # Crash-time status decides where to resume. PREFLIGHT/UNLOADING:
            # workers not yet (fully) unloaded -> run preflight + unload.
            # TRAINING/GATING: workers already unloaded -> straight to the loop.
            # RELOADING: all domains done -> straight to reload.
            status = self.state.status if resume else "IDLE"
            if status in ("IDLE", "PREFLIGHT", "UNLOADING"):
                if not await self._preflight():
                    return
                await self._unload()
            if status != "RELOADING":
                await self._domain_loop(status)
            self._set("RELOADING")
            await self._reload()
            self._set("ABORTED" if self.state.abort_requested else "DONE")
        except Exception as exc:  # noqa: BLE001
            log.exception("campaign crashed")
            self._set("FAILED", error=str(exc))
            try:
                await self._reload()
            except Exception:  # noqa: BLE001
                log.exception("reload during failure recovery also failed")
