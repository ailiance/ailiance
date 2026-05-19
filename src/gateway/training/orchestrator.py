"""medium35 campaign state machine — drives StudioOps over SSH."""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.gateway.training import domains as D
from src.gateway.training.progress import classify_val_loss, parse_domain_log
from src.gateway.training.state import CampaignState, load_state, save_state

log = logging.getLogger(__name__)

PREFLIGHT_MIN_FREE_GB = 320.0
POLL_INTERVAL_S = 30.0
SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts" / "studio")


def build_training_503(state: CampaignState, alias: str) -> dict:
    """OpenAI-compatible 503 body with the live training step."""
    remaining = max(0, len(state.domains) - state.domain_index)
    return {
        "error": {
            "message": f"Modèle '{alias}' indisponible : campagne "
                       f"d'entraînement {state.campaign} en cours.",
            "type": "model_unavailable_training",
            "code": "training_in_progress",
        },
        "training": {
            "campaign": state.campaign,
            "current_domain": state.current_domain,
            "domain_index": state.domain_index + 1,
            "domains_total": len(state.domains),
            "phase": state.phase,
            "phase_total": 3,
            "iter": state.iter,
            "iter_total": state.iter_total,
            "eta_campaign": f"~{remaining * D.HOURS_PER_DOMAIN // 24} jours",
            "last_verdicts": state.verdicts,
        },
        "available_models": ["ailiance-mistral-small", "ailiance-qwen"],
    }


class TrainingOrchestrator:
    def __init__(self, ops, state_path: Path) -> None:
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

    async def abort(self) -> None:
        self._set("ABORTED")
        await self._reload()

    def status(self) -> dict:
        return {
            "status": self.state.status,
            "campaign": self.state.campaign,
            "current_domain": self.state.current_domain,
            "domain_index": self.state.domain_index,
            "domains_total": len(self.state.domains),
            "phase": self.state.phase,
            "iter": self.state.iter,
            "verdicts": self.state.verdicts,
            "reload_failed": self.state.reload_failed,
            "error": self.state.error,
        }

    async def _preflight(self) -> bool:
        if not await self._ops.venv_ok():
            self._set("FAILED", error="Studio venv invalide (import mlx.core)")
            return False
        free = await self._ops.free_memory_gb()
        if free < PREFLIGHT_MIN_FREE_GB:
            self._set("FAILED",
                      error=f"mémoire libre {free:.0f} GB < {PREFLIGHT_MIN_FREE_GB:.0f} GB")
            return False
        await self._ops.deploy_scripts(SCRIPTS_DIR)
        return True

    async def _unload(self) -> None:
        self._set("UNLOADING")
        self.state.unloaded_ports = await self._ops.unload_workers()
        self._save()

    async def _train_domain(self, domain: str) -> None:
        self._set("TRAINING", phase=0, iter=0)
        pid = await self._ops.spawn_domain(domain)
        self.state.batch_pid = pid
        self._save()
        while await self._ops.pid_alive(pid):
            text = await self._ops.read_domain_log(domain)
            prog = parse_domain_log(text, domain)
            self.state.phase, self.state.iter = prog.phase, prog.iter
            self._save()
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

    async def _run_campaign(self) -> None:
        try:
            if not await self._preflight():
                return
            await self._unload()
            while self.state.domain_index < len(self.state.domains):
                if self.state.status == "ABORTED":
                    return
                domain = self.state.domains[self.state.domain_index]
                await self._train_domain(domain)
                await self._gate_domain(domain)
                self.state.domain_index += 1
                self._save()
            self._set("RELOADING")
            await self._reload()
            self._set("DONE")
        except Exception as exc:  # noqa: BLE001
            log.exception("campaign crashed")
            self._set("FAILED", error=str(exc))
            await self._reload()
