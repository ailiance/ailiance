"""Persisted campaign state — survives a gateway restart."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

TERMINAL: frozenset[str] = frozenset({"IDLE", "DONE", "FAILED", "ABORTED"})


@dataclass
class CampaignState:
    status: str = "IDLE"
    campaign: str = "medium35"
    domains: list[str] = field(default_factory=list)
    domain_index: int = 0
    phase: int = 0
    iter: int = 0
    iter_total: int = 0
    batch_pid: int | None = None
    verdicts: dict[str, str] = field(default_factory=dict)
    unloaded_ports: list[int] = field(default_factory=list)
    reload_failed: list[int] = field(default_factory=list)
    started_at: str | None = None
    error: str | None = None

    @property
    def current_domain(self) -> str | None:
        if 0 <= self.domain_index < len(self.domains):
            return self.domains[self.domain_index]
        return None

    @property
    def is_active(self) -> bool:
        return self.status not in TERMINAL


def load_state(path: Path) -> CampaignState:
    """Load campaign state from `path`.

    Returns a default IDLE `CampaignState` if the file is absent or
    contains invalid JSON — a corrupt state file must never crash the
    gateway at startup.
    """
    if not path.exists():
        return CampaignState()
    try:
        return CampaignState(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError):
        return CampaignState()


def save_state(path: Path, state: CampaignState) -> None:
    """Atomic write: a crash mid-write never corrupts the state file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(path)
