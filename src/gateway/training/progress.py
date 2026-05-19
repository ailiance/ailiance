"""Pure parsing of Studio training logs and the val-loss quality gate."""
from __future__ import annotations

import re
from dataclasses import dataclass

_PHASE_RE = re.compile(r"### PHASE (\d+)/(\d+)")
_ITER_RE = re.compile(r"Iter (\d+):")
_DONE_RE = re.compile(r"### DOMAIN COMPLETE (\S+) final_val_loss=([0-9.]+)")

GATE_OVERFIT_BELOW = 0.05
GATE_UNDERTRAIN_ABOVE = 1.5


@dataclass(frozen=True)
class DomainProgress:
    domain: str
    phase: int = 0          # 1..3, 0 = not started
    phase_total: int = 3
    iter: int = 0
    complete: bool = False
    final_val_loss: float | None = None


def parse_domain_log(text: str, domain: str) -> DomainProgress:
    """Derive the latest progress from a domain's training log."""
    phase = 0
    phase_total = 3
    for m in _PHASE_RE.finditer(text):
        phase, phase_total = int(m.group(1)), int(m.group(2))
    iters = _ITER_RE.findall(text)
    cur_iter = int(iters[-1]) if iters else 0
    done = _DONE_RE.search(text)
    if done:
        return DomainProgress(
            domain=domain, phase=phase_total, phase_total=phase_total,
            iter=cur_iter, complete=True, final_val_loss=float(done.group(2)),
        )
    return DomainProgress(
        domain=domain, phase=phase, phase_total=phase_total, iter=cur_iter,
    )


def classify_val_loss(val_loss: float) -> str:
    """Non-blocking quality verdict for a finished domain."""
    if val_loss < GATE_OVERFIT_BELOW:
        return "SUSPECT_OVERFIT"
    if val_loss > GATE_UNDERTRAIN_ABOVE:
        return "SUSPECT_UNDERTRAIN"
    return "OK"
