#!/usr/bin/env python3
"""Run eval_n3 5-axis scorer over a directory of .kicad_sch files x 5 seeds.

Outputs:
  --out          : flat list of per-(file, seed) records (audit detail)
  --out-aggregate: bench_comparison-compatible cells (model_key, domain,
                   pass_rate, n_samples) where pass_rate = mean composite
                   across all (file, seed) pairs.

Audit:
  --audit-dir <dir> writes NDJSON via AuditLogger, sha256-signed at end.

Determinism:
  Seeds locked to [42, 137, 1024, 8675309, 31415] per spec.
  kicad-cli itself is deterministic; seeds drive any future LLM regen
  wrapper -- they are stored on each record for traceability.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Allow running both as a script (python scripts/run_eval_n3.py) and as a
# module under tests. Add scripts/ to sys.path so kicad_sch.eval_n3 resolves.
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kicad_sch.eval_n3 import eval_all  # noqa: E402

SEEDS = [42, 137, 1024, 8675309, 31415]


class _NoopAudit:
    """Used when AuditLogger import fails (e.g. Foundation not yet shipped)."""

    def log_event(self, event_type, payload):
        return None

    def sha256_sign(self):
        return ""


class _AuditAdapter:
    """Adapts Plan F AuditLogger (log(type, **fields), sha256_manifest module
    fn) to the duck-typed (log_event, sha256_sign) interface used by eval_n3.
    """

    def __init__(self, logger, log_path):
        self._logger = logger
        self._log_path = log_path

    def log_event(self, event_type, payload):
        # Foundation AuditLogger expects **fields, so flatten the payload.
        try:
            self._logger.log(event_type, **payload)
        except TypeError:
            # Fallback: pass payload as a single "data" field.
            self._logger.log(event_type, data=payload)

    def sha256_sign(self):
        try:
            from kicad_sch.audit_log import sha256_manifest  # type: ignore
            return sha256_manifest(self._log_path)
        except Exception:
            return ""


def _get_audit(audit_dir: Path | None):
    if audit_dir is None:
        return _NoopAudit()
    try:
        from kicad_sch.audit_log import AuditLogger  # type: ignore
    except ImportError:
        print("WARN: kicad_sch.audit_log unavailable; using no-op audit",
              file=sys.stderr)
        return _NoopAudit()
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_path = audit_dir / f"eval_n3_{time.strftime('%Y%m%d_%H%M')}.ndjson"
    return _AuditAdapter(AuditLogger(log_path), log_path)


class _MockEvalAll:
    """Replaces eval_all under --mock-cli to keep tests hermetic."""

    @staticmethod
    def __call__(sch_path, ref, cli, audit):
        scores = {
            "parse_ok": 1, "erc_clean": 1, "sch_render": 1,
            "drc_clean": 0, "sem_equiv": 1.0,
        }
        from kicad_sch.eval_n3 import composite as _c
        scores["composite"] = _c(scores)
        audit.log_event("eval_n3.axis.mock",
                        {"sch": str(sch_path), "scores": scores})
        return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sch-dir", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-aggregate", type=Path, default=None)
    ap.add_argument("--audit-dir", type=Path, default=None)
    ap.add_argument("--cli-path", type=Path, default=Path("kicad-cli"))
    ap.add_argument("--mock-cli", action="store_true",
                    help="Bypass kicad-cli (testing only).")
    args = ap.parse_args()

    audit = _get_audit(args.audit_dir)
    runner = _MockEvalAll() if args.mock_cli else eval_all

    records: list[dict] = []
    sch_files = sorted(args.sch_dir.glob("*.kicad_sch"))
    if not sch_files:
        sys.exit(f"No .kicad_sch files under {args.sch_dir}")

    for sch in sch_files:
        ref_candidate = args.ref_dir / sch.name
        ref = ref_candidate if ref_candidate.exists() else None
        for seed in SEEDS:
            scores = runner(sch, ref, args.cli_path, audit)
            rec = {
                "model_key": args.model_key,
                "domain": args.domain,
                "sch": sch.name,
                "seed": seed,
                **{k: scores[k] for k in scores},
            }
            records.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))
    print(f"Wrote {args.out} ({len(records)} records)")

    if args.out_aggregate is not None:
        # bench_comparison-compatible aggregate: one row per (model, domain).
        composites = [r["composite"] for r in records]
        cell = {
            "model_key": args.model_key,
            "domain": args.domain,
            "pass_rate": round(statistics.mean(composites), 4) if composites else 0.0,
            "n_samples": len(records),
        }
        # Per-axis aggregates (consumed by bench_comparison --metric-axes).
        for axis in ["parse_ok", "erc_clean", "sch_render",
                     "drc_clean", "sem_equiv"]:
            vals = [r[axis] for r in records]
            cell[f"axis_{axis}"] = (round(statistics.mean(vals), 4)
                                    if vals else 0.0)
        args.out_aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.out_aggregate.write_text(json.dumps([cell], indent=2))
        print(f"Wrote {args.out_aggregate} (1 aggregate cell)")

    sig = audit.sha256_sign()
    if sig:
        print(f"Audit signed: sha256={sig[:16]}...")


if __name__ == "__main__":
    main()
