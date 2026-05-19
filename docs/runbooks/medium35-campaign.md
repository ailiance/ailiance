# Runbook — medium35 training campaign

The medium35 campaign fine-tunes Mistral-Medium-128B LoRA adapters for 28
domains on the Mac Studio, orchestrated by the gateway. The run is ~20 days.
This runbook covers launching, monitoring, and recovery. The campaign itself
is an operational action triggered via the admin API — not code.

## Prerequisites

- `AILIANCE_ADMIN_TOKEN` must be set in the gateway's systemd environment.
  Without it the `/admin/training/*` endpoints return 503 (admin disabled).
- The launchd-label discovery in `scripts/studio/medium35_workers.sh` is
  dynamic — no manual map to fill.

## Pre-flight (manual sanity check)

1. venv: `ssh clems@100.116.92.12 "/Users/clems/KIKI-Mac_tunner/.venv/bin/python -c 'import mlx.core'"` — must succeed.
2. free RAM: `ssh clems@100.116.92.12 "top -l1 -s0 | awk '/PhysMem/'"` — note the unused GB. The orchestrator pre-flight refuses to start below 320 GB free.

## Quick win — kicad-sch-qwen36-D1 (before the campaign)

A separate single run on a different base model (Qwen3.6-35B). Its config is
already corrected (seq 2048, grad_accum 4, chunked data). It needs only port
9301 free, not the full unload. Run it directly on Studio:

    ssh clems@100.116.92.12 "cd /Users/clems/KIKI-Mac_tunner && source .venv/bin/activate && python -m mlx_lm.lora -c configs/eu-kiki-v3-qwen36-kicad-sch-D1.yaml"

## Start the campaign

    curl -X POST https://gateway.ailiance.fr/admin/training/start \
      -H "X-Admin-Token: $AILIANCE_ADMIN_TOKEN" \
      -H 'content-type: application/json' -d '{}'

An empty body trains all 28 domains hardware-first. To train a subset, pass
`{"domains": ["kicad-dsl", "kicad-pcb"]}`.

## Monitor

    curl -H "X-Admin-Token: $AILIANCE_ADMIN_TOKEN" https://gateway.ailiance.fr/admin/training/status

Reports the state machine status, current domain/phase/iter, per-domain
verdicts, and `reload_failed`.

## After completion

- Check `reload_failed` in the status. For each listed port, start the worker
  from a **local Terminal on Studio** — `launchctl` over SSH cannot start
  gui-domain agents.
- Review `verdicts` for any `SUSPECT_OVERFIT` / `SUSPECT_UNDERTRAIN` /
  `FAILED_OOM` / `INCOMPLETE` domains.

## Abort

    curl -X POST https://gateway.ailiance.fr/admin/training/abort \
      -H "X-Admin-Token: $AILIANCE_ADMIN_TOKEN"

Abort is graceful: the domain currently training finishes, then the campaign
stops and workers are reloaded. The detached Studio batch is never killed.

## Gateway restart mid-campaign

The orchestrator persists `campaign_state.json`. On startup the gateway
re-attaches automatically: it resumes from the crash-time status, re-attaching
to the in-progress domain's live batch (or restarting it — resume-safe via the
Studio-side `phaseN_done` sentinels). No manual action required.
A phase that was interrupted mid-way restarts from iteration 0 (resume is
phase-granular via the phaseN_done sentinels); its val-loss is therefore not
strictly comparable to the baseline runs the gate thresholds came from.
