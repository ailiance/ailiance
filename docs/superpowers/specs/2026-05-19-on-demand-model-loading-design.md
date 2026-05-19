# On-demand model loading — multi-host swap pool

**Date:** 2026-05-19
**Status:** design — approved for spec, implementation in a dedicated session

## Problem

The gateway advertises 46 model aliases. The Mac Studio (512 GB) currently
keeps ~9 MLX servers resident (~417 GB used, ~94 GB free). The long-tail
models — `apertus-*`, `mixtral`, `llama`, `qwen-235b`/`flagship`,
`devstral-*` — cannot all stay resident, so today they silently fall back
to the Tower Gemma 3 4B worker. End-to-end audit (2026-05-19): 28 of 46
aliases did not serve their named model.

**Goal:** make all 46 aliases serve their real model, without every model
being resident at once.

## Key enabler

`mlx_lm.server` ≥ 0.31 reloads the model when a request names a different
`model`: `ModelProvider` compares the requested `model_key` against the
loaded one and reloads on mismatch (server.py:393). One server process can
therefore swap models on its own — we orchestrate swap servers rather than
writing a model-loading engine.

A single `mlx_lm.server` holds exactly one model at a time; alternating
requests for two models cause a reload each time (thrashing). The design
mitigates this with a small pool plus model-affinity routing.

## Architecture — three tiers

### Pinned tier (Studio, always resident)
Genuinely hot models, one dedicated always-on `mlx_lm.server` each:
`ailiance-mistral`/`-mistral-medium` (Mistral-Medium-128B Q8, kept at Q8 —
no quality downgrade), `mascarade` :9340 (10 hardware LoRA experts),
`ailiance-coder-pro`, `ailiance-reasoning-r1`.

### Swap pool — Studio (distinct base models only)
`mlx_lm.server` swaps a *base model* per request — it cannot hot-swap a
LoRA adapter per request (`--adapter-path` is a startup flag; the OpenAI
request has no adapter field). So the swap pool serves **distinct base
models only**, confirmed on disk:
- **slot XL** (~120 GB budget): `qwen-235b`/`flagship` (Qwen3-235B),
  `mixtral`/`mixtral-8x22b` (Mixtral-8x22B), `llama` (Llama-3.3-70B).
- **slot M** (~45 GB budget): `qwen36` (Qwen3.6-35B), `devstral-base`
  (Devstral-Small-2-24B).

### LoRA-variant families — multi-LoRA servers (not the swap pool)
The 9 `apertus-*` aliases (Apertus-70B base + 9 domain LoRAs) and the 5
devstral aliases `python`/`cpp`/`rust-emb`/`html`/`ml-training` (Devstral
base + 5 LoRAs) are **LoRA variants**, not distinct base models. They need
the multi-LoRA-server pattern — one base in VRAM, N adapters hot-swapped
per request — exactly what `mascarade_multi_server` (:9340) already does
for the 10 mascarade hardware LoRAs. Each family gets its own multi-LoRA
server; this is a separate workstream from the swap pool.

### Swap server — macM1 (1 slot)
One `mlx_lm.server` swap instance on macM1 (32 GB) for medium models
≤ ~24 GB: `ministral`, `ministral-reasoning`, `gemma2`/`gemma4`,
`mistral-small`, `qwen2.5`-class.

## Host capacity (measured 2026-05-19)

- **Studio** — 512 GB. ~369 GB genuinely resident: ~208 GB of standalone
  `mlx_lm.server` workers, ~43 GB EuroLLM, and **~77 GB for
  `mascarade_multi_server`** (`mascarade-mlx`, runs in the `eu-kiki` venv —
  a legacy name; it is NOT a voice pipeline). That process serves `:9340`
  (the 10 mascarade hardware LoRA experts) and is bound to the wider fleet
  port layout — gateway-critical, must stay. ~92 GB free + ~42 GB
  reclaimable file cache → ~130 GB headroom before any change.
- **macM1** — 32 GB. One swap server, models ≤ ~24 GB.
- **kxkm-ai** — 62 GB RAM (~47 GB available) + RTX 4090 24 GB VRAM
  (~5 GB free). **NVIDIA/llama.cpp stack — MLX does not run here.** It is
  *not* a swap-pool host; it stays as the fixed GGUF pair `ailiance-qwen`
  (Qwen3-Next-80B) + `ailiance-granite` (Granite-30B). On-demand GGUF
  loading on kxkm-ai is a possible separate follow-up, out of scope here.

## Memory budget — without downgrading Mistral-Medium

Mistral-Medium-128B stays at Q8 (130 GB). The swap-pool budget is funded by
**demoting low-traffic pinned models to the swap tier** — they stop being
resident and load on demand instead:

| Demoted from pinned | Approx RSS freed |
|---|---|
| `qwen36` (Qwen3.6-35B) | ~19 GB |
| `eurollm` (EuroLLM-22B) | ~45 GB |
| `pixtral` (Pixtral-12B) | ~7 GB |
| `qwen2.5-7B` | ~4 GB |
| `qwen3-4b` base | ~8 GB |
| `mistral-small` (→ macM1-swap) | ~13 GB |

Freed ≈ 96 GB + ~94 GB already free ≈ **~190 GB swap budget on Studio** —
enough for slot XL (235B ≈ 120 GB) plus slot M (~45 GB). Footprints are
estimates; the implementation must measure real RSS and store them in the
`MODEL_FOOTPRINT` table.

Constraint: with Q8 kept, the pool cannot hold two XL models at once. The
memory-aware router (below) must never route a model whose footprint
exceeds the chosen slot's free budget.

## Components

### Swap servers (ops)
`mlx_lm.server` instances launched via launchd plists **from a Terminal on
the host** — launchctl over SSH reports success without actually starting
the agent (documented gui-domain limitation). New plists:
`cc.ailiance.swap-xl.plist`, `cc.ailiance.swap-m.plist` (Studio),
`cc.ailiance.swap-macm1.plist` (macM1). Each points at the host's MLX model
library directory.

### ModelManager (gateway — new module `src/gateway/model_manager.py`)
Holds three tables:
- `ALIAS_TIER`: alias → `pinned | studio-swap | macm1-swap`.
- `MODEL_FOOTPRINT`: alias → estimated GB.
- `SWAP_SLOTS`: slot → `{url, budget_gb, current_model, last_used}`.

`resolve(alias) -> worker_url`:
1. `pinned` → existing `MODEL_FORCE_MAP` path, unchanged.
2. swap-tier → pick a slot:
   - the slot already holding `alias` → return it (warm hit);
   - else the LRU slot whose `budget_gb ≥ MODEL_FOOTPRINT[alias]` → the
     request itself triggers the reload (mlx_lm swaps on `model` mismatch);
   - update `current_model` + `last_used`.
3. No compatible slot free → reuse the LRU slot of the right size class
   (its model is evicted by the swap).

Concurrency: reuse the gateway's existing per-`worker_url` `asyncio.Lock`
(PR #68 FIFO) so concurrent requests to one slot serialise — no double
load. A swap in progress makes same-slot requests wait on the lock.

### Cold-start signalling
A swap reload of a 70B+ model takes minutes. When the manager routes to a
slot that must reload, the gateway emits `event: loading` on the SSE stream
before the first token; the cockpit playground shows "chargement du
modèle…". Non-streaming callers get a longer timeout budget for swap-tier
aliases.

## Control plane — model selection & router retraining

An operator can choose, from `admin.ailiance.fr`, which models are
available; the router is retrained on the resulting domain set.

### Data model
- `configs/domain_models.yaml` — explicit map: each of the 47 router
  domains → the model alias(es) that serve it (today implicit in routing).
- `configs/enabled_models.yaml` — the operator-controlled set of enabled
  aliases (default: all enabled).
- Derived `active_domains` = domains with ≥ 1 enabled serving model.
  Disabling the last model of a domain makes it *orphan* → dropped from the
  router's label set.

### Router artifact cache (avoid redundant retraining)
Each router build is keyed by a short hash of its sorted `active_domains`
set. A registry `output/router-registry.yaml` maps
`domain_set_hash → {artifact_path, domains, trained_at}`. "Retrain" first
hashes the current `active_domains`: a registry **hit** hot-reloads the
existing artifact instantly (no training); a **miss** trains, stores the
artifact under the hash, registers it, then hot-reloads. Selecting a
previously-used configuration is therefore instant.

### Admin UI (`cockpit-admin`)
A "Modèles & Routeur" page: every alias with an on/off toggle, the domains
each serves, and a warning when disabling one would orphan a domain. A
**manual** "Réentraîner le routeur" button (enabled only when the selection
differs from the live router's build); it shows whether the target config
is cached (instant) or needs training, and the resulting domain count +
job status.

### Gateway endpoints (behind the `admin.ailiance.fr` Keycloak/tailnet gate)
- `GET /admin/models` — aliases, enabled state, domains served.
- `PUT /admin/models/enabled` — persist `enabled_models.yaml`.
- `POST /admin/router/retrain` — manual trigger; hashes `active_domains`,
  serves from cache or starts a background training job; returns a job id.
- `GET /admin/router/retrain/{job_id}` — job status.

### Retraining cost
Only the MLP head (256→N) is retrained — the MiniLM encoder is frozen and
its corpus embeddings are cached, so a train run is minutes, not hours. The
job is a short FastAPI background task, not heavy infrastructure.
`num_domains` is already a `gateway.yaml` parameter, so a variable-size
head is supported. On success the gateway hot-reloads `app.state.router`.

## Data flow

```
client → gateway /v1/chat/completions {model: X}
  → ModelManager.resolve(X)
      pinned    → MODEL_FORCE_MAP[X]  (resident, fast)
      swap      → slot holding X      (warm, fast)
                  or LRU compatible slot (cold: emit event:loading, reload)
  → forward request to chosen worker_url under its asyncio.Lock
```

## Error handling

- Swap server unreachable → existing Gemma fallback + `WARN` log.
- Alias's model absent from the host library → `404` with a clear message
  (no silent Gemma fallback for an explicitly named swap model).
- Footprint exceeds every slot budget → `503` "model too large for the
  current fleet" rather than an OOM crash.
- OOM during a reload → the manager never routes above a slot budget; if it
  still happens, the slot's server is restarted (watchdog) and the request
  retried once.

## Testing

- Unit (`tests/test_model_manager.py`): slot selection — warm hit, LRU
  pick, memory-aware rejection, size-class matching.
- Integration: mock swap servers; assert a model change triggers a reload
  route and an `event: loading` frame; assert concurrent same-slot requests
  serialise.
- Regression: `test_gateway_alias_inventory.py` — every alias resolves to a
  tier; no alias is orphaned.

## Phasing

- **P1 — "all models available"**: model library + one Studio swap server +
  `ALIAS_TIER` + simple single-slot routing. Long-tail aliases stop falling
  back to Gemma. Thrashing accepted.
- **P2 — pool + optimisation**: 2-3 slots, memory-aware LRU routing, macM1
  swap server, `event: loading` signalling, idle eviction.

## Out of scope

- Auto-downloading missing model weights.
- On-the-fly quantization.
- Replacing the auto-router's domain→port table (separate concern).
- The dead `apertus-*` aliases: the Apertus base model was deleted
  (2026-05-12). They are kept routable to the M slot only if the
  `Apertus-70B-...-4bit-MLX` weights on disk are wired to a swap server;
  otherwise they should be removed from `MODEL_FORCE_MAP` and `/v1/models`
  in a separate cleanup. Flagged, not decided here.
