# Served-model derivation — design (Fil A: #115 + #116)

**Date:** 2026-05-29
**Issues:** #115 (restore served-model observability post-omlx), #116 (prune dead aliases from MODEL_FORCE_MAP)
**Repo:** `ailiance/ailiance` (gateway)

## Problem

Post omlx+qwen36 consolidation (`482f877`, `f375c12`, `79f4136`), the routing/observability layer has two gaps:

- **#115** — `resolve_effective_alias` collapses all auto-routed traffic to the bare `"ailiance"` label, so the actually-serving specialist is no longer visible in headers/audit. The information exists (`DOMAIN_TO_QWEN36[domain]` / `DOMAIN_TO_OMLX_MODEL[domain]`) but is not surfaced.
- **#116** — `MODEL_FORCE_MAP` statically enumerates ~60 aliases across ~14 ports, most of which are retired per-port workers. The runtime liveness filter (#12) hides them from `/v1/models` when down, but the static map still carries the cruft and the entries remain force-routable to dead ports.

## Goal

Resolve both via a single shared derivation point, with **no routing refactor** and **no dependency on operator-provided live-set data**. (The recurring "red main from routing changes" class is already addressed structurally by branch protection requiring the `test` CI check — so a larger source-of-truth refactor is not justified here.)

## Approach (Option 3 — intermediate)

A shared pure helper `served_model_for()` that derives the serving specialist from the existing maps, reused by observability; plus a boot-time self-maintaining filter of `MODEL_FORCE_MAP` against the resolved worker set.

### Component 1 — `served_model_for()` (shared derivation)

Pure function in `src/gateway/alias_inventory.py` (the observability module — it may import `domain_map`; `domain_map` must not import it, avoiding a cycle).

```
served_model_for(*, alias: str, domain: str | None, worker_port: int) -> str
```

Resolution order:
- `domain` set AND `worker_port` in the qwen36 ports (`QWEN36_PORT` 9360 / `QWEN36_PORT_B` 9361) → `DOMAIN_TO_QWEN36[domain]` (e.g. `qwen36-emc-dsp-power`).
- `domain` set AND `worker_port == OMLX_PORT` (8500) → `DOMAIN_TO_OMLX_MODEL[domain]` (e.g. `EuroLLM-22B-Instruct-2512`).
- otherwise (explicit alias, no classifier domain) → `_REGISTRY[alias].base_model` if present, else `alias`.
- any miss / unexpected input → fallback to `alias` (or `"unknown"` if alias falsy).

Invariant: **never raises.** Observability must not break a response. All lookups are `.get(...)` with fallbacks.

### Component 2 — #115: expose the served specialist (header + audit)

In `src/gateway/server.py`, in BOTH the non-streaming and streaming response paths, after `domain` and `worker_port` are resolved:

- compute `served = served_model_for(alias=req.model, domain=domain, worker_port=worker_port)`.
- set response header **`X-Ailiance-Served-Model: <served>`**.
- pass `served` into the `track_chat(...)` audit stamp (new field, e.g. `served_model=served`).

Explicitly unchanged: `resolve_effective_alias` return value and the response body `model` field (OpenAI clients that echo the requested model are not affected). This is observability-only.

### Component 3 — #116: boot-time self-maintaining force-map filter

At app construction in `src/gateway/server.py`:

```
effective_force_map = {a: p for a, p in MODEL_FORCE_MAP.items() if p in WORKER_URLS}
```

where `WORKER_URLS` is the resolved worker table (`AILIANCE_WORKERS_JSON` merged over `_DEFAULT_WORKER_URLS`). The effective map is used for `/v1/models` advertising AND explicit force-routing.

- Self-maintaining: ports never configured (9322/9335/9340/…) drop automatically; no manual prune, no operator data required from the author.
- Complementary to the #12 runtime liveness filter: boot-filter removes *never-configured* ports; liveness-filter removes *configured-but-currently-down* ports. Both layers apply to `/v1/models`.
- Cold-start safety: `_DEFAULT_WORKER_URLS` always contains the core ports, so the effective map is never empty even if `AILIANCE_WORKERS_JSON` is absent/misconfigured.
- Effect: `model=ailiance-<dead>` resolves to "unknown model" (clean error) instead of routing to a dead port.

## File structure

- Modify: `src/gateway/alias_inventory.py` — add `served_model_for()` (+ import the domain maps from `domain_map`).
- Modify: `src/gateway/server.py` — build `effective_force_map` at app construction; wire `X-Ailiance-Served-Model` header + `track_chat` served_model field in both response paths; route explicit aliases / `/v1/models` through `effective_force_map`.
- Tests: `tests/test_gateway_alias_inventory.py` (served_model_for unit tests), `tests/test_gateway.py` (header + boot-filter integration).

## Testing

- `served_model_for`: qwen36 domain → adapter; omlx domain → model; explicit alias → `base_model`; unknown alias/domain → fallback to alias; never raises on bad input.
- #115: an auto-routed request response carries `X-Ailiance-Served-Model` with the correct specialist; the audit stamp (track_chat) includes the served model. Body `model` unchanged.
- #116: a `MODEL_FORCE_MAP` entry whose port ∉ `WORKER_URLS` is absent from the effective map and `/v1/models`; an entry whose port ∈ `WORKER_URLS` is kept; with `AILIANCE_WORKERS_JSON` absent, the core ports survive (cold-start not nuked).

## Error handling

- `served_model_for` never raises; returns a safe fallback string.
- Boot filter falls back to `_DEFAULT_WORKER_URLS`-backed ports if `AILIANCE_WORKERS_JSON` is missing; logs once if it drops a large fraction of aliases (optional, for operator visibility).

## Out of scope

- No refactor of `domain_map` routing or `resolve_effective_alias`.
- No rewrite/static prune of `MODEL_FORCE_MAP` (the boot filter supersedes the need).
- No change to the response body `model` field.
- #118 (scripts path policy) — separate thread (Fil B).
- #119 deploy-key — operator action (Node 24 part already done).

## Workflow note

`main` is branch-protected (requires the `test` CI check). This work lands via a PR from `feat/served-model-derivation`, validated green before merge.
