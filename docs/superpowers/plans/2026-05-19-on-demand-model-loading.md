# On-demand Model Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all 46 gateway aliases serve their real model — long-tail
models load on demand instead of silently degrading to the Gemma fallback.

**Architecture:** `mlx_lm.server` ≥ 0.31 reloads the model when a request
names a different one. Phase 1 routes every long-tail alias to a single
swap `mlx_lm.server` via the existing `MODEL_FORCE_MAP` + `ALIAS_MODEL_REWRITES`
config — no new code. Phase 2 adds a `ModelManager` with a 2-3 slot pool and
memory-aware LRU routing.

**Tech Stack:** Python 3.14, FastAPI gateway (`src/gateway/server.py`),
`mlx_lm.server` workers on Apple Silicon, pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-19-on-demand-model-loading-design.md`

---

## Phase 1 — All models available (single swap server)

Phase 1 is config-only and ships working software on its own: long-tail
aliases stop falling back to Gemma. Thrashing (two cold models requested
alternately) is accepted and fixed in Phase 2.

### File structure (Phase 1)

- Modify `src/gateway/server.py` — `_DEFAULT_WORKER_URLS`, `MODEL_FORCE_MAP`,
  `ALIAS_MODEL_REWRITES`.
- Create `tests/test_swap_pool_routing.py` — routing assertions.
- New ops artefact on the Studio: `cc.ailiance.swap-1.plist` launchd agent.

The long-tail aliases routed to the swap server in Phase 1:
`apertus-real`, `apertus-electronics-hw`, `apertus-math-reasoning`,
`apertus-math-gsm8k`, `apertus-math`, `apertus-security-fenrir`,
`apertus-spice-sim`, `apertus-emc-dsp-power`, `apertus-embedded`,
`devstral-base`, `python`, `cpp`, `rust-emb`, `html`, `ml-training`,
`flagship`, `qwen-235b`, `mixtral`, `mixtral-8x22b`, `llama`, `qwen36`.
(All prefixed `ailiance-`.)

---

### Task 1: Inventory the Studio model library

The swap server resolves `body["model"]` as an on-disk path. We need the
exact path of each long-tail model so `ALIAS_MODEL_REWRITES` can point at it.

- [ ] **Step 1: List MLX model directories on the Studio**

Run (from the dev host):

```bash
ssh electron-server "ssh clems@100.116.92.12 'ls -1d /Users/clems/KIKI-Mac_tunner/models/*/ ~/.cache/huggingface/hub/models--*/ 2>/dev/null'"
```

Expected: a list of model directories. Record, for each long-tail alias,
the directory whose name matches the model (e.g. `ailiance-llama` →
`.../Llama-3.3-70B-Instruct-MLX-4bit`).

- [ ] **Step 2: Record the path table**

Write the alias→path mapping into this plan's Task 4 table before starting
Task 4. If a model is absent from disk (e.g. the deleted Apertus base),
mark the alias `MISSING` — it will be excluded from Phase 1 and flagged for
the separate Apertus cleanup.

- [ ] **Step 3: Commit the inventory note**

```bash
git add docs/superpowers/plans/2026-05-19-on-demand-model-loading.md
git commit -m "docs: record Studio model library inventory"
```

---

### Task 2: Launch the swap mlx_lm.server on the Studio

A single `mlx_lm.server` on port `9350`, started with a small default model
so it boots fast; it swaps to the requested model per request.

- [ ] **Step 1: Create the launchd plist**

Create `/Users/clems/Library/LaunchAgents/cc.ailiance.swap-1.plist` on the
Studio (via a Terminal on the Studio — launchctl over SSH does not reliably
start gui-domain agents):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>cc.ailiance.swap-1</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/clems/.venv-mistral/bin/python</string>
    <string>-m</string><string>mlx_lm.server</string>
    <string>--model</string>
    <string>/Users/clems/KIKI-Mac_tunner/models/Mistral-Small-3.1-24B-Instruct-MLX-4bit</string>
    <string>--host</string><string>0.0.0.0</string>
    <string>--port</string><string>9350</string>
    <string>--log-level</string><string>INFO</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/private/tmp/ailiance-swap-1.log</string>
  <key>StandardErrorPath</key><string>/private/tmp/ailiance-swap-1.log</string>
</dict>
</plist>
```

- [ ] **Step 2: Load the agent (in a Terminal on the Studio)**

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/cc.ailiance.swap-1.plist
```

- [ ] **Step 3: Verify it listens and the autossh tunnel reaches it**

The gateway host (electron-server) reaches `:9350` via an autossh tunnel —
add one mirroring the existing `:9340` tunnel:

```bash
# On electron-server, mirror mascarade-studio-tunnel.service for :9350
ssh electron-server "curl -sS -o /dev/null -w '%{http_code}\n' --max-time 8 http://localhost:9350/v1/models"
```

Expected: `200`.

---

### Task 3: Add the swap port to WORKER_URLS

**Files:** Modify `src/gateway/server.py` — `_DEFAULT_WORKER_URLS` (around
line 84, after the `9340` entry).

- [ ] **Step 1: Write the failing test**

Create `tests/test_swap_pool_routing.py`:

```python
"""Phase 1 swap-pool routing: long-tail aliases reach the swap server."""
from src.gateway.server import (
    ALIAS_MODEL_REWRITES,
    MODEL_FORCE_MAP,
    WORKER_URLS,
)

SWAP_PORT = 9350

SWAP_ALIASES = [
    "ailiance-devstral-base", "ailiance-python", "ailiance-cpp",
    "ailiance-rust-emb", "ailiance-html", "ailiance-ml-training",
    "ailiance-flagship", "ailiance-qwen-235b", "ailiance-mixtral",
    "ailiance-mixtral-8x22b", "ailiance-llama", "ailiance-qwen36",
]


def test_swap_port_is_registered():
    assert SWAP_PORT in WORKER_URLS
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_swap_pool_routing.py::test_swap_port_is_registered -v`
Expected: FAIL — `9350 not in WORKER_URLS`.

- [ ] **Step 3: Add the swap port to `_DEFAULT_WORKER_URLS`**

In `src/gateway/server.py`, after the `9340: "http://localhost:9340",`
entry, add:

```python
    # Studio swap server :9350 — one mlx_lm.server with no fixed model;
    # loads the requested long-tail model on demand (ModelProvider reloads
    # when the request names a different model). via autossh tunnel
    # electron-server:9350 → studio:9350.
    9350: "http://localhost:9350",
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_swap_pool_routing.py::test_swap_port_is_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gateway/server.py tests/test_swap_pool_routing.py
git commit -m "feat(gateway): register swap server port 9350"
```

---

### Task 4: Route long-tail aliases to the swap port

**Files:** Modify `src/gateway/server.py` — `MODEL_FORCE_MAP` (lines
381-453). Repoint the long-tail aliases from their dead Studio ports
(9316/9322/9328/9329/9330/9305) to `9350`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_swap_pool_routing.py`:

```python
def test_long_tail_aliases_route_to_swap_port():
    for alias in SWAP_ALIASES:
        assert MODEL_FORCE_MAP.get(alias) == SWAP_PORT, alias
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_swap_pool_routing.py::test_long_tail_aliases_route_to_swap_port -v`
Expected: FAIL — aliases still point at 9316/9328/9329/9330/9305.

- [ ] **Step 3: Repoint the aliases**

In `MODEL_FORCE_MAP`, change the port of every alias in `SWAP_ALIASES` to
`9350`. Example — the Devstral block becomes:

```python
    "ailiance-devstral-base": 9350,
    "ailiance-python": 9350,
    "ailiance-cpp": 9350,
    "ailiance-rust-emb": 9350,
    "ailiance-html": 9350,
    "ailiance-ml-training": 9350,
```

and likewise `ailiance-flagship`, `ailiance-qwen-235b`, `ailiance-mixtral`,
`ailiance-mixtral-8x22b`, `ailiance-llama`, `ailiance-qwen36` → `9350`.
Leave the `apertus-*` aliases for Step 5.

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_swap_pool_routing.py::test_long_tail_aliases_route_to_swap_port -v`
Expected: PASS.

- [ ] **Step 5: Decide the Apertus aliases**

The Apertus base model was deleted (2026-05-12). Per the spec's open
question: if `Apertus-70B-...-4bit-MLX` is present on disk (Task 1
inventory), route the 9 `apertus-*` aliases to `9350` and add them to
`SWAP_ALIASES` in the test. If absent, remove the 9 `apertus-*` entries
from `MODEL_FORCE_MAP` and from the `/v1/models` listing instead. Apply
whichever the inventory dictates; update the test to match.

- [ ] **Step 6: Commit**

```bash
git add src/gateway/server.py tests/test_swap_pool_routing.py
git commit -m "feat(gateway): route long-tail aliases to the swap server"
```

---

### Task 5: Add swap-server model rewrites

**Files:** Modify `src/gateway/server.py` — `ALIAS_MODEL_REWRITES`
(lines 529-560).

The swap `mlx_lm.server` loads `body["model"]` as an on-disk path. Each
swap alias needs an `ALIAS_MODEL_REWRITES` entry with the path recorded in
Task 1.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_swap_pool_routing.py`:

```python
def test_swap_aliases_have_a_model_rewrite():
    for alias in SWAP_ALIASES:
        assert alias in ALIAS_MODEL_REWRITES, alias
        assert ALIAS_MODEL_REWRITES[alias].get("model"), alias
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_swap_pool_routing.py::test_swap_aliases_have_a_model_rewrite -v`
Expected: FAIL — missing rewrites.

- [ ] **Step 3: Add the rewrites**

In `ALIAS_MODEL_REWRITES`, add one entry per swap alias using the path from
Task 1. Example shape (replace paths with the Task 1 inventory values):

```python
    "ailiance-llama": {"model": "/Users/clems/KIKI-Mac_tunner/models/Llama-3.3-70B-Instruct-MLX-4bit"},
    "ailiance-mixtral": {"model": "/Users/clems/KIKI-Mac_tunner/models/Mixtral-8x22B-Instruct-MLX-4bit"},
    "ailiance-qwen-235b": {"model": "/Users/clems/KIKI-Mac_tunner/models/Qwen3-235B-A22B-Instruct-MLX-4bit"},
    # ... one entry per alias in SWAP_ALIASES, paths from Task 1 ...
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_swap_pool_routing.py::test_swap_aliases_have_a_model_rewrite -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gateway/server.py tests/test_swap_pool_routing.py
git commit -m "feat(gateway): add swap-server model-path rewrites"
```

---

### Task 6: Verify the existing routing invariants still hold

**Files:** none modified — run the existing suite.

- [ ] **Step 1: Run the full gateway test suite**

Run: `uv run pytest tests/ -q`
Expected: PASS, including `test_gateway_alias_inventory.py` and the
`_validate` check that every `MODEL_FORCE_MAP` port is in `WORKER_URLS`
(9350 is now registered, so it passes).

- [ ] **Step 2: Fix any failure**

If `test_gateway_alias_inventory.py` asserts a removed Apertus alias, update
that test to match the Task 5 decision. Re-run until green.

- [ ] **Step 3: Commit any test updates**

```bash
git add tests/
git commit -m "test: realign alias inventory with swap-pool routing"
```

---

### Task 7: Deploy and verify end-to-end

- [ ] **Step 1: Open a PR and merge after CI passes**

```bash
git push -u origin feat/swap-pool-phase1
gh pr create --title "feat: on-demand swap server for long-tail models" --body "Phase 1 of the on-demand model loading spec."
```

- [ ] **Step 2: Deploy the gateway**

```bash
ssh electron-server "cd /home/electron/ailiance && git pull --ff-only origin main && sudo systemctl restart ailiance-gateway"
```

- [ ] **Step 3: End-to-end check**

```bash
for m in ailiance-llama ailiance-mixtral ailiance-qwen36; do
  curl -sS --max-time 180 -X POST https://gateway.ailiance.fr/v1/chat/completions \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"max_tokens\":4,\"stream\":false}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('$m ->', d.get('model'))"
done
```

Expected: each alias reports `served` = its real model (not `eu-kiki-gemma`).
First call per model is slow (cold load).

---

## Phase 2 — Swap pool + ModelManager (follow-on plan)

Phase 2 removes the single-server thrashing. It is large enough for its own
detailed plan, written once Phase 1 ships. Outline:

- **ModelManager module** (`src/gateway/model_manager.py`): `ALIAS_TIER`,
  `MODEL_FOOTPRINT`, `SWAP_SLOTS` tables; `resolve(alias) -> worker_url`
  with warm-hit / memory-aware LRU slot selection.
- **Second Studio swap server** (`:9351`) + **macM1 swap server** — extra
  plists, extra `WORKER_URLS` entries.
- **Server integration**: `forced_port = MODEL_FORCE_MAP.get(...)` becomes
  a `ModelManager.resolve(...)` call for swap-tier aliases; reuse the
  per-`worker_url` `asyncio.Lock` (already in `server.py`) to serialise
  swaps.
- **Cold-start signalling**: emit `event: loading` on the SSE stream when
  the chosen slot must reload; the cockpit playground renders it.
- **Demotion**: move `qwen36`, `eurollm`, `pixtral`, `qwen2.5-7B`,
  `qwen3-4b`, `mistral-small` from pinned to swap/macM1 tiers to free the
  ~190 GB swap budget (see spec "Memory budget").
- **Idle eviction**: optional background unload of swap models idle > N min.

Each Phase 2 component gets test-first tasks in its own plan document.

---

## Out of scope

- Auto-downloading absent model weights.
- On-the-fly quantization.
- kxkm-ai on-demand GGUF loading (separate llama.cpp stack).
