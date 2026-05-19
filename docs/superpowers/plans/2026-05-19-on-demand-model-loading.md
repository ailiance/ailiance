# On-demand Model Loading Implementation Plan (rev. A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Serve every gateway alias's real model — long-tail models load on
demand — without crashing the memory-constrained Mac Studio.

**Architecture:** Free RAM first (stop low-traffic resident servers), then a
swap `mlx_lm.server` loads distinct base models on demand. LoRA-variant
families and the multi-slot pool are later phases.

**Tech Stack:** Python 3.14, FastAPI gateway (`src/gateway/server.py`),
`mlx_lm.server` on Apple Silicon, pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-19-on-demand-model-loading-design.md`

## Model taxonomy (drives the phasing)

- **Distinct text base models** → swap pool: `llama` (Llama-3.3-70B),
  `mixtral`/`mixtral-8x22b` (Mixtral-8x22B), `qwen-235b`/`flagship`
  (Qwen3-235B), `qwen36` (Qwen3.6-35B), `devstral-base` (Devstral-24B),
  `eurollm` (EuroLLM-22B), `mistral-small` (Mistral-Small-24B). All
  confirmed on disk under `/Users/clems/KIKI-Mac_tunner/models/`.
- **LoRA variants** → multi-LoRA servers (Phase 1bis): 9 `apertus-*`,
  5 devstral (`python`/`cpp`/`rust-emb`/`html`/`ml-training`).
- **Vision** → `pixtral` needs `mlx_vlm`, not `mlx_lm`; keep its own small
  resident server (~7 GB), not in the swap pool.
- **Local-only** (no gateway alias): `qwen2.5-7B` :8501, `qwen3-4b` :9341 —
  stopping them frees RAM with no routing change.

**Hard memory rule:** never route a model to a swap server unless the
server's free budget ≥ the model's footprint. Routing Qwen3-235B (~120 GB)
into ~92 GB free would OOM and risk a kernel panic (precedent 2026-05-12).
Phase 1 frees the RAM before Phase 2 routes the big models.

---

## Phase 1 — Free RAM (demotion)

Stop the low-traffic resident MLX servers so later phases have a real swap
budget. Phase 1 is ops-only; it changes no gateway code. Standalone
outcome: ~87 GB reclaimed, Studio free RAM ~92 → ~180 GB.

Servers to stop (measured RSS): `qwen36` :9305 (~19 GB), `eurollm` :9303
(~43 GB), `qwen2.5-7B` :8501 (~4 GB), `qwen3-4b` :9341 (~8 GB),
`mistral-small` :9326 (~13 GB). `pixtral` stays (vision, only ~7 GB).

### Task 1: Inventory — DONE

The Studio model library was inventoried 2026-05-19. All swap-pool base
models are present on disk. No action; recorded for reference.

### Task 2: Identify each cold server's launchd job / PID

- [ ] **Step 1: List the launchd jobs and PIDs for the 5 cold servers**

On the Studio (Terminal or bastion):

```bash
launchctl list | grep -iE 'cc.ailiance|cc.kiki|mlx'
ps -axo pid,command | grep -E 'mlx_lm|mlx-lm' | grep -v grep
```

Map each of the 5 cold ports (9305, 9303, 8501, 9341, 9326) to its launchd
label (e.g. `cc.ailiance.qwen36`) or bare PID.

### Task 3: Stop the 5 cold servers

- [ ] **Step 1: Stop each, one at a time, watching memory**

For a launchd-managed server: `launchctl bootout gui/$(id -u)/<label>`.
For a bare process: `kill <pid>`. After each, check `top -l1 | grep PhysMem`
and confirm free RAM rises. Stop if anything unexpected happens.

- [ ] **Step 2: Verify ~180 GB free**

```bash
top -l 1 -n 0 | grep PhysMem
```

Expected: unused ≥ ~170 GB.

### Task 4: Mark the demoted aliases pending in the gateway

The aliases `ailiance-qwen36`, `ailiance-eurollm`, `ailiance-mistral-small`
now point at stopped ports. Until Phase 2 routes them to the swap pool they
will fall back to Gemma — acceptable and unchanged from today's degraded
state. No code change in Phase 1.

- [ ] **Step 1: Note the degraded aliases in the gateway audit log**

Append to `docs/transparency/router-training-data.md` (or the audit dir) a
dated line listing the 3 demoted aliases as "pending swap-pool routing
(Phase 2)". Commit.

---

## Phase 2 — Swap server + on-demand base models

With ~180 GB free, stand up the swap server and route every distinct text
base model to it. Standalone outcome: `llama`, `mixtral`, `qwen-235b`,
`qwen36`, `devstral-base`, `eurollm`, `mistral-small` all serve their real
model on demand instead of Gemma.

### File structure
- Modify `src/gateway/server.py` — `_DEFAULT_WORKER_URLS`, `MODEL_FORCE_MAP`,
  `ALIAS_MODEL_REWRITES`.
- Create `tests/test_swap_pool_routing.py`.
- Ops: swap `mlx_lm.server` on `:9350` + autossh tunnel.

### Task 1: Launch the swap mlx_lm.server (:9350)

- [ ] **Step 1: Start it (bastion, nohup — survives over SSH)**

```bash
ssh electron-server "ssh clems@100.116.92.12 'cd ~ && nohup /Users/clems/.venv-mistral/bin/python -m mlx_lm.server --model /Users/clems/KIKI-Mac_tunner/models/Qwen3.5-4B --host 0.0.0.0 --port 9350 --log-level INFO > /private/tmp/ailiance-swap-1.log 2>&1 &'"
```

(Small default model so it boots fast; it swaps to the requested model on
each request.)

- [ ] **Step 2: Verify the ModelProvider frees the old model on swap**

Send two requests naming different models; watch `top` RSS. If RSS spikes
to old+new (double load), add a guarded `mx.clear_cache()` shim — record
the finding before relying on big-model swaps.

- [ ] **Step 3: Add the autossh tunnel electron-server:9350 → studio:9350**

Mirror `mascarade-studio-tunnel.service`. Verify:
`ssh electron-server "curl -sf -o /dev/null -w '%{http_code}' http://localhost:9350/v1/models"` → `200`.

### Task 2: Register the swap port in WORKER_URLS

- [ ] **Step 1: Write the failing test**

Create `tests/test_swap_pool_routing.py`:

```python
"""Swap-pool routing: distinct base models reach the swap server."""
from src.gateway.server import (
    ALIAS_MODEL_REWRITES,
    MODEL_FORCE_MAP,
    WORKER_URLS,
)

SWAP_PORT = 9350

SWAP_ALIASES = [
    "ailiance-llama", "ailiance-mixtral", "ailiance-mixtral-8x22b",
    "ailiance-qwen-235b", "ailiance-flagship", "ailiance-qwen36",
    "ailiance-devstral-base", "ailiance-mistral-small",
]
# Note: EuroLLM is stopped in Phase 1 to free RAM but has no public gateway
# alias (absent from /v1/models), so it is not in SWAP_ALIASES.


def test_swap_port_is_registered():
    assert SWAP_PORT in WORKER_URLS
```

- [ ] **Step 2: Run it, verify it fails** — `uv run pytest tests/test_swap_pool_routing.py::test_swap_port_is_registered -v` → FAIL.

- [ ] **Step 3: Add the port** — in `_DEFAULT_WORKER_URLS`, after the `9340`
entry:

```python
    # Studio swap server :9350 — one mlx_lm.server, no fixed model; loads
    # the requested base model on demand. autossh tunnel :9350 → studio.
    9350: "http://localhost:9350",
```

- [ ] **Step 4: Run it, verify it passes.**

- [ ] **Step 5: Commit** — `git commit -m "feat(gateway): register swap server port 9350"`.

### Task 3: Route base aliases to the swap port

- [ ] **Step 1: Write the failing test** — append:

```python
def test_base_aliases_route_to_swap_port():
    for alias in SWAP_ALIASES:
        assert MODEL_FORCE_MAP.get(alias) == SWAP_PORT, alias
```

- [ ] **Step 2: Run it, verify it fails.**

- [ ] **Step 3: Repoint** — set every `SWAP_ALIASES` entry's `MODEL_FORCE_MAP`
port to `9350`. Leave `apertus-*`, the 5 devstral LoRA aliases, and
`pixtral` untouched.

- [ ] **Step 4: Run it, verify it passes.**

- [ ] **Step 5: Commit** — `git commit -m "feat(gateway): route base models to the swap server"`.

### Task 4: Add swap-server model rewrites

- [ ] **Step 1: Write the failing test** — append:

```python
def test_swap_aliases_have_a_model_rewrite():
    for alias in SWAP_ALIASES:
        assert alias in ALIAS_MODEL_REWRITES, alias
        assert ALIAS_MODEL_REWRITES[alias].get("model"), alias
```

- [ ] **Step 2: Run it, verify it fails.**

- [ ] **Step 3: Add the rewrites** — in `ALIAS_MODEL_REWRITES`, one entry per
swap alias pointing at the on-disk path (Task 1 inventory):

```python
    "ailiance-llama": {"model": "/Users/clems/KIKI-Mac_tunner/models/Llama-3.3-70B-Instruct-MLX-4bit"},
    "ailiance-mixtral": {"model": "/Users/clems/KIKI-Mac_tunner/models/Mixtral-8x22B-Instruct-MLX-4bit"},
    "ailiance-mixtral-8x22b": {"model": "/Users/clems/KIKI-Mac_tunner/models/Mixtral-8x22B-Instruct-MLX-4bit"},
    "ailiance-qwen-235b": {"model": "/Users/clems/KIKI-Mac_tunner/models/Qwen3-235B-A22B-Instruct-MLX-4bit"},
    "ailiance-flagship": {"model": "/Users/clems/KIKI-Mac_tunner/models/Qwen3-235B-A22B-Instruct-MLX-4bit"},
    "ailiance-qwen36": {"model": "/Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B-MLX-BF16"},
    "ailiance-devstral-base": {"model": "/Users/clems/KIKI-Mac_tunner/models/Devstral-Small-2-24B-MLX-4bit"},
    "ailiance-eurollm": {"model": "/Users/clems/KIKI-Mac_tunner/models/EuroLLM-22B-Instruct-2512"},
    "ailiance-mistral-small": {"model": "/Users/clems/KIKI-Mac_tunner/models/Mistral-Small-3.1-24B-Instruct-MLX-4bit"},
```

- [ ] **Step 4: Run it, verify it passes.**

- [ ] **Step 5: Commit** — `git commit -m "feat(gateway): add swap-server model-path rewrites"`.

### Task 5: Full suite + deploy

- [ ] **Step 1: Run the suite** — `uv run pytest tests/ -q`. Fix any
`test_gateway_alias_inventory.py` drift; the `_validate` port check passes
(9350 registered).

- [ ] **Step 2: PR, CI, merge** — `gh pr create --title "feat: swap server for on-demand base models"`.

- [ ] **Step 3: Deploy** — `ssh electron-server "cd /home/electron/ailiance && git pull --ff-only origin main && sudo systemctl restart ailiance-gateway"`.

- [ ] **Step 4: End-to-end** — POST a short completion to `ailiance-llama`,
`ailiance-mixtral`, `ailiance-qwen36`; each must report `served` = its real
model. First call per model is slow (cold load).

---

## Phase 1bis — LoRA-variant families (multi-LoRA servers)

The 9 `apertus-*` and 5 devstral LoRA aliases need the multi-LoRA-server
pattern (one base in VRAM, adapters hot-swapped per request) — as
`mascarade_multi_server` (:9340) already does. Outline: stand up one
multi-LoRA server per family (Apertus-70B-4bit + 9 adapters, Devstral-24B +
5 adapters) reusing that code; repoint the 14 aliases; confirm adapter
weights under `/Users/clems/lora-adapters`. Own detailed plan later.

## Phase 3 — Multi-slot pool + ModelManager

Single-server swapping thrashes when two cold models alternate. Phase 3
adds `src/gateway/model_manager.py` (2-3 slots, memory-aware LRU routing,
per-slot `asyncio.Lock`), a macM1 swap server, and `event: loading` SSE
signalling. Own detailed plan later.

## Out of scope

- Auto-downloading absent weights; on-the-fly quantization.
- kxkm-ai on-demand GGUF (separate llama.cpp stack).
- Vision-model swapping (`pixtral` keeps a dedicated `mlx_vlm` server).
