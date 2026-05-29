# Served-model derivation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the actually-serving specialist model in observability (#115) and stop carrying dead per-port aliases (#116), via one shared pure derivation + a boot-time self-maintaining force-map filter — no routing refactor.

**Architecture:** A pure `served_model_for()` in `alias_inventory.py` derives the served model/adapter from the existing `domain_map` maps. It feeds the `X-Ailiance-Served-Model` response header (in `_worker_headers`) and the `track_chat` audit stamp. Separately, a module-level `EFFECTIVE_FORCE_MAP` filters `MODEL_FORCE_MAP` to aliases whose port is in the resolved `WORKER_URLS`, used for `/v1/models` advertising and explicit force-routing.

**Tech Stack:** Python 3.13/3.14, FastAPI, pytest. Run tests with `.venv-gaiax/bin/python -m pytest ...`. Branch: `feat/served-model-derivation` (already created, spec committed). `main` is branch-protected (requires the `test` CI check) — land via PR.

---

## Conventions for every task

- Stay on branch `feat/served-model-derivation` (`git branch --show-current`).
- Tests: `.venv-gaiax/bin/python -m pytest <path> -v`.
- **Before the FINAL commit/PR, run the FULL CI command** `.venv-gaiax/bin/python -m pytest -m "not network and not e2e" -q` and confirm `0 failed` (cross-module breakage has bitten before — the per-file run is not enough).
- Commit hook (STRICT): subject ≤50 chars, conventional type `feat`/`fix`/`docs`/`refactor`/`chore` (NOT `test`), scope may use hyphen, no AI attribution, no `--no-verify`.

## File Structure

- Modify `src/gateway/alias_inventory.py` — add `served_model_for()` (imports the domain maps from `src.router.domain_map`; `domain_map` does not import this module → no cycle).
- Modify `src/gateway/server.py` — `_worker_headers` adds the `X-Ailiance-Served-Model` header; success-path `track_chat` calls pass `served_model`; module-level `EFFECTIVE_FORCE_MAP` + rewire advertising/force-routing to it.
- Modify `src/gateway/observability.py` — `track_chat` + `_send_trace` accept a `served_model` field.
- Tests: `tests/test_gateway_alias_inventory.py` (served_model_for unit), `tests/test_gateway.py` (header + force-map integration).

---

## Task 1: `served_model_for()` derivation

**Files:**
- Modify: `src/gateway/alias_inventory.py` (add import + function after `_REGISTRY`)
- Test: `tests/test_gateway_alias_inventory.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gateway_alias_inventory.py`:

```python
class TestServedModelFor:
    def test_qwen36_instance_a_domain(self):
        from src.router.domain_map import QWEN36_PORT
        from src.gateway.alias_inventory import served_model_for
        # "emc" is a hardware domain on instance A (:9360)
        assert served_model_for(alias="ailiance", domain="emc",
                                worker_port=QWEN36_PORT) == "qwen36-emc-dsp-power"

    def test_qwen36_instance_b_domain(self):
        from src.router.domain_map import QWEN36_PORT_B
        from src.gateway.alias_inventory import served_model_for
        # "cpp" is a code domain on instance B (:9361)
        assert served_model_for(alias="ailiance", domain="cpp",
                                worker_port=QWEN36_PORT_B) == "qwen36-cpp"

    def test_omlx_domain(self):
        from src.router.domain_map import OMLX_PORT, DOMAIN_TO_OMLX_MODEL
        from src.gateway.alias_inventory import served_model_for
        # "python" stays on omlx :8500
        assert served_model_for(alias="ailiance", domain="python",
                                worker_port=OMLX_PORT) == DOMAIN_TO_OMLX_MODEL["python"]

    def test_explicit_alias_uses_registry_base_model(self):
        from src.gateway.alias_inventory import served_model_for, _REGISTRY
        # explicit alias, no classifier domain → registry base_model
        expected = _REGISTRY["ailiance-eurollm"].base_model
        assert served_model_for(alias="ailiance-eurollm", domain=None,
                                worker_port=9303) == expected

    def test_unknown_falls_back_to_alias(self):
        from src.gateway.alias_inventory import served_model_for
        assert served_model_for(alias="ailiance-nope", domain=None,
                                worker_port=12345) == "ailiance-nope"

    def test_never_raises_on_empty(self):
        from src.gateway.alias_inventory import served_model_for
        assert served_model_for(alias="", domain=None, worker_port=0) == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-gaiax/bin/python -m pytest tests/test_gateway_alias_inventory.py::TestServedModelFor -v`
Expected: FAIL — `ImportError: cannot import name 'served_model_for'`

- [ ] **Step 3: Add the import near the top of `src/gateway/alias_inventory.py`**

After the existing `from dataclasses import ...` import block, add:

```python
from src.router.domain_map import (
    DOMAIN_TO_OMLX_MODEL,
    DOMAIN_TO_QWEN36,
    OMLX_PORT,
    QWEN36_PORT,
    QWEN36_PORT_B,
)

_QWEN36_PORTS = frozenset({QWEN36_PORT, QWEN36_PORT_B})
```

- [ ] **Step 4: Add the function after the `_REGISTRY` definition** (anywhere after `_REGISTRY = {...}`, e.g. next to `resolve_effective_alias`)

```python
def served_model_for(*, alias: str, domain: str | None, worker_port: int) -> str:
    """Derive the model/adapter actually serving a routed request.

    For observability only (X-Ailiance-Served-Model header + audit
    stamp) — never used for routing. Resolution:

    - auto-routed domain on a qwen36 instance (:9360/:9361) → the
      qwen36 adapter name (DOMAIN_TO_QWEN36).
    - auto-routed domain on omlx (:8500) → the omlx model
      (DOMAIN_TO_OMLX_MODEL).
    - otherwise (explicit alias, no classifier domain) → the alias's
      registry base_model, falling back to the alias string.

    Never raises; returns "unknown" only when given no usable input.
    """
    if domain:
        if worker_port in _QWEN36_PORTS:
            adapter = DOMAIN_TO_QWEN36.get(domain)
            if adapter:
                return adapter
        elif worker_port == OMLX_PORT:
            model = DOMAIN_TO_OMLX_MODEL.get(domain)
            if model:
                return model
    inv = _REGISTRY.get(alias)
    if inv and inv.base_model:
        return inv.base_model
    return alias or "unknown"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv-gaiax/bin/python -m pytest tests/test_gateway_alias_inventory.py::TestServedModelFor -v`
Expected: PASS (6 tests). If `served_model_for(alias="ailiance", ...)` hits the registry fallback for a no-domain case and returns "auto-router", that's only for the no-domain branch — the domain tests assert the specialist correctly.

- [ ] **Step 6: Commit**

```bash
git add src/gateway/alias_inventory.py tests/test_gateway_alias_inventory.py
git commit -m "feat(gateway): served_model_for derivation helper"
```

---

## Task 2: `X-Ailiance-Served-Model` header

**Files:**
- Modify: `src/gateway/server.py` (`_worker_headers`, ~line 329; + import)
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gateway.py`:

```python
def test_worker_headers_includes_served_model():
    from src.router.domain_map import QWEN36_PORT
    from src.gateway.server import _worker_headers
    h = _worker_headers(
        worker_port=QWEN36_PORT,
        domain="emc",
        effective_alias="ailiance",
    )
    assert h["X-Ailiance-Served-Model"] == "qwen36-emc-dsp-power"


def test_worker_headers_omits_served_model_when_uninformative():
    from src.gateway.server import _worker_headers
    # no domain + auto-router alias → no meaningful specialist → header absent
    h = _worker_headers(worker_port=9304, domain="", effective_alias="ailiance")
    assert "X-Ailiance-Served-Model" not in h
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-gaiax/bin/python -m pytest tests/test_gateway.py::test_worker_headers_includes_served_model -v`
Expected: FAIL — KeyError `X-Ailiance-Served-Model`

- [ ] **Step 3: Import `served_model_for` in `src/gateway/server.py`**

In the existing `from src.gateway.alias_inventory import (...)` block near the top, add `served_model_for` to the imported names.

- [ ] **Step 4: Add the header inside `_worker_headers`**

In `_worker_headers` (~line 329), after the base `headers = {...}` dict is built and before the `return headers`, add:

```python
    served = served_model_for(
        alias=effective_alias or "",
        domain=domain,
        worker_port=worker_port,
    )
    if served and served not in ("unknown", "auto-router"):
        headers["X-Ailiance-Served-Model"] = served
```

(`effective_alias` and `domain` and `worker_port` are existing parameters of `_worker_headers`. The `("unknown", "auto-router")` guard suppresses the non-informative fallbacks so the header only appears when a real specialist served the request.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv-gaiax/bin/python -m pytest tests/test_gateway.py::test_worker_headers_includes_served_model tests/test_gateway.py::test_worker_headers_omits_served_model_when_uninformative -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/gateway/server.py tests/test_gateway.py
git commit -m "feat(gateway): X-Ailiance-Served-Model header"
```

---

## Task 3: `served_model` in the audit stamp

**Files:**
- Modify: `src/gateway/observability.py` (`track_chat` + `_send_trace`)
- Modify: `src/gateway/server.py` (success-path `track_chat` calls)
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gateway.py`:

```python
def test_track_chat_accepts_served_model(monkeypatch):
    import src.gateway.observability as obs
    captured = {}

    async def _fake_send_trace(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(obs, "_send_trace", _fake_send_trace)
    # must run inside an event loop so create_task works
    import asyncio

    async def _run():
        obs.track_chat(
            model_alias="ailiance", domain="emc", kind="direct",
            request_body={}, response_body={}, started_at=0.0,
            served_model="qwen36-emc-dsp-power",
        )
        await asyncio.sleep(0)  # let the created task run

    asyncio.run(_run())
    assert captured.get("served_model") == "qwen36-emc-dsp-power"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-gaiax/bin/python -m pytest tests/test_gateway.py::test_track_chat_accepts_served_model -v`
Expected: FAIL — `TypeError: track_chat() got an unexpected keyword argument 'served_model'`

- [ ] **Step 3: Add the param to `track_chat` and `_send_trace` in `src/gateway/observability.py`**

In `track_chat` (~line 132), add `served_model: str | None = None,` to the keyword-only params (e.g. after `upstream_model`), and pass it through to `_send_trace`:

```python
def track_chat(
    *,
    model_alias: str,
    domain: str,
    kind: str,
    request_body: dict,
    response_body: dict,
    started_at: float,
    upstream_model: str | None = None,
    served_model: str | None = None,
    chain_id: str | None = None,
    error: str | None = None,
) -> None:
    ...
    asyncio.get_event_loop().create_task(
        _send_trace(
            model_alias=model_alias,
            domain=domain,
            kind=kind,
            request_body=request_body,
            response_body=response_body,
            latency_ms=latency_ms,
            upstream_model=upstream_model,
            served_model=served_model,
            chain_id=chain_id,
            error=error,
        )
    )
```

In `_send_trace`, add `served_model: str | None = None,` to its signature and record it wherever the trace fields are assembled (mirror how `upstream_model` is stored in the trace payload — find the dict/record `_send_trace` builds and add `"served_model": served_model`).

- [ ] **Step 4: Pass `served_model` at the SUCCESS-path `track_chat` calls in `src/gateway/server.py`**

There are several `track_chat(...)` calls (lines ~1697, 1842, 1858, 1891, 1925, 1973, 2004). Add `served_model=...` ONLY to the ones on a successful completion path (those that pass a real `response_body=` and no `error=`). For each such success call, pass:

```python
                served_model=served_model_for(
                    alias=effective_alias or req.model,
                    domain=domain,
                    worker_port=worker_port,
                ),
```

Read each `track_chat` call's surrounding context: if it has `error=...` (the 502/503/empty/mid-stream branches from earlier work), LEAVE IT unchanged. If it stamps a successful response (streaming success + non-streaming success), add the `served_model=` line. `served_model_for`, `effective_alias`/`req.model`, `domain`, `worker_port` are all in scope at the dispatch. (`served_model_for` is already imported from Task 2.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv-gaiax/bin/python -m pytest tests/test_gateway.py::test_track_chat_accepts_served_model -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/gateway/observability.py src/gateway/server.py tests/test_gateway.py
git commit -m "feat(gateway): served_model in audit stamp"
```

---

## Task 4: `EFFECTIVE_FORCE_MAP` (self-maintaining #116)

**Files:**
- Modify: `src/gateway/server.py` (define `EFFECTIVE_FORCE_MAP`; rewire advertising + force-routing)
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gateway.py`:

```python
def test_effective_force_map_drops_unconfigured_ports():
    from src.gateway.server import EFFECTIVE_FORCE_MAP, MODEL_FORCE_MAP, WORKER_URLS
    # every effective alias maps to a configured worker port
    for alias, port in EFFECTIVE_FORCE_MAP.items():
        assert port in WORKER_URLS, f"{alias}:{port} not in WORKER_URLS"
    # an alias whose MODEL_FORCE_MAP port is NOT configured is dropped
    dead = [a for a, p in MODEL_FORCE_MAP.items() if p not in WORKER_URLS]
    for a in dead:
        assert a not in EFFECTIVE_FORCE_MAP


def test_effective_force_map_not_empty_at_cold_start():
    # _DEFAULT_WORKER_URLS always backs the core ports → never fully empty
    from src.gateway.server import EFFECTIVE_FORCE_MAP
    assert len(EFFECTIVE_FORCE_MAP) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-gaiax/bin/python -m pytest tests/test_gateway.py::test_effective_force_map_drops_unconfigured_ports -v`
Expected: FAIL — `ImportError: cannot import name 'EFFECTIVE_FORCE_MAP'`

- [ ] **Step 3: Define `EFFECTIVE_FORCE_MAP` after `MODEL_FORCE_MAP`**

In `src/gateway/server.py`, immediately after the `MODEL_FORCE_MAP = {...}` block (ends near line 467; `WORKER_URLS` is already defined at line 161), add:

```python
# Self-maintaining force-map: only aliases whose port is actually in the
# resolved worker table (AILIANCE_WORKERS_JSON over _DEFAULT_WORKER_URLS).
# Retired per-port workers drop automatically — no manual prune, no
# operator data needed. Complements the runtime liveness filter (#12):
# this drops *never-configured* ports; liveness drops *configured-but-down*.
EFFECTIVE_FORCE_MAP = {
    alias: port for alias, port in MODEL_FORCE_MAP.items() if port in WORKER_URLS
}
```

- [ ] **Step 4: Rewire advertising + force-routing to `EFFECTIVE_FORCE_MAP`**

Make these substitutions (read each site first; only the routing/advertising reads change — do NOT touch the training-`unloaded` display logic):

1. `_compute_public_aliases` (~line 482-503, the loop `for alias in MODEL_FORCE_MAP:` at ~499) — iterate `EFFECTIVE_FORCE_MAP` instead, so `ALL_PUBLIC_ALIASES` only contains configured aliases.
2. The non-streaming dispatch force lookup at ~line 1453: `forced_port = MODEL_FORCE_MAP.get(req.model)` → `forced_port = EFFECTIVE_FORCE_MAP.get(req.model)`. (Effect: an explicit `model=ailiance-<dead>` no longer routes to a dead port; `forced_port` is None → falls through to the existing auto-route/unknown handling.)
3. The helper at ~line 1129: `forced = MODEL_FORCE_MAP.get(model, HEALTH_FALLBACK_PORT)` → `forced = EFFECTIVE_FORCE_MAP.get(model, HEALTH_FALLBACK_PORT)`.
4. The `/v1/models` liveness filter at ~line 1211 (`MODEL_FORCE_MAP.get(a) in _healthy_ports`): leave as-is OR switch to `EFFECTIVE_FORCE_MAP.get(a)` — since `ids` now derive from `EFFECTIVE_FORCE_MAP` via `ALL_PUBLIC_ALIASES`, both are equivalent; prefer `EFFECTIVE_FORCE_MAP` for consistency.

Leave the training-unloaded reference (~line 1228) on `MODEL_FORCE_MAP` (it is about training state, orthogonal to liveness).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv-gaiax/bin/python -m pytest tests/test_gateway.py::test_effective_force_map_drops_unconfigured_ports tests/test_gateway.py::test_effective_force_map_not_empty_at_cold_start tests/test_gateway_alias_inventory.py -v`
Expected: PASS. If `TestCatalogCoverage` now fails because an alias that WAS advertised is dropped (its port unconfigured), that is the intended #116 effect — but verify the dropped alias genuinely has an unconfigured port; if a still-served alias got dropped, the port is missing from `_DEFAULT_WORKER_URLS`/`AILIANCE_WORKERS_JSON` and should be added there, not worked around.

- [ ] **Step 6: Commit**

```bash
git add src/gateway/server.py tests/test_gateway.py
git commit -m "feat(gateway): self-maintaining effective force-map"
```

---

## Task 5: Full verification + PR

- [ ] **Step 1: Run the FULL CI command**

Run: `.venv-gaiax/bin/python -m pytest -m "not network and not e2e" -q`
Expected: `0 failed` (skips/deselected/xfailed allowed). If a pre-existing test asserts on the full `MODEL_FORCE_MAP` advertising surface and now sees fewer aliases (Task 4 effect), update that assertion to `EFFECTIVE_FORCE_MAP` semantics — that is the intended behavior, not a regression.

- [ ] **Step 2: Push the branch**

```bash
git push -u origin feat/served-model-derivation
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create --repo ailiance/ailiance \
  --title "feat: served-model derivation (#115, #116)" \
  --base main --head feat/served-model-derivation \
  --body "Implements docs/superpowers/specs/2026-05-29-served-model-derivation-design.md. #115: served_model_for() → X-Ailiance-Served-Model header + audit stamp (resolve_effective_alias/body.model unchanged). #116: EFFECTIVE_FORCE_MAP self-maintaining filter (no manual prune). Closes #115, #116."
```

- [ ] **Step 4: Wait for CI `test` green, then merge**

```bash
gh pr checks <PR#> --repo ailiance/ailiance   # wait for test=pass
gh pr merge <PR#> --repo ailiance/ailiance --squash --delete-branch
```

---

## Self-Review

**Spec coverage:**
- Component 1 `served_model_for` → Task 1. ✅
- Component 2 #115 header + audit → Task 2 (header) + Task 3 (audit). ✅
- Component 3 #116 boot-filter → Task 4. ✅
- "never raises" / cold-start safety → Task 1 (fallback) + Task 4 (cold-start test). ✅
- resolve_effective_alias / body.model unchanged → no task touches them. ✅
- Testing section → each task has TDD steps. ✅

**Placeholder scan:** Task 3 Step 4 and Task 4 Step 4 require reading the actual call sites (multiple `track_chat`/`MODEL_FORCE_MAP` references) — the exact lines are given with the substitution and the rule (success-only / routing-only), which is the real instruction, not a placeholder. No "TBD"/"handle edge cases" left.

**Type consistency:** `served_model_for(*, alias: str, domain: str | None, worker_port: int) -> str` used identically in Tasks 1, 2, 3. `EFFECTIVE_FORCE_MAP: dict[str, int]` used consistently in Task 4. `track_chat(..., served_model: str | None = None, ...)` matches its call in Task 3.
