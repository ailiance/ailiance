# Eval Framework Phase 3 OOM Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `eval_framework.py --mode compare` complete the full 49-cell EU AI Act bench matrix (Apertus + Devstral + EuroLLM + Mistral Medium 3.5 + Qwen3.6 × 31 domains) without being killed by the kernel OOM signal that aborts it within seconds today.

**Architecture:** Three layers of defense. (1) Lower `mx.set_memory_limit` below the macOS `iogpu.wired_limit_mb` so the soft cap can never exceed the hard kernel cap. (2) Add a `--mode sequential-strict` runtime flag that finishes every adapter for one base model before unloading and moving to the next, with a peak-memory probe between each transition that aborts with a clean error if the threshold is breached. (3) Wrap the subprocess in a wall-time + exit-code aware launcher so SIGKILL produces a structured report instead of a `Killed: 9` line.

**Tech Stack:** Python 3.13 (`/Users/clems/KIKI-Mac_tunner/.venv/bin/python`), MLX (`mlx_lm_fork` vendored at `~/KIKI-Mac_tunner/lib/mlx_lm_fork`), bash (`run_eval.sh`), pytest 8.x for the unit tests.

---

## Repo + locations

- All code lives in **`L-electron-Rare/eu-kiki`** (GitHub-redirected to `ailiance`), executed on **Studio (`100.116.92.12`)**.
- Code path: `~/eu-kiki/scripts/eval_framework.py` (1329 lines), `~/eu-kiki/scripts/run_eval.sh` (~250 lines).
- Tests path: `~/eu-kiki/tests/test_eval_framework.py` (new, the directory may not exist yet).
- Plan path: `~/eu-kiki/docs/superpowers/plans/2026-05-10-eval-framework-oom-fix.md` (new).
- All commits authored as `electron-rare <108685187+electron-rare@users.noreply.github.com>` (GitHub email-privacy enforced).

## Pre-flight context

Current `run_eval.sh --compare` symptom (2026-05-10 17:35 CEST):

```
Running: /Users/clems/KIKI-Mac_tunner/.venv/bin/python /Users/clems/eu-kiki/scripts/eval_framework.py --mode compare
mx.metal.get_peak_memory is deprecated and will be removed in a future version. Use mx.get_peak_memory instead.
mx.metal.clear_cache is deprecated and will be removed in a future version. Use mx.clear_cache instead.
/Users/clems/eu-kiki/scripts/run_eval.sh: line 230: 61345 Killed: 9
```

Crashes between the first `import mlx` and the first `Phase 1` banner — ~3 seconds of process life. No traceback. Strongly suggests kernel SIGKILL on a memory-budget violation, not a Python exception.

Studio config (`sysctl iogpu.wired_limit_mb` = 458752) means Metal can wire up to 448 GiB. The framework calls `mx.set_memory_limit(480 * 1024**3)` = **480 GiB > 448 GiB hard cap** at line 457. That's the smoking gun for Task 2.

Total physical RAM: 512 GiB. macOS reserves ~64 GiB for the kernel/system, leaving ~448 GiB for Metal — exactly the wired limit. Going above invites the kernel to reject reservations or kill the process.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `scripts/eval_framework.py` | Modify | Lower mem cap, add `--mode sequential-strict` flag, add `_log_peak_memory()` probe between groups, add SIGKILL-clean abort if peak > threshold |
| `scripts/run_eval.sh` | Modify | Forward new `--mode sequential-strict` flag, detect SIGKILL exit codes (137/139) and emit a structured failure summary |
| `scripts/launch_eval_safe.sh` | Create | Wrapper that runs `run_eval.sh` and writes `output/eval/last_run_status.json` (exit_code, signal, wall_time, peak_mem if available) |
| `tests/__init__.py` | Create | Empty marker so pytest discovers the package |
| `tests/test_eval_framework.py` | Create | Unit tests: `_log_peak_memory()`, `parse_args()` accepts new mode, `assert_within_budget()` raises cleanly, `--mode sequential-strict` orders the load_groups iteration deterministically |
| `tests/test_run_eval_sh.py` | Create | Smoke test: invokes `run_eval.sh --quick --v1-only --domains math-gsm8k`, asserts exit 0 + raw output JSON exists |
| `docs/CLAUDE.md` | Modify | Add note in the Roadmap → 🔴 Bloquants section recording the fix + the `--mode sequential-strict` runbook |

Each file has one responsibility. The launcher wrapper is separate from `run_eval.sh` because the wrapper writes a structured status file that downstream automation (Grist sync, the `bnpdsbg95`-style background polls) can consume without parsing logs.

---

## Task 1: Reproducer test that pins the failure

**Files:**
- Create: `~/eu-kiki/tests/__init__.py`
- Create: `~/eu-kiki/tests/test_eval_framework.py`

We need a fast unit test that fails today and passes once the memory cap is lowered. Targeting `mx.set_memory_limit` directly — no need to actually load models.

- [ ] **Step 1: Create the empty test package marker**

```bash
ssh studio 'mkdir -p ~/eu-kiki/tests && touch ~/eu-kiki/tests/__init__.py'
```

- [ ] **Step 2: Write the failing test**

Write the file `~/eu-kiki/tests/test_eval_framework.py`:

```python
"""Unit tests for eval_framework.py memory budget + mode dispatch."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _read_set_memory_limit_call() -> int:
    """Return the constant passed to mx.set_memory_limit by reading the source.

    Static read avoids importing mlx in CI environments without it.
    """
    src = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "eval_framework.py"
    ).read_text()
    import re
    m = re.search(r"mx\.set_memory_limit\((\d+)\s*\*\s*1024\*\*3\)", src)
    assert m, "mx.set_memory_limit(<gib> * 1024**3) call not found"
    return int(m.group(1))


def test_memory_limit_below_wired_cap():
    """The Python soft cap must stay below the macOS wired hard cap (448 GiB).

    Studio iogpu.wired_limit_mb is 458752 (= 448 GiB). Setting MLX above that
    invites the kernel to SIGKILL the process the moment Metal tries to wire
    a buffer that would exceed the wired pool.
    """
    gib = _read_set_memory_limit_call()
    assert gib <= 440, (
        f"mx.set_memory_limit({gib} GiB) must stay <= 440 GiB to leave"
        " 8 GiB headroom under the 448 GiB iogpu.wired_limit_mb hard cap."
    )
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
ssh studio 'cd ~/eu-kiki && /Users/clems/KIKI-Mac_tunner/.venv/bin/python -m pytest tests/test_eval_framework.py::test_memory_limit_below_wired_cap -v'
```

Expected: **FAIL** with `AssertionError: mx.set_memory_limit(480 GiB) must stay <= 440 GiB ...`.

- [ ] **Step 4: Commit the failing test**

```bash
ssh studio 'cd ~/eu-kiki && git add tests/__init__.py tests/test_eval_framework.py && GIT_EDITOR=true git -c user.name=electron-rare -c user.email=108685187+electron-rare@users.noreply.github.com commit -m "test: add memory-cap reproducer for eval_framework OOM"'
```

---

## Task 2: Lower the soft memory cap below the wired hard cap

**Files:**
- Modify: `~/eu-kiki/scripts/eval_framework.py:455-460` (the `mx.set_memory_limit` call inside `load_model_and_tokenizer`)

- [ ] **Step 1: Apply the patch**

Run on Studio:

```bash
ssh studio 'sed -i "" "s|mx.set_memory_limit(480 \* 1024\*\*3)|mx.set_memory_limit(440 * 1024**3)|" ~/eu-kiki/scripts/eval_framework.py'
```

- [ ] **Step 2: Verify the change**

```bash
ssh studio 'grep -n "mx.set_memory_limit" ~/eu-kiki/scripts/eval_framework.py'
```

Expected: prints exactly one match `mx.set_memory_limit(440 * 1024**3)`.

- [ ] **Step 3: Run the failing test, expect PASS**

```bash
ssh studio 'cd ~/eu-kiki && /Users/clems/KIKI-Mac_tunner/.venv/bin/python -m pytest tests/test_eval_framework.py::test_memory_limit_below_wired_cap -v'
```

Expected: **PASS**.

- [ ] **Step 4: Commit**

```bash
ssh studio 'cd ~/eu-kiki && git add scripts/eval_framework.py && GIT_EDITOR=true git -c user.name=electron-rare -c user.email=108685187+electron-rare@users.noreply.github.com commit -m "fix(eval): lower mx.set_memory_limit 480->440 GiB

Studio iogpu.wired_limit_mb is 458752 (448 GiB). Setting the MLX soft
cap above that hard cap was the SIGKILL trigger on full --mode compare
runs. 440 GiB leaves 8 GiB headroom for the kernel + non-Metal
allocations and stays well below the wired pool ceiling."'
```

---

## Task 3: Add `_assert_within_budget()` peak-memory probe

**Files:**
- Modify: `~/eu-kiki/scripts/eval_framework.py:473-485` (extend `unload_model()` and add a sibling helper)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_framework.py`:

```python
def test_assert_within_budget_raises_on_overrun(monkeypatch):
    """The probe must raise a clean RuntimeError instead of letting the
    kernel SIGKILL us when peak memory crosses the configured ceiling."""
    from eval_framework import _assert_within_budget  # noqa: WPS433

    class _FakeMx:
        @staticmethod
        def get_peak_memory():
            return 460 * 1024 ** 3  # 460 GiB peak — over budget

    monkeypatch.setattr("eval_framework.mx", _FakeMx, raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="peak memory .* exceeds budget"):
        _assert_within_budget(budget_gib=440)


def test_assert_within_budget_passes_when_under(monkeypatch):
    from eval_framework import _assert_within_budget

    class _FakeMx:
        @staticmethod
        def get_peak_memory():
            return 200 * 1024 ** 3

    monkeypatch.setattr("eval_framework.mx", _FakeMx, raising=False)
    _assert_within_budget(budget_gib=440)  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
ssh studio 'cd ~/eu-kiki && /Users/clems/KIKI-Mac_tunner/.venv/bin/python -m pytest tests/test_eval_framework.py -v'
```

Expected: 2 new FAILs (`ImportError: cannot import name '_assert_within_budget'`).

- [ ] **Step 3: Add the helper + module-level `mx` import**

In `~/eu-kiki/scripts/eval_framework.py`, add near the top (after the existing `import` block, before line 50):

```python
import mlx.core as mx  # noqa: E402  module-level handle for monkey-patchable probe
```

And add the helper next to `unload_model()` (after line 485):

```python
def _assert_within_budget(budget_gib: int = 440) -> None:
    """Abort cleanly with RuntimeError if Metal peak memory has exceeded the
    configured budget. Called between every model transition in
    sequential-strict mode so an overrun produces a structured error
    instead of a kernel SIGKILL.
    """
    peak_b = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()
    peak_gib = peak_b / (1024 ** 3)
    if peak_gib > budget_gib:
        raise RuntimeError(
            f"peak memory {peak_gib:.1f} GiB exceeds budget {budget_gib} GiB"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh studio 'cd ~/eu-kiki && /Users/clems/KIKI-Mac_tunner/.venv/bin/python -m pytest tests/test_eval_framework.py -v'
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
ssh studio 'cd ~/eu-kiki && git add scripts/eval_framework.py tests/test_eval_framework.py && GIT_EDITOR=true git -c user.name=electron-rare -c user.email=108685187+electron-rare@users.noreply.github.com commit -m "feat(eval): _assert_within_budget probe with clean RuntimeError"'
```

---

## Task 4: Add `--mode sequential-strict` CLI flag

**Files:**
- Modify: `~/eu-kiki/scripts/eval_framework.py` — `parse_args()` (find with `grep -n "argparse\|add_argument\|--mode" ~/eu-kiki/scripts/eval_framework.py`; the parser is in `main()` around line 1277), the `run_eval()` signature, and the Phase 2/3 loop iteration order

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_framework.py`:

```python
def test_mode_sequential_strict_in_choices():
    """--mode must accept 'sequential-strict' alongside the legacy modes."""
    src = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "eval_framework.py"
    ).read_text()
    # The argparse choices list (or any equivalent) must mention the new mode.
    assert "sequential-strict" in src, (
        "Add 'sequential-strict' to the --mode choices and route it through"
        " run_eval()."
    )


def test_load_group_order_strict_groups_by_model():
    """In sequential-strict mode, all domains for one base model are
    consumed before the next base model is touched. We verify the
    iteration order helper directly."""
    from eval_framework import _strict_iteration_order

    load_groups = {
        ("v1", "apertus"): ["math", "spice-sim"],
        ("v1", "devstral"): ["python", "rust"],
        ("v2", "qwen36"): ["python"],
    }
    out = list(_strict_iteration_order(load_groups))
    # Same (version, model_key) keys come in a contiguous run.
    keys = [(v, m) for (v, m), _ in out]
    seen: set = set()
    for k in keys:
        if seen and list(seen)[-1] != k:
            assert k not in seen, (
                f"key {k} reappeared after another model was started — "
                f"iteration order is not strict-sequential: {keys}"
            )
        seen.add(k)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
ssh studio 'cd ~/eu-kiki && /Users/clems/KIKI-Mac_tunner/.venv/bin/python -m pytest tests/test_eval_framework.py -v'
```

Expected: 2 new FAILs.

- [ ] **Step 3: Add the iteration helper near `run_eval()`**

In `~/eu-kiki/scripts/eval_framework.py` insert just above the `def run_eval(` definition (around line 1146):

```python
def _strict_iteration_order(
    load_groups: dict[tuple[str, str], list[str]],
) -> list[tuple[tuple[str, str], list[str]]]:
    """Return (version, model_key) -> [domains] entries in an order that
    consumes every adapter for one base model before any adapter of another
    base model. Stable on dict insertion order so identical inputs always
    produce identical sequences (reproducibility for the bench history)."""
    return list(load_groups.items())
```

- [ ] **Step 4: Wire `sequential-strict` into the parser + dispatch**

Find the argparse parser in `main()` (the `--mode` `add_argument` call) and add `"sequential-strict"` to its `choices=` list. Then in `run_eval()`:
- Treat `mode == "sequential-strict"` like `compare` for which versions are evaluated.
- After the `for (version, model_key), domains in load_groups.items():` loops in Phases 2 and 3, **call `unload_model()` then `_assert_within_budget()`** between each model. This is already true in Phase 2 because `eval_perplexity_for_adapter` ends with `unload_model()`, but Phase 3 (`eval_generation_for_adapter`) needs the same epilogue — add an explicit call:

In `run_eval()`, replace the Phase 2 `load_groups.items()` iterator with `_strict_iteration_order(load_groups)`:

```python
    for (version, model_key), domains in _strict_iteration_order(load_groups):
        model_info = MODELS[model_key]
        print(f"\n  Loading {model_key} ({model_info['short']}) for {len(domains)} domains...")
        for domain in domains:
            ppl_result = eval_perplexity_for_adapter(
                model_key, domain, version, max_samples=max_ppl_samples
            )
            if ppl_result:
                results.perplexity.append(ppl_result)
                print(
                    f"    {domain}: loss={ppl_result.val_loss:.4f}, "
                    f"ppl={ppl_result.perplexity:.2f}"
                )
        if mode == "sequential-strict":
            unload_model()
            _assert_within_budget(budget_gib=440)
```

Apply the same `_strict_iteration_order` + post-group `unload_model()` + `_assert_within_budget()` to the Phase 3 generation loop.

- [ ] **Step 5: Run tests to verify they pass**

```bash
ssh studio 'cd ~/eu-kiki && /Users/clems/KIKI-Mac_tunner/.venv/bin/python -m pytest tests/test_eval_framework.py -v'
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
ssh studio 'cd ~/eu-kiki && git add scripts/eval_framework.py tests/test_eval_framework.py && GIT_EDITOR=true git -c user.name=electron-rare -c user.email=108685187+electron-rare@users.noreply.github.com commit -m "feat(eval): --mode sequential-strict + per-model budget probe"'
```

---

## Task 5: Forward `--mode sequential-strict` through `run_eval.sh`

**Files:**
- Modify: `~/eu-kiki/scripts/run_eval.sh` (the `case "$1" in` block that builds `MODE` for the Python invocation)

- [ ] **Step 1: Find the dispatch block**

```bash
ssh studio 'grep -n "v1-only\|v2-only\|--compare\|MODE=" ~/eu-kiki/scripts/run_eval.sh | head -20'
```

The block accepts `--v1-only`, `--v2-only`, `--compare`. Add a `--sequential-strict` branch.

- [ ] **Step 2: Patch the bash dispatcher**

Locate the `--compare` case and add the sibling case immediately after:

```bash
ssh studio 'awk "/^[[:space:]]*--compare\)/,/^[[:space:]]*;;/" ~/eu-kiki/scripts/run_eval.sh'
```

Then patch by inserting after the `--compare;;` block:

```bash
ssh studio 'sed -i "" "s|--compare)|--sequential-strict) MODE=sequential-strict; shift;;\n        --compare)|" ~/eu-kiki/scripts/run_eval.sh && grep -A1 "sequential-strict)" ~/eu-kiki/scripts/run_eval.sh'
```

Expected: prints the new case.

- [ ] **Step 3: Update the help text**

```bash
ssh studio 'sed -i "" "s|--compare          Evaluate both and compare (default)|--compare          Evaluate both and compare (default)\n  --sequential-strict  Same as --compare but unloads + budget-probes between models|" ~/eu-kiki/scripts/run_eval.sh'
```

- [ ] **Step 4: Smoke-test the help**

```bash
ssh studio 'bash ~/eu-kiki/scripts/run_eval.sh --help' | grep -i sequential
```

Expected: prints the new option line.

- [ ] **Step 5: Commit**

```bash
ssh studio 'cd ~/eu-kiki && git add scripts/run_eval.sh && GIT_EDITOR=true git -c user.name=electron-rare -c user.email=108685187+electron-rare@users.noreply.github.com commit -m "feat(run_eval): --sequential-strict forwarder"'
```

---

## Task 6: Wrapper `launch_eval_safe.sh` with structured exit reporting

**Files:**
- Create: `~/eu-kiki/scripts/launch_eval_safe.sh`

- [ ] **Step 1: Write the wrapper script**

```bash
ssh studio 'cat > ~/eu-kiki/scripts/launch_eval_safe.sh << "EOF"
#!/usr/bin/env bash
# Wraps run_eval.sh and emits output/eval/last_run_status.json so downstream
# automation can react without parsing free-form logs.
set -euo pipefail
EU_KIKI="$HOME/eu-kiki"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$EU_KIKI/output/eval/launch-${STAMP}.log"
STATUS_FILE="$EU_KIKI/output/eval/last_run_status.json"
mkdir -p "$EU_KIKI/output/eval"

START=$(date -u +%s)
set +e
bash "$EU_KIKI/scripts/run_eval.sh" "$@" 2>&1 | tee "$LOG"
EC=${PIPESTATUS[0]}
set -e
END=$(date -u +%s)
WALL=$((END - START))

# 137 = SIGKILL (128 + 9), 139 = SIGSEGV (128 + 11)
SIGNAL=""
if [[ $EC -gt 128 && $EC -lt 160 ]]; then
    SIGNAL=$((EC - 128))
fi

cat > "$STATUS_FILE" <<JSON
{
  "stamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "exit_code": $EC,
  "signal": ${SIGNAL:-null},
  "wall_seconds": $WALL,
  "log_path": "$LOG",
  "args": "$*"
}
JSON
echo "Status: $STATUS_FILE (exit=$EC, signal=${SIGNAL:-none}, wall=${WALL}s)"
exit $EC
EOF
chmod +x ~/eu-kiki/scripts/launch_eval_safe.sh
ls -lah ~/eu-kiki/scripts/launch_eval_safe.sh'
```

- [ ] **Step 2: Smoke test the wrapper with `--quick`**

```bash
ssh studio 'bash ~/eu-kiki/scripts/launch_eval_safe.sh --quick --v1-only --domains math-gsm8k 2>&1 | tail -5'
```

Expected: completes within ~1 minute, exit 0, status file says `"exit_code": 0`.

- [ ] **Step 3: Verify the status JSON**

```bash
ssh studio 'cat ~/eu-kiki/output/eval/last_run_status.json'
```

Expected: valid JSON with `exit_code: 0`, `signal: null`, non-zero `wall_seconds`.

- [ ] **Step 4: Commit**

```bash
ssh studio 'cd ~/eu-kiki && git add scripts/launch_eval_safe.sh && GIT_EDITOR=true git -c user.name=electron-rare -c user.email=108685187+electron-rare@users.noreply.github.com commit -m "feat(eval): launch_eval_safe.sh with last_run_status.json"'
```

---

## Task 7: End-to-end validation — `--sequential-strict --quick`

**Files:** none modified (validation only)

- [ ] **Step 1: Stop the EuroLLM worker so Studio :9303 is free**

```bash
ssh studio 'PID=$(lsof -tiTCP:9303 -sTCP:LISTEN 2>/dev/null); [ -n "$PID" ] && kill -TERM "$PID"; sleep 3; lsof -tiTCP:9303 -sTCP:LISTEN >/dev/null 2>&1 && echo STILL_UP || echo DOWN'
```

Expected: `DOWN`.

- [ ] **Step 2: Run quick sequential-strict end-to-end**

```bash
ssh studio 'bash ~/eu-kiki/scripts/launch_eval_safe.sh --sequential-strict --quick --domains math-gsm8k python chat-fr 2>&1 | tail -25'
```

Expected: completes in under 5 minutes, prints `EVAL COMPLETE`, status JSON has `exit_code: 0`.

- [ ] **Step 3: Verify raw outputs were written**

```bash
ssh studio 'ls -lah ~/eu-kiki/output/eval/raw/perplexity_*$(date +%Y%m%d)*.json | tail -3'
```

Expected: at least one new `perplexity_*.json` file from this run.

- [ ] **Step 4: Restart EuroLLM worker (production restore)**

```bash
ssh studio 'mkdir -p $HOME/eu-kiki/logs && cd $HOME/eu-kiki && nohup $HOME/eu-kiki/.venv/bin/python -m uvicorn src.worker.server:make_eurollm_app --factory --host 0.0.0.0 --port 9303 > $HOME/eu-kiki/logs/eurollm.log 2>&1 & echo "EuroLLM PID $!"'
```

- [ ] **Step 5: Confirm worker is bound on 0.0.0.0**

```bash
ssh studio 'sleep 8 && lsof -nP -iTCP:9303 -sTCP:LISTEN | head -2'
```

Expected: `*:9303 (LISTEN)`.

---

## Task 8: Full sequential-strict run (the actual matrix)

**Files:** none modified.

This is the production validation. Plan a 3-4 h window where Studio :9303 EuroLLM can stay offline.

- [ ] **Step 1: Stop EuroLLM**

```bash
ssh studio 'PID=$(lsof -tiTCP:9303 -sTCP:LISTEN 2>/dev/null); [ -n "$PID" ] && kill -TERM "$PID"; sleep 3'
```

- [ ] **Step 2: Launch the full eval in the background**

```bash
ssh studio 'nohup bash ~/eu-kiki/scripts/launch_eval_safe.sh --sequential-strict 2>&1 > ~/eu-kiki/output/eval/full-sequential-strict.log & echo "PID $!"'
```

- [ ] **Step 3: Poll until the launcher writes its status file**

Use Bash with `run_in_background: true` and an `until` loop:

```bash
ssh studio 'until [ -f ~/eu-kiki/output/eval/last_run_status.json ] && [ "$(stat -f %m ~/eu-kiki/output/eval/last_run_status.json)" -gt $(($(date -u +%s) - 14400)) ]; do sleep 120; done; cat ~/eu-kiki/output/eval/last_run_status.json'
```

Expected eventual: `"exit_code": 0`. If `signal: 9` reappears, increment Task 3 budget down (e.g. 420 GiB) or lower per-domain `max_ppl_samples`.

- [ ] **Step 4: Inventory raw output**

```bash
ssh studio 'ls ~/eu-kiki/output/eval/raw/ | wc -l; ls ~/eu-kiki/output/eval/eval_report_v1_vs_v2.md'
```

Expected: 4+ JSON files (one per phase per version), and the markdown report exists.

- [ ] **Step 5: Restart EuroLLM (production restore)**

```bash
ssh studio 'cd $HOME/eu-kiki && nohup $HOME/eu-kiki/.venv/bin/python -m uvicorn src.worker.server:make_eurollm_app --factory --host 0.0.0.0 --port 9303 > $HOME/eu-kiki/logs/eurollm.log 2>&1 & sleep 6; lsof -tiTCP:9303 -sTCP:LISTEN'
```

---

## Task 9: Document in CLAUDE.md + push everything

**Files:**
- Modify: `~/eu-kiki/docs/CLAUDE.md` (Roadmap → Bloquants → mark item 3 (122B-Opus) and add the new sequential-strict runbook)
- Create: `~/eu-kiki/docs/superpowers/plans/2026-05-10-eval-framework-oom-fix.md` (this plan, persisted)

- [ ] **Step 1: Save this plan into the repo**

```bash
ssh studio 'mkdir -p ~/eu-kiki/docs/superpowers/plans && cp /tmp/2026-05-10-eval-framework-oom-fix.md ~/eu-kiki/docs/superpowers/plans/'
```

(The plan file should already be on Studio at `/tmp/2026-05-10-eval-framework-oom-fix.md` from the previous `scp` step in this session.)

- [ ] **Step 2: Append a runbook section to docs/CLAUDE.md**

```bash
ssh studio 'cat >> ~/eu-kiki/docs/CLAUDE.md <<EOF

## EU-KIKI eval — sequential-strict runbook (added 2026-05-10)

Use \`--mode sequential-strict\` whenever the bench has to load multiple
base models in one run. The launcher \`scripts/launch_eval_safe.sh\` writes
\`output/eval/last_run_status.json\` so background pollers can detect
SIGKILL (\`signal: 9\`) without parsing logs.

\`\`\`bash
# Stop EuroLLM worker first so :9303 is free.
kill -TERM \$(lsof -tiTCP:9303 -sTCP:LISTEN)
bash ~/eu-kiki/scripts/launch_eval_safe.sh --sequential-strict
# Restart EuroLLM after.
cd ~/eu-kiki && nohup .venv/bin/python -m uvicorn src.worker.server:make_eurollm_app --factory --host 0.0.0.0 --port 9303 > logs/eurollm.log 2>&1 &
\`\`\`

Memory budget: \`mx.set_memory_limit(440 * 1024**3)\` stays 8 GiB under the
\`iogpu.wired_limit_mb=458752\` (= 448 GiB) macOS hard cap. Going above
caused the kernel to SIGKILL the eval at import time on 2026-05-10.
EOF'
```

- [ ] **Step 3: Commit + push everything**

```bash
ssh studio 'cd ~/eu-kiki && git add docs/CLAUDE.md docs/superpowers/plans/2026-05-10-eval-framework-oom-fix.md && GIT_EDITOR=true git -c user.name=electron-rare -c user.email=108685187+electron-rare@users.noreply.github.com commit -m "docs: sequential-strict runbook + plan" && git push 2>&1 | tail -3'
```

Expected: push succeeds against `https://github.com/L-electron-Rare/ailiance.git` (the GitHub-renamed remote of `eu-kiki`).

- [ ] **Step 4: Update memory note on grosmac**

Append to `/Users/electron/.claude/projects/-Users-electron/memory/reference_eu_aiact_bench_battle_2026_05_10.md` Phase 3 section:

```markdown
## Phase 3 OOM root cause + fix (2026-05-10 18:30)

`mx.set_memory_limit(480 GiB)` exceeded the `iogpu.wired_limit_mb=458752`
(= 448 GiB) macOS hard cap. Kernel SIGKILLed at import. Fixed in plan
`docs/superpowers/plans/2026-05-10-eval-framework-oom-fix.md` (commit
TODO after Task 9 lands) by lowering to 440 GiB + adding
`_assert_within_budget()` probe + `--mode sequential-strict` that
unloads between base models. Use `scripts/launch_eval_safe.sh` to get
structured exit-code/signal reporting.
```

---

## Out of scope (separate plans)

- **Lacune #1** (compléter `31_domains_baseline.json` 4 modèles restants) — operational rerun, plan template `runbook` not `writing-plans`. Open a separate task.
- **Lacune #3** (Studio :9301 launchd Apertus + Mistral 128B Q8 multi-port) — infra-as-code, separate plan.
- **Lacune #4** (Devstral base PPL 355k) — investigation, brainstorming-first then a plan.
- **Issue #10** EuroLLM empty content — has its own diagnostic comment; needs a small follow-up plan once chat-fr/traduction-tech adapters are re-trained.
