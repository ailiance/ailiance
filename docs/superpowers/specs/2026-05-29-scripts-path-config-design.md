# Scripts path config — design (Fil B: #118)

**Date:** 2026-05-29
**Issue:** #118 (remove hardcoded `$HOME/ailiance` path literals from scripts)
**Repo:** `ailiance/ailiance` (gateway)
**Status:** design drafted during brainstorming; **awaiting user review/approval** before plan.

## Problem

~28 scripts under `scripts/` hardcode `AILIANCE="$HOME/ailiance"` (and `KIKI_TUNNER="$HOME/ailiance-mac-tuner"`, a sibling repo). The regression test `tests/test_eval_framework.py::test_no_ailiance_path_constants_in_scripts` forbids any `$HOME/ailiance` / `~/ailiance` / `Path.home()/"ailiance"` literal; it is currently `xfail` (set during CI #26 work). The literal `$HOME/ailiance` breaks on machines/users where the checkout lives elsewhere.

## Decision (from brainstorming)

Intent confirmed: `$HOME/ailiance` IS the correct deploy path (it equals the repo root — scripts live in `<repo>/scripts/`), but hardcoding it is non-portable. **Make the path configurable** (derive from script location, with `$AILIANCE_HOME` override), then re-enable the test. NOT retiring the test; NOT a deeper rewrite.

Note: `$HOME/ailiance-mac-tuner` also matches the test regex (prefix `$HOME/ailiance`), so `KIKI_TUNNER` must be de-literal'd too. It is a sibling of the repo root.

## Design

### Shell scripts (`train_batch*.sh`, `run_eval*.sh`, `launch_eval_safe.sh`, `phase3_launcher.sh`, `train_vlm_poc.sh`, etc.)
Replace:
```sh
AILIANCE="$HOME/ailiance"
KIKI_TUNNER="$HOME/ailiance-mac-tuner"
```
with:
```sh
AILIANCE="${AILIANCE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KIKI_TUNNER="${KIKI_TUNNER_HOME:-$(dirname "$AILIANCE")/ailiance-mac-tuner}"
```
- `AILIANCE` = `$AILIANCE_HOME` override, else the repo root derived from the script's own location (`scripts/..`). No `$HOME/ailiance` literal.
- `KIKI_TUNNER` = `$KIKI_TUNNER_HOME` override, else sibling of the repo root (`dirname $AILIANCE` = `$HOME` on the deploy host, then `/ailiance-mac-tuner`). Contains `ailiance-mac-tuner` but not `$HOME/ailiance` → passes the regex.
- For scripts invoked via `sh` (not bash) where `BASH_SOURCE` is unset, use `$0` instead: `$(cd "$(dirname "$0")/.." && pwd)`. Check each script's shebang; prefer `${BASH_SOURCE[0]:-$0}` for safety.

### Python scripts (`scripts/bench_base.py`, `scripts/bench_comparison.py`, `scripts/eval_framework.py`)
Replace:
```python
AILIANCE = Path.home() / "ailiance"
```
with:
```python
import os
AILIANCE = Path(os.environ.get("AILIANCE_HOME", Path(__file__).resolve().parent.parent))
```
- Default = repo root (parent of `scripts/`), `$AILIANCE_HOME` override. No `Path.home()/"ailiance"` literal.

### Re-enable the test
- Remove the `@pytest.mark.xfail(...)` from `test_no_ailiance_path_constants_in_scripts` (added during CI #26).
- After scrubbing all literals, the test passes (0 violations). It then guards against regressions on the main CI `test` check.

## Scope / risk

- ~28 files edited (mechanical, ~2 lines each).
- **Behaviorally equivalent on the deploy host**: script-location → `$HOME/ailiance`; same effective paths. Pure portability change, no functional difference.
- Assumption: scripts run from within the repo checkout (true on the deploy host). `$AILIANCE_HOME` covers the override case.
- Edge: scripts run via `sh` (not bash) — use `${BASH_SOURCE[0]:-$0}` to stay portable.

## Testing

- The de-xfail'd `test_no_ailiance_path_constants_in_scripts` is the primary validation: 0 violations.
- Smoke: `bash -n <script>` on each edited shell script (syntax) + `python -c "import ast; ast.parse(open(p).read())"` (or import) on the 3 `.py` scripts.
- Full CI command (`pytest -m "not network and not e2e"`) green before PR.

## Out of scope

- No change to what the scripts DO (training/eval logic untouched).
- Not retiring the test (the policy is kept, made satisfiable).
- #115/#116 (Fil A) — already merged via PR #121.

## Workflow note

Lands via PR from `feat/scripts-path-config` (main is branch-protected, requires the `test` CI check).
