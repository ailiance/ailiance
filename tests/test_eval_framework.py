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
    m = re.search(r"mx\.set_memory_limit\(([A-Za-z_][A-Za-z0-9_]*|\d+)\s*\*\s*1024\*\*3\)", src)
    assert m, "mx.set_memory_limit(<gib> * 1024**3) call not found"
    token = m.group(1)
    if token.isdigit():
        return int(token)
    # Resolve module-level constant by name (e.g. WIRED_MEMORY_BUDGET_GIB = 440).
    cm = re.search(rf"^{re.escape(token)}\s*=\s*(\d+)\s*$", src, re.MULTILINE)
    assert cm, f"constant {token} not found at module level in eval_framework.py"
    return int(cm.group(1))


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


def test_mode_sequential_strict_in_choices():
    """--mode must accept 'sequential-strict' alongside the legacy modes."""
    src = (
        Path(__file__).resolve().parent.parent
        / 'scripts'
        / 'eval_framework.py'
    ).read_text()
    # The argparse choices list (or any equivalent) must mention the new mode.
    assert 'sequential-strict' in src, (
        "Add 'sequential-strict' to the --mode choices and route it through"
        ' run_eval().'
    )


def test_load_group_order_strict_groups_by_model():
    """In sequential-strict mode, all domains for one base model are
    consumed before the next base model is touched. We verify the
    iteration order helper directly."""
    from eval_framework import _strict_iteration_order

    load_groups = {
        ('v1', 'apertus'): ['math', 'spice-sim'],
        ('v1', 'devstral'): ['python', 'rust'],
        ('v2', 'qwen36'): ['python'],
    }
    out = list(_strict_iteration_order(load_groups))
    # Same (version, model_key) keys come in a contiguous run.
    keys = [(v, m) for (v, m), _ in out]
    seen: set = set()
    for k in keys:
        if seen and list(seen)[-1] != k:
            assert k not in seen, (
                f'key {k} reappeared after another model was started — '
                f'iteration order is not strict-sequential: {keys}'
            )
        seen.add(k)
