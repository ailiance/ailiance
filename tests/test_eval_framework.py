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
    """The argparse parser's --mode choices list must include
    'sequential-strict'. A substring scan over the whole file would also
    match the constant's docstring, so we anchor on the choices=[...]
    literal next to add_argument('--mode', ...)."""
    src = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "eval_framework.py"
    ).read_text()
    import re
    # Find the --mode argument's choices list. Tolerate single or double
    # quotes around each value and arbitrary whitespace within the list.
    m = re.search(
        r'add_argument\(\s*'
        + r"(?:'|\")"
        + r'--mode'
        + r"(?:'|\")"
        + r'[^)]*choices\s*=\s*\[([^\]]+)\]',
        src, re.DOTALL,
    )
    assert m, "Could not locate add_argument('--mode', ..., choices=[...]) literal"
    choices_text = m.group(1)
    assert '"sequential-strict"' in choices_text or "'sequential-strict'" in choices_text, (
        f"--mode choices list missing 'sequential-strict': {choices_text!r}"
    )


def test_strict_iteration_order_preserves_insertion_order():
    """Identical inputs must produce identical sequences. The helper is a
    thin wrapper around list(load_groups.items()) — the dict's insertion
    order IS the iteration contract."""
    from eval_framework import _strict_iteration_order

    load_groups = {
        ("v1", "apertus"): ["math", "spice-sim"],
        ("v1", "devstral"): ["python", "rust"],
        ("v2", "qwen36"): ["python"],
    }
    out = _strict_iteration_order(load_groups)
    assert out == list(load_groups.items()), (
        f"Helper must preserve dict.items() order, got {out}"
    )


def test_strict_iteration_order_groups_each_model_contiguously():
    """A (version, model_key) must not reappear after a different one was
    started. Uses an ordered list (not a set) for previous-key tracking so
    the assertion logic is deterministic on CPython where set iteration is
    insertion-order-independent."""
    from eval_framework import _strict_iteration_order

    load_groups = {
        ("v1", "apertus"): ["math"],
        ("v1", "devstral"): ["python"],
        ("v2", "qwen36"): ["math"],
    }
    out = _strict_iteration_order(load_groups)
    seen_keys: list = []
    for (version, model_key), _ in out:
        key = (version, model_key)
        if seen_keys and seen_keys[-1] != key:
            assert key not in seen_keys, (
                f"key {key} reappeared after another model was started: "
                f"{[k for k, _ in out]}"
            )
        seen_keys.append(key)


def test_no_ailiance_path_constants_in_scripts():
    """Sibling scripts must not write to ~/ailiance/. The disk root is
    ~/eu-kiki/ — the GitHub repo is named 'ailiance' but the on-disk
    path stayed eu-kiki to avoid breaking sibling tooling. This
    regression test was added 2026-05-10 after multiple cycles of
    the same path bug being introduced and reverted.
    """
    import re
    from pathlib import Path
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    violations = []
    pat = re.compile(
        r"(AILIANCE=\"\$HOME/ailiance\"|\$HOME/ailiance|~/ailiance|Path\.home\(\)\s*/\s*\"ailiance\")"
    )
    for p in scripts_dir.rglob("*"):
        if not p.is_file() or p.suffix not in (".sh", ".py"):
            continue
        text = p.read_text()
        for ln, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                violations.append(f"{p.name}:{ln}: {line.strip()[:80]}")
    assert not violations, (
        "Path constants pointing at ~/ailiance/ found:\n  "
        + "\n  ".join(violations)
    )
