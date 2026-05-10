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
