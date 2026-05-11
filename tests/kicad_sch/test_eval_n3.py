"""Tests for kicad_sch.eval_n3 (5-axis evaluator)."""
from pathlib import Path

from scripts.kicad_sch.eval_n3 import eval_parse_ok  # noqa: F401


def test_module_imports():
    """eval_n3 module must be importable from scripts.kicad_sch package."""
    from scripts.kicad_sch import eval_n3
    assert hasattr(eval_n3, "eval_parse_ok")
