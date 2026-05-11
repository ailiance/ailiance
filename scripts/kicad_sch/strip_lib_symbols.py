"""Strip ``(lib_symbols ...)`` block from a ``.kicad_sch`` file.

Reduces context size from 5-50 KB down to 2-5 KB; ``lib_id`` references
inside ``(symbol ...)`` placements are resolved by ``kicad-cli`` at
schematic-load time so dropping the inline library is lossless for
v10 schematics.

Returns ``0`` on success, nonzero on parse failure (unbalanced
parentheses). Idempotent when no ``(lib_symbols`` block is present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_top_block(text: str, head: str) -> tuple[int, int] | None:
    """Find the first balanced s-expression starting with ``head``.

    Returns ``(start, end_exclusive)`` byte offsets, or ``None`` when
    the marker is absent.
    """
    i = text.find(head)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return (i, j + 1)
    return None


def strip_lib_symbols(src: Path, out: Path) -> int:
    """Strip ``(lib_symbols ...)`` from ``src`` and write to ``out``.

    Returns ``0`` on success, ``2`` on unbalanced-paren parse failure.
    """
    text = Path(src).read_text(encoding="utf-8")
    if text.count("(") != text.count(")"):
        return 2
    span = _find_top_block(text, "(lib_symbols")
    if span is None:
        Path(out).write_text(text, encoding="utf-8")
        return 0
    a, b = span
    new = text[:a] + "(lib_symbols)" + text[b:]
    Path(out).write_text(new, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.input.is_dir():
        a.output.mkdir(parents=True, exist_ok=True)
        rc_total = 0
        for f in a.input.glob("*.kicad_sch"):
            rc = strip_lib_symbols(f, a.output / f.name)
            if rc != 0:
                rc_total |= 1
        return rc_total
    return strip_lib_symbols(a.input, a.output)


if __name__ == "__main__":
    sys.exit(main())
