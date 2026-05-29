"""Tests for the NDJSON audit-log writer (EU AI Act Annex IV §7)."""

from __future__ import annotations

import json
import multiprocessing
from datetime import datetime, timezone
from pathlib import Path

from scripts.kicad_sch.audit_log import AuditLogger, sha256_manifest, verify


def test_audit_logger_appends_ndjson(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    logger = AuditLogger(log_path)
    logger.log("generation", model_id="apertus", prompt_hash="abc123", seed=42)
    logger.log("eval", validator="kicad-erc", exit_code=0, axis_scores={"parse_ok": 1})
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    assert e1["event_type"] == "generation"
    assert e1["seed"] == 42
    assert e1["model_id"] == "apertus"
    e2 = json.loads(lines[1])
    assert e2["event_type"] == "eval"
    assert e2["axis_scores"] == {"parse_ok": 1}


def test_audit_logger_appends_across_instances(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    AuditLogger(log_path).log("a", x=1)
    AuditLogger(log_path).log("b", y=2)
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "a"
    assert json.loads(lines[1])["event_type"] == "b"


def test_sha256_manifest_deterministic(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    log_path.write_text('{"a": 1}\n{"b": 2}\n')
    h1 = sha256_manifest(log_path)
    h2 = sha256_manifest(log_path)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_verify_detects_tampering(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    log_path.write_text('{"a": 1}\n')
    sha = sha256_manifest(log_path)
    log_path.write_text('{"a": 2}\n')
    assert verify(log_path, sha) is False


def test_verify_passes_untampered(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    log_path.write_text('{"a": 1}\n')
    sha = sha256_manifest(log_path)
    assert verify(log_path, sha) is True


def test_audit_logger_auto_injects_timestamp(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    logger = AuditLogger(log_path)
    logger.log("generation", model_id="apertus")
    record = json.loads(log_path.read_text().strip())
    assert "timestamp" in record
    # ISO 8601 UTC: YYYY-MM-DDTHH:MM:SS.microsecond+00:00
    assert record["timestamp"].endswith("+00:00") or record["timestamp"].endswith("Z")


def test_audit_logger_respects_explicit_timestamp(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    logger = AuditLogger(log_path)
    logger.log("generation", timestamp="2026-05-11T05:00:00+00:00", model_id="x")
    record = json.loads(log_path.read_text().strip())
    assert record["timestamp"] == "2026-05-11T05:00:00+00:00"


def test_audit_logger_serializes_path_and_datetime(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    logger = AuditLogger(log_path)
    logger.log("eval", adapter_path=Path("/tmp/adapter"), generated_at=datetime(2026, 5, 11, 5, 0, 0, tzinfo=timezone.utc))
    record = json.loads(log_path.read_text().strip())
    # default=str converts Path → str and datetime → str
    assert record["adapter_path"] == "/tmp/adapter"
    assert "2026-05-11" in record["generated_at"]


# ---------------------------------------------------------------------------
# #34 — New tests: unicode roundtrip, large payload, empty-file hash,
#         and concurrent-append correctness (proves #33 flock hardening).
# ---------------------------------------------------------------------------


def test_unicode_roundtrip(tmp_path: Path) -> None:
    """#34-1: ensure_ascii=False must preserve accented + emoji values."""
    log_path = tmp_path / "audit.ndjson"
    logger = AuditLogger(log_path)
    note = "café 🚀 naïve"
    logger.log("probe", note=note)
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["note"] == note, f"Expected {note!r}, got {record['note']!r}"


def test_large_payload_roundtrip(tmp_path: Path) -> None:
    """#34-2: a field >PIPE_BUF (~4 KB) must round-trip intact."""
    log_path = tmp_path / "audit.ndjson"
    logger = AuditLogger(log_path)
    big_value = "x" * 8192  # 8 KiB > PIPE_BUF (4096 bytes)
    logger.log("bench", axis_scores=big_value)
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["axis_scores"] == big_value
    assert len(record["axis_scores"]) == 8192


def test_sha256_manifest_empty_file(tmp_path: Path) -> None:
    """#34-3: sha256 of zero bytes is well-known and deterministic."""
    log_path = tmp_path / "empty.ndjson"
    log_path.write_bytes(b"")
    digest = sha256_manifest(log_path)
    assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _worker_write(log_path_str: str, worker_id: int, n_records: int) -> None:
    """Child-process writer used by the concurrency test."""
    logger = AuditLogger(Path(log_path_str))
    big_field = "y" * 5000  # >PIPE_BUF to stress interleaving
    for i in range(n_records):
        logger.log("concurrent", worker_id=worker_id, seq=i, payload=big_field)


def test_concurrent_no_torn_lines(tmp_path: Path) -> None:
    """#34-4/#33: 8 processes × 10 records → 80 intact JSON lines.

    Uses multiprocessing.Process for true cross-process flock semantics.
    Each record carries a >4 KB payload to trigger the PIPE_BUF boundary
    that flock is meant to guard against.
    """
    log_path = tmp_path / "concurrent.ndjson"
    n_workers = 8
    n_records = 10

    procs = [
        multiprocessing.Process(
            target=_worker_write,
            args=(str(log_path), wid, n_records),
        )
        for wid in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"Worker process exited with {p.exitcode}"

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_workers * n_records, (
        f"Expected {n_workers * n_records} lines, got {len(lines)}"
    )
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Line {i} is not valid JSON: {exc}\nContent: {line[:120]!r}") from exc
