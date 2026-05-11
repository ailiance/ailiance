"""Tests for the NDJSON audit-log writer (EU AI Act Annex IV §7)."""

from __future__ import annotations

import json
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
