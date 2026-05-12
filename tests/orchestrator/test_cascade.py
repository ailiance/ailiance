"""Unit tests for the v0.4 cascade complexity-based selection."""

from __future__ import annotations

import os

import pytest

from src.gateway.server import (
    _CASCADE_OVERRIDES,
    _cascade_pick,
    _complexity_estimate,
)


def test_complexity_simple_short_prompt() -> None:
    assert _complexity_estimate("hi") == "simple"
    assert _complexity_estimate("Quelle heure est-il ?") == "simple"


def test_complexity_complex_reasoning_marker() -> None:
    assert _complexity_estimate("Explain why the sky is blue") == "complex"
    assert _complexity_estimate("Pourquoi le ciel est bleu ?") == "complex"


def test_complexity_complex_long_prompt() -> None:
    long_prompt = "word " * 110
    assert _complexity_estimate(long_prompt) == "complex"


def test_complexity_medium_default() -> None:
    medium = "Write a small function that reads a file and prints its size in bytes."
    assert _complexity_estimate(medium) == "medium"


def test_cascade_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AILIANCE_CASCADE_ENABLED", raising=False)
    assert _cascade_pick("chat-fr", "hi") is None


def test_cascade_simple_chat_fr_picks_gemma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AILIANCE_CASCADE_ENABLED", "1")
    assert _cascade_pick("chat-fr", "Salut") == "ailiance-gemma"


def test_cascade_complex_chat_fr_picks_mistral_medium(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AILIANCE_CASCADE_ENABLED", "1")
    long = "Explain step by step why " + ("token " * 50)
    assert _cascade_pick("chat-fr", long) == "ailiance-mistral-medium"


def test_cascade_complex_math_reasoning_picks_r1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AILIANCE_CASCADE_ENABLED", "1")
    assert (
        _cascade_pick("math-reasoning", "Prove that sqrt(2) is irrational.")
        == "ailiance-reasoning-r1"
    )


def test_cascade_medium_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AILIANCE_CASCADE_ENABLED", "1")
    medium = "Write a short helper function in Python that counts words."
    assert _cascade_pick("python", medium) is None


def test_cascade_table_has_default_keys() -> None:
    assert "simple" in _CASCADE_OVERRIDES
    assert "complex" in _CASCADE_OVERRIDES
    assert "_default" in _CASCADE_OVERRIDES["simple"]
    assert "_default" in _CASCADE_OVERRIDES["complex"]
