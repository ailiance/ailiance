"""Tests for :func:`src.gateway.server._normalize_response_body`.

Two worker quirks need smoothing:

1. MLX ``mlx_lm.server`` (>= 0.31.3) puts thinking-model output in
   ``message.reasoning`` and leaves ``message.content`` empty.
2. Ministral-3 Reasoning / R1 distills emit literal ``[THINK]…[/THINK]``
   or ``<think>…</think>`` tags inside ``content``.
"""

from __future__ import annotations

import pytest

from src.gateway.server import _BLOCKED_CHAT_ALIASES, _normalize_response_body


def _wrap(message: dict) -> dict:
    return {"choices": [{"index": 0, "message": message}]}


class TestReasoningPromotion:
    def test_empty_content_promotes_reasoning(self):
        body = _wrap({"role": "assistant", "content": "", "reasoning": "The answer is 4."})
        out = _normalize_response_body(body)
        assert out["choices"][0]["message"]["content"] == "The answer is 4."

    def test_missing_content_promotes_reasoning(self):
        body = _wrap({"role": "assistant", "reasoning": "OK."})
        out = _normalize_response_body(body)
        assert out["choices"][0]["message"]["content"] == "OK."

    def test_existing_content_is_preserved(self):
        body = _wrap({
            "role": "assistant",
            "content": "real answer",
            "reasoning": "hidden chain of thought",
        })
        out = _normalize_response_body(body)
        assert out["choices"][0]["message"]["content"] == "real answer"

    def test_blank_reasoning_does_not_overwrite(self):
        body = _wrap({"role": "assistant", "content": "", "reasoning": "   "})
        out = _normalize_response_body(body)
        assert out["choices"][0]["message"]["content"] == ""


class TestThinkTagStripping:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Complete block: chain-of-thought hidden, final answer kept.
            ("[THINK]ignore me[/THINK]The answer is 4.", "The answer is 4."),
            ("<think>plan</think>final", "final"),
            ("<THINK>upper</THINK>x", "x"),
            # No tags: passthrough.
            ("no tags here", "no tags here"),
            # Truncated stream (opening tag only): orphan tag stripped, body kept.
            ("[THINK]the user asked", "the user asked"),
            # Multiple blocks back to back.
            ("[THINK]a[/THINK]hello[THINK]b[/THINK]world", "helloworld"),
        ],
    )
    def test_strip_variants(self, raw: str, expected: str):
        body = _wrap({"role": "assistant", "content": raw})
        out = _normalize_response_body(body)
        assert out["choices"][0]["message"]["content"] == expected


class TestSafety:
    def test_non_dict_body_passthrough(self):
        assert _normalize_response_body("not a dict") == "not a dict"  # type: ignore[arg-type]

    def test_missing_choices_passthrough(self):
        body = {"id": "x", "object": "chat.completion"}
        assert _normalize_response_body(body) == body

    def test_choices_not_list_passthrough(self):
        body = {"choices": "broken"}
        assert _normalize_response_body(body) == body


class TestBlockedAliases:
    def test_embed_is_blocked(self):
        assert "ailiance-embed" in _BLOCKED_CHAT_ALIASES

    def test_chat_aliases_not_blocked(self):
        for alias in ("ailiance", "ailiance-qwen", "ailiance-gemma2", "ailiance-reasoning-r1"):
            assert alias not in _BLOCKED_CHAT_ALIASES
