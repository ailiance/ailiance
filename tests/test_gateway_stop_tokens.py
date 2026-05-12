"""Tests for :func:`_inject_stop_tokens`.

The Pixtral worker on Mac Studio :9325 leaks ``\\nUSER:`` past the end
of the assistant turn. The gateway injects a default stop list for
known-buggy aliases while preserving any caller-supplied values.
"""

from __future__ import annotations

from src.gateway.server import _STOP_TOKEN_DEFAULTS, _inject_stop_tokens


class TestInjectStopTokens:
    def test_unknown_alias_noop(self):
        body = {"messages": []}
        _inject_stop_tokens(body, "ailiance-qwen")
        assert "stop" not in body

    def test_pixtral_gets_defaults_when_no_user_stop(self):
        body = {"messages": []}
        _inject_stop_tokens(body, "ailiance-pixtral")
        assert body["stop"] == list(_STOP_TOKEN_DEFAULTS["ailiance-pixtral"])

    def test_user_string_stop_preserved_and_merged(self):
        body = {"stop": "###END"}
        _inject_stop_tokens(body, "ailiance-pixtral")
        # User value first, defaults appended.
        assert body["stop"][0] == "###END"
        for tok in _STOP_TOKEN_DEFAULTS["ailiance-pixtral"]:
            assert tok in body["stop"]

    def test_user_list_stop_preserved_no_dup(self):
        body = {"stop": ["\nUSER:", "###END"]}  # one overlaps with defaults
        _inject_stop_tokens(body, "ailiance-pixtral")
        # \nUSER: present exactly once.
        assert body["stop"].count("\nUSER:") == 1
        assert "###END" in body["stop"]
        # All defaults still present.
        for tok in _STOP_TOKEN_DEFAULTS["ailiance-pixtral"]:
            assert tok in body["stop"]

    def test_non_string_user_stop_ignored(self):
        # If a malformed body sends a non-string stop value (e.g. None
        # slipped past validation), we drop it rather than crash.
        body = {"stop": [None, 42, "ok"]}
        _inject_stop_tokens(body, "ailiance-pixtral")
        assert "ok" in body["stop"]
        assert None not in body["stop"]
        assert 42 not in body["stop"]
