"""Tests for per-alias inference defaults applied to /v1/chat/completions bodies."""

from __future__ import annotations

import pytest

from src.gateway.inference_defaults import (
    InferenceDefaults,
    apply_inference_defaults,
    registered_aliases,
)


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


class TestRegistryContract:
    def test_known_reasoning_aliases_registered(self):
        # All four reasoning aliases that the Playground bumps to 2048
        # must also have a server-side default, otherwise API callers
        # without the Playground hit truncation.
        for alias in (
            "ailiance-reasoning-r1",
            "ailiance-gemma2",
            "ailiance-ministral-reasoning",
            "ailiance-apertus-math-reasoning",
        ):
            assert alias in registered_aliases(), f"{alias} missing defaults"

    def test_pixtral_has_stop_tokens(self):
        from src.gateway.inference_defaults import _INFERENCE_DEFAULTS

        assert "\nUSER:" in _INFERENCE_DEFAULTS["ailiance-pixtral"].stop

    def test_qwen_has_disabled_thinking(self):
        from src.gateway.inference_defaults import _INFERENCE_DEFAULTS

        for alias in ("ailiance-qwen", "ailiance-qwen36"):
            d = _INFERENCE_DEFAULTS[alias]
            assert d.chat_template_kwargs == {"enable_thinking": False}


# ---------------------------------------------------------------------------
# apply_inference_defaults — primitive merging
# ---------------------------------------------------------------------------


class TestApplyDefaults:
    def test_unknown_alias_noop(self):
        body = {"temperature": 0.7, "max_tokens": 2048}
        mutated = apply_inference_defaults(body, "ailiance-no-such-thing")
        assert mutated is False
        assert body == {"temperature": 0.7, "max_tokens": 2048}

    def test_reasoning_alias_sets_max_tokens_when_caller_at_schema_default(self):
        # Pydantic emits temperature=0.7, max_tokens=2048 even when the
        # caller didn't supply them. Our schema default for max_tokens
        # is 2048; reasoning defaults push it to 2048 too — so this is
        # a no-op for that case. Test instead with a reasoning alias
        # whose temperature default differs.
        body = {"temperature": 0.7, "max_tokens": 2048}
        mutated = apply_inference_defaults(body, "ailiance-reasoning-r1")
        # temperature was at schema default 0.7 → replaced with 0.3
        assert body["temperature"] == 0.3
        # max_tokens already at schema default 2048; defaults also want
        # 2048 — value stays.
        assert body["max_tokens"] == 2048
        assert mutated is True

    def test_caller_set_temperature_preserved(self):
        body = {"temperature": 0.95, "max_tokens": 2048}
        apply_inference_defaults(body, "ailiance-pixtral")
        assert body["temperature"] == 0.95  # caller wins

    def test_caller_set_max_tokens_preserved(self):
        body = {"temperature": 0.7, "max_tokens": 100}
        apply_inference_defaults(body, "ailiance-reasoning-r1")
        assert body["max_tokens"] == 100  # caller wins

    def test_pixtral_stop_tokens_injected(self):
        body = {"temperature": 0.7, "max_tokens": 2048}
        apply_inference_defaults(body, "ailiance-pixtral")
        assert "\nUSER:" in body["stop"]
        assert "USER:" in body["stop"]
        # temp lowered
        assert body["temperature"] == 0.2
        # max_tokens lowered from schema default
        assert body["max_tokens"] == 512

    def test_pixtral_user_stop_preserved(self):
        body = {"temperature": 0.7, "max_tokens": 2048, "stop": ["###END"]}
        apply_inference_defaults(body, "ailiance-pixtral")
        # User value first.
        assert body["stop"][0] == "###END"
        # Defaults appended without duplicating.
        for tok in ("\nUSER:", "USER:", "</s>", "[INST]"):
            assert tok in body["stop"]
            assert body["stop"].count(tok) == 1


# ---------------------------------------------------------------------------
# chat_template_kwargs deep merge
# ---------------------------------------------------------------------------


class TestChatTemplateKwargs:
    def test_qwen_thinking_disabled_when_absent(self):
        body = {"temperature": 0.7, "max_tokens": 2048}
        apply_inference_defaults(body, "ailiance-qwen")
        assert body["chat_template_kwargs"]["enable_thinking"] is False

    def test_caller_thinking_override_preserved(self):
        # A caller who explicitly wants thinking on (deep-research path)
        # must not be overridden.
        body = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        apply_inference_defaults(body, "ailiance-qwen")
        assert body["chat_template_kwargs"]["enable_thinking"] is True

    def test_caller_unrelated_template_kwarg_preserved(self):
        # Caller sets some other knob — we must add enable_thinking
        # without touching theirs.
        body = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "chat_template_kwargs": {"some_other": "value"},
        }
        apply_inference_defaults(body, "ailiance-qwen")
        assert body["chat_template_kwargs"]["some_other"] == "value"
        assert body["chat_template_kwargs"]["enable_thinking"] is False


# ---------------------------------------------------------------------------
# Mutation reporting (used by observability headers)
# ---------------------------------------------------------------------------


class TestMutationReport:
    def test_unknown_returns_false(self):
        body = {"temperature": 0.7, "max_tokens": 2048}
        assert apply_inference_defaults(body, "ailiance-unknown") is False

    def test_known_returns_true(self):
        body = {"temperature": 0.7, "max_tokens": 2048}
        assert apply_inference_defaults(body, "ailiance-reasoning-r1") is True

    def test_known_but_caller_set_everything_returns_false(self):
        # Caller already supplied everything our defaults would set →
        # no mutation reported.
        body = {
            "temperature": 0.1,
            "max_tokens": 64,
            "stop": ["\nUSER:", "USER:", "</s>", "[INST]"],
        }
        assert apply_inference_defaults(body, "ailiance-pixtral") is False


# ---------------------------------------------------------------------------
# Dataclass smoke
# ---------------------------------------------------------------------------


class TestDataclass:
    def test_empty_defaults_have_empty_stop_tuple(self):
        d = InferenceDefaults()
        assert d.stop == ()

    def test_apply_with_empty_defaults_no_op(self):
        # Sanity: registering an alias with no fields set is a no-op
        # — defensive check against future entries that add nothing.
        from src.gateway import inference_defaults as mod

        body = {"temperature": 0.7, "max_tokens": 2048}
        mod._INFERENCE_DEFAULTS["__test_empty__"] = InferenceDefaults()
        try:
            mutated = apply_inference_defaults(body, "__test_empty__")
            assert mutated is False
            assert body == {"temperature": 0.7, "max_tokens": 2048}
        finally:
            mod._INFERENCE_DEFAULTS.pop("__test_empty__", None)
