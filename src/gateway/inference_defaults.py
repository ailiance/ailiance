"""Per-alias inference defaults for /v1/chat/completions.

Each backend behind ailiance-* has its own sweet spot for sampling
knobs and worker-specific quirks:

* **Reasoning models** (DeepSeek-R1-Distill, Gemma-3 thinking,
  Ministral-3 Reasoning, Apertus math reasoning) spend the first 300-
  1000 tokens on a hidden chain-of-thought before producing the final
  answer. Default ``max_tokens=1024`` truncates them mid-thought.
  Default 2048+ is required.
* **Pixtral** (vision-language) gives the most stable answers at low
  temperature; the high-temperature default 0.7 leads to colour /
  object hallucination on solid-tone images.
* **Qwen3.5 family** (``ailiance-qwen``, ``ailiance-qwen36``) injects
  a thinking phase into every reply unless
  ``chat_template_kwargs.enable_thinking=false`` is passed. For tool-
  call / classification / short-answer prompts the thinking phase
  burns tokens and slows the worker.
* **Stop tokens** for workers whose chat template leaks past end-of-
  turn (Pixtral fabricating ``USER:`` turns) — was PR #82, now folded
  in here.

This module exposes :func:`apply_inference_defaults` that merges the
registered defaults into a request body in-place. The contract is
**caller wins**: any field the caller already set is preserved
verbatim. Nested ``chat_template_kwargs`` is deep-merged so a caller
overriding ``enable_thinking`` doesn't lose other keys we set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InferenceDefaults:
    """Default knobs for a single alias.

    Fields default to ``None`` — only non-None values are applied. Keep
    this conservative: tuning each entry is a per-model decision, not
    a default-everything sweep.
    """

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    # Stop sequences appended to caller's ``stop`` list (de-duplicated).
    stop: tuple[str, ...] = field(default_factory=tuple)
    # Worker-specific knobs forwarded via extra_body / direct field.
    # OpenAI-compatible llama.cpp and mlx_lm.server both honour the
    # ``chat_template_kwargs`` field at the top level of the body.
    chat_template_kwargs: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REASONING_MAX = 2048
_REASONING_TEMP = 0.3  # R1 / Ministral reasoning empirically tighter at low temp

_INFERENCE_DEFAULTS: dict[str, InferenceDefaults] = {
    # ----- Reasoning models — need budget for the thinking phase -----
    "ailiance-reasoning-r1": InferenceDefaults(
        max_tokens=_REASONING_MAX,
        temperature=_REASONING_TEMP,
    ),
    "ailiance-gemma2": InferenceDefaults(
        max_tokens=_REASONING_MAX,
        # Gemma-3 reasoning emits <think>…</think> blocks; the gateway
        # strips them in _normalize_response_body, but the worker still
        # needs room to write them.
        temperature=0.4,
    ),
    "ailiance-ministral-reasoning": InferenceDefaults(
        max_tokens=_REASONING_MAX,
        temperature=_REASONING_TEMP,
    ),
    "ailiance-apertus-math-reasoning": InferenceDefaults(
        max_tokens=_REASONING_MAX,
        temperature=0.2,
    ),
    # ----- Vision-language — low temp + stop-token quirk -----
    "ailiance-pixtral": InferenceDefaults(
        # Pixtral 12B 4-bit MLX hallucinates colours / objects at temp
        # 0.7+. Validated 2026-05-12: temp=0.2 on a wikipedia-hosted
        # red image returned "red"; temp=0.7 sometimes returned "blue".
        temperature=0.2,
        max_tokens=512,
        # Worker leaks Vicuna template past end-of-turn. Folded in
        # from former _STOP_TOKEN_DEFAULTS (PR #82).
        stop=("\nUSER:", "USER:", "</s>", "[INST]"),
    ),
    # ----- Qwen3.5 family — disable thinking by default -----
    # ``feedback_qwen3_thinking_mode``: short-output workloads need
    # enable_thinking=false on the chat template, otherwise the model
    # burns budget on <think>…</think> before answering.
    "ailiance-qwen": InferenceDefaults(
        chat_template_kwargs={"enable_thinking": False},
    ),
    "ailiance-qwen36": InferenceDefaults(
        chat_template_kwargs={"enable_thinking": False},
    ),
    # ----- Flagship text — moderate temp for general use -----
    "ailiance-mistral-medium": InferenceDefaults(
        temperature=0.5,
    ),
    "ailiance-mistral": InferenceDefaults(
        temperature=0.5,
    ),
    # ----- Devstral code variants — deterministic for code -----
    "ailiance-devstral-base": InferenceDefaults(
        temperature=0.2,
        max_tokens=1024,
    ),
    "ailiance-coder-pro": InferenceDefaults(
        temperature=0.2,
        max_tokens=1024,
    ),
    "ailiance-python": InferenceDefaults(temperature=0.2),
    "ailiance-cpp": InferenceDefaults(temperature=0.2),
    "ailiance-rust-emb": InferenceDefaults(temperature=0.2),
    "ailiance-html": InferenceDefaults(temperature=0.2),
    "ailiance-ml-training": InferenceDefaults(temperature=0.3),
}


def _merge_stop(body: dict[str, Any], defaults: tuple[str, ...]) -> None:
    if not defaults:
        return
    user_stop = body.get("stop")
    if isinstance(user_stop, str):
        merged = [user_stop]
    elif isinstance(user_stop, list):
        merged = [s for s in user_stop if isinstance(s, str)]
    else:
        merged = []
    for tok in defaults:
        if tok not in merged:
            merged.append(tok)
    body["stop"] = merged


def _deep_merge_dict_field(
    body: dict[str, Any], key: str, defaults: dict[str, Any]
) -> None:
    """Deep-merge ``defaults`` into ``body[key]`` without overwriting caller.

    Caller-provided sub-keys win; defaults fill gaps only.
    """
    current = body.get(key)
    if not isinstance(current, dict):
        body[key] = dict(defaults)
        return
    for k, v in defaults.items():
        current.setdefault(k, v)


def _detect_user_set(body: dict[str, Any], field: str) -> bool:
    """True iff the caller explicitly set ``field`` to a non-default value.

    Pydantic's ``model_dump(exclude_none=True)`` only drops ``None``;
    schema defaults like ``temperature: float = 0.7`` and
    ``max_tokens: int = 2048`` *always* show up in the body even when
    the caller omitted them. We treat the schema default as "not user
    set" so per-model defaults can take effect.
    """
    if field not in body:
        return False
    # The two fields with non-None schema defaults that matter for
    # our overrides. Anything else, if present, is treated as user-set.
    schema_defaults = {"temperature": 0.7, "max_tokens": 2048}
    sentinel = schema_defaults.get(field)
    if sentinel is None:
        return True
    return body[field] != sentinel


def apply_inference_defaults(body: dict[str, Any], alias: str) -> bool:
    """Merge per-alias defaults into ``body``. Returns ``True`` if any
    field was actually mutated — useful for observability headers.

    Caller wins on every primitive field (we only fill if not user-set);
    ``stop`` is appended (de-duplicated); ``chat_template_kwargs`` is
    deep-merged.
    """
    defaults = _INFERENCE_DEFAULTS.get(alias)
    if defaults is None:
        return False
    mutated = False
    for field_name in (
        "temperature",
        "max_tokens",
        "top_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
    ):
        value = getattr(defaults, field_name)
        if value is None:
            continue
        if not _detect_user_set(body, field_name):
            body[field_name] = value
            mutated = True
    if defaults.stop:
        before = body.get("stop")
        _merge_stop(body, defaults.stop)
        if body.get("stop") != before:
            mutated = True
    if defaults.chat_template_kwargs:
        before = dict(body.get("chat_template_kwargs") or {})
        _deep_merge_dict_field(
            body, "chat_template_kwargs", defaults.chat_template_kwargs
        )
        if body.get("chat_template_kwargs") != before:
            mutated = True
    return mutated


def registered_aliases() -> frozenset[str]:
    """Read-only view of the keys that have defaults. Used by tests."""
    return frozenset(_INFERENCE_DEFAULTS.keys())
