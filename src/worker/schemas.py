"""OpenAI-compatible request/response schemas."""

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator


_MULTIMODAL_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        # vision / audio
        "image_url",
        "image",
        "input_image",
        "audio_url",
        "audio",
        "input_audio",
        # documents — the gateway extracts these inline at /v1/chat/completions
        # (input_file → text block carrying the markdown).
        "input_file",
        "file",
    }
)


def _content_has_multimodal_block(value: Any) -> bool:
    """True if a content list contains a non-text part (image/audio/etc.)."""
    if not isinstance(value, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") in _MULTIMODAL_BLOCK_TYPES
        for b in value
    )


def _flatten_content(value: Any) -> Any:
    """Coerce OpenAI native content blocks to a worker-friendly value.

    Two cases:

    * Text-only multipart content (``[{"type":"text","text":"..."}, ...]``)
      from OpenAI clients (Cline/Dirac, Anthropic-via-OpenAI, …) is
      flattened to a plain string so the tokenizer's chat template
      applies uniformly across text-only workers.
    * Multimodal content (any block with type in
      ``_MULTIMODAL_BLOCK_TYPES``) is preserved as a list so vision
      workers (Pixtral) receive the raw image parts. The gateway is
      responsible for routing to a vision-capable worker — text-only
      workers will receive the list unchanged and may error; that is
      the correct behaviour (no silent data loss).
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        if _content_has_multimodal_block(value):
            return value
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(parts) if parts else None
    return str(value)


class ChatMessage(BaseModel, extra="ignore"):
    role: str
    # Either a flat string (text-only) or a list of OpenAI content
    # blocks (multimodal — preserved for vision workers).
    content: str | list[dict[str, Any]] | None = None
    # Allow assistant tool_calls in conversation history (Dirac sends these
    # back when continuing a tool-use loop). Free-form list of dicts so we
    # don't enforce OpenAI's exact ToolCall shape on input.
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, value: Any) -> Any:
        return _flatten_content(value)


class FunctionDef(BaseModel, extra="ignore"):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDef(BaseModel, extra="ignore"):
    type: str = "function"
    function: FunctionDef


class ChatCompletionRequest(BaseModel, extra="ignore"):
    model: str = "ailiance"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    stream_options: dict[str, Any] | None = None
    # Router v0.3 opt-in. When set, the gateway dispatches through
    # the chain orchestrator. Recognised keys: chain_policy (str),
    # max_retries (int), include_audit (bool). Unknown keys ignored.
    extra_body: dict[str, Any] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "ailiance"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
