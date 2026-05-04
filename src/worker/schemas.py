"""OpenAI-compatible request/response schemas."""

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _flatten_content(value: Any) -> str | None:
    """Accept OpenAI native content blocks (list[{type, text}]) and flatten to text.

    OpenAI clients (Cline/Dirac, Anthropic-via-OpenAI, etc.) often send messages with
    `content` as a list of typed blocks instead of a plain string. We coerce to string
    so the worker tokenizer's chat template can apply uniformly.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
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
    content: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, value: Any) -> str | None:
        return _flatten_content(value)


class ChatCompletionRequest(BaseModel, extra="ignore"):
    model: str = "eu-kiki"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


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
    model: str = "eu-kiki"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
