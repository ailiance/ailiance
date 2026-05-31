"""Tests for gateway message-role normalization.

The gateway collapses consecutive same-role messages before forwarding so
strict backend chat templates (Qwen3 / Mistral / DeepSeek) don't raise
"Conversation roles must alternate user/assistant/...".
"""

from src.gateway.inference_defaults import normalize_message_roles
from src.worker.schemas import ChatMessage


def _roles(messages):
    return [m["role"] if isinstance(m, dict) else m.role for m in messages]


def _content(msg):
    return msg["content"] if isinstance(msg, dict) else msg.content


def test_collapses_consecutive_user_messages():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "tool result"},
    ]
    out = normalize_message_roles(msgs)
    assert _roles(out) == ["system", "user"]
    assert _content(out[1]) == "first\n\ntool result"


def test_alternating_sequence_unchanged():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    out = normalize_message_roles(msgs)
    assert _roles(out) == ["system", "user", "assistant", "user"]
    assert [_content(m) for m in out] == ["sys", "a", "b", "c"]


def test_three_consecutive_users_merge_into_one():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "user", "content": "y"},
        {"role": "user", "content": "z"},
    ]
    out = normalize_message_roles(msgs)
    assert _roles(out) == ["user"]
    assert _content(out[0]) == "x\n\ny\n\nz"


def test_tool_use_messages_are_not_merged():
    # assistant with tool_calls then a tool result must stay distinct, and a
    # following assistant tool_calls must not be merged with another.
    msgs = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "1"},
        {"role": "tool", "content": "result2", "tool_call_id": "2"},
    ]
    out = normalize_message_roles(msgs)
    # nothing collapsed: tool_calls / tool messages are structural boundaries
    assert _roles(out) == ["user", "assistant", "tool", "tool"]


def test_system_messages_merge_but_never_fold_into_user():
    msgs = [
        {"role": "system", "content": "s1"},
        {"role": "system", "content": "s2"},
        {"role": "user", "content": "hi"},
    ]
    out = normalize_message_roles(msgs)
    assert _roles(out) == ["system", "user"]
    assert _content(out[0]) == "s1\n\ns2"
    assert _content(out[1]) == "hi"


def test_multimodal_content_merge_preserves_blocks():
    msgs = [
        {"role": "user", "content": "look:"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
    ]
    out = normalize_message_roles(msgs)
    assert _roles(out) == ["user"]
    merged = _content(out[0])
    assert isinstance(merged, list)
    assert {"type": "text", "text": "look:"} in merged
    assert any(b.get("type") == "image_url" for b in merged)


def test_works_with_pydantic_chatmessage():
    msgs = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="user", content="b"),
    ]
    out = normalize_message_roles(msgs)
    assert [m.role for m in out] == ["user"]
    assert out[0].content == "a\n\nb"


def test_empty_and_none_safe():
    assert normalize_message_roles([]) == []
    assert normalize_message_roles(None) == []
