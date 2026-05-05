"""Integration tests for /v1/chat/completions with tools + streaming."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.worker.runtime import WorkerConfig
from src.worker.server import make_worker_app

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}]


def _make_client(monkeypatch, fake_text: str) -> TestClient:
    cfg = WorkerConfig(
        model_path="/tmp/fake",
        adapters_dir="/tmp/fake",
        domains=["python"],
        port=9210,
    )
    app = make_worker_app(cfg, skip_model_load=True)

    # Stub generate() so we don't need a real model.
    from src.worker.runtime import MLXWorkerRuntime

    def fake_generate(self, messages, max_tokens=2048, temperature=0.7):
        return fake_text, {"domain": None}

    def fake_apply(self, domain):
        return 0.0

    monkeypatch.setattr(MLXWorkerRuntime, "generate", fake_generate)
    monkeypatch.setattr(MLXWorkerRuntime, "apply", fake_apply)
    return TestClient(app)


def test_non_stream_with_tool_call(monkeypatch):
    fake = 'sure: {"name":"read_file","arguments":{"path":"/etc/hosts"}}'
    client = _make_client(monkeypatch, fake)
    resp = client.post("/v1/chat/completions", json={
        "model": "eu-kiki",
        "messages": [{"role": "user", "content": "read /etc/hosts"}],
        "tools": TOOLS,
        "stream": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tcs = choice["message"]["tool_calls"]
    assert tcs is not None and len(tcs) == 1
    assert tcs[0]["function"]["name"] == "read_file"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"path": "/etc/hosts"}


def test_stream_with_tool_call(monkeypatch):
    fake = '{"name":"read_file","arguments":{"path":"/x"}}'
    client = _make_client(monkeypatch, fake)
    resp = client.post("/v1/chat/completions", json={
        "model": "eu-kiki",
        "messages": [{"role": "user", "content": "go"}],
        "tools": TOOLS,
        "stream": True,
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "data: [DONE]" in body
    # Find first JSON chunk
    chunks = [
        line[len("data: "):]
        for line in body.split("\n\n")
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    assert len(chunks) >= 2
    first = json.loads(chunks[0])
    assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read_file"
    last = json.loads(chunks[-1])
    assert last["choices"][0]["finish_reason"] == "tool_calls"
    assert "usage" in last


def test_stream_content_only_no_tool_call(monkeypatch):
    client = _make_client(monkeypatch, "just a plain answer")
    resp = client.post("/v1/chat/completions", json={
        "model": "eu-kiki",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS,
        "stream": True,
    })
    assert resp.status_code == 200
    body = resp.text
    chunks = [
        line[len("data: "):]
        for line in body.split("\n\n")
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    first = json.loads(chunks[0])
    assert first["choices"][0]["delta"]["content"] == "just a plain answer"
    last = json.loads(chunks[-1])
    assert last["choices"][0]["finish_reason"] == "stop"


def test_no_tools_preserves_legacy_behavior(monkeypatch):
    client = _make_client(monkeypatch, "hello world")
    resp = client.post("/v1/chat/completions", json={
        "model": "eu-kiki",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "hello world"
    assert choice["message"].get("tool_calls") is None


def test_assistant_history_with_tool_calls_accepted(monkeypatch):
    """Dirac sends back assistant messages containing tool_calls in history."""
    client = _make_client(monkeypatch, "ok")
    resp = client.post("/v1/chat/completions", json={
        "model": "eu-kiki",
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_x", "content": "result"},
        ],
    })
    assert resp.status_code == 200
