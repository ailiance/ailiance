"""End-to-end smoke tests against production ailiance-gateway and cockpit.

These tests hit ``https://gateway.ailiance.fr`` and ``https://www.ailiance.fr``
and are intentionally **skipped by the default test runner**. Invoke
explicitly after a deploy with::

    uv run python -m pytest -m e2e -v

Configure via env if needed::

    AILIANCE_E2E_GATEWAY=https://gateway.ailiance.fr
    AILIANCE_E2E_COCKPIT=https://www.ailiance.fr

The suite mirrors the 8 manual verification tours we run after each
ship — sanity, reasoning normalize (non-stream + stream), vision,
file extract, inline files, inference defaults, Playground bundle.

Tests use ``httpx`` for HTTPS / streaming support. ``pytest -m e2e``
sets up no fixtures other than the configured base URLs; each test
is independent so flakes can be re-run in isolation.
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
import zlib

import httpx
import pytest

pytestmark = pytest.mark.e2e

GATEWAY = os.environ.get("AILIANCE_E2E_GATEWAY", "https://gateway.ailiance.fr")
COCKPIT = os.environ.get("AILIANCE_E2E_COCKPIT", "https://www.ailiance.fr")
TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_png(r: int, g: int, b: int, w: int = 64, h: int = 64) -> bytes:
    """Generate a tiny solid-colour PNG. Used by vision tests."""
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    raw = b"".join(b"\x00" + bytes([r, g, b]) * w for _ in range(h))
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _minimal_pdf(text: str) -> bytes:
    return (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources"
        b"<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>"
        b"/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 80>>stream\n"
        b"BT /F1 24 Tf 100 700 Td (" + text.encode() + b") Tj ET\n"
        b"endstream endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000052 00000 n \n0000000098 00000 n \n0000000214 00000 n \n"
        b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n325\n%%EOF"
    )


@pytest.fixture(scope="module")
def client():
    with httpx.Client(timeout=TIMEOUT) as c:
        yield c


# ---------------------------------------------------------------------------
# Tour 1 — Sanity
# ---------------------------------------------------------------------------


class TestTour1Sanity:
    def test_health(self, client: httpx.Client):
        resp = client.get(f"{GATEWAY}/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["router_loaded"] is True

    def test_models_listing_excludes_embed(self, client: httpx.Client):
        resp = client.get(f"{GATEWAY}/v1/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["data"]]
        assert len(ids) >= 40
        assert "ailiance-embed" not in ids, "embed must not be listed as chat-completable"
        assert "ailiance" in ids
        assert "ailiance-pixtral" in ids

    def test_staged_unknown_key_404(self, client: httpx.Client):
        resp = client.get(f"{GATEWAY}/v1/_staged/this-key-does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tour 2 — CORS preflight (this is what catches "Load failed" regressions)
# ---------------------------------------------------------------------------


class TestTour2CORS:
    @pytest.mark.parametrize("endpoint", ["/v1/files/extract", "/v1/chat/completions"])
    def test_options_preflight_from_www(self, client: httpx.Client, endpoint: str):
        resp = client.request(
            "OPTIONS",
            f"{GATEWAY}{endpoint}",
            headers={
                "Origin": COCKPIT,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code == 200, f"preflight failed on {endpoint}"
        assert resp.headers.get("access-control-allow-origin") == COCKPIT
        assert "POST" in resp.headers.get("access-control-allow-methods", "")


# ---------------------------------------------------------------------------
# Tour 3 — Reasoning normalization (non-streaming)
# ---------------------------------------------------------------------------


class TestTour3Reasoning:
    @pytest.mark.parametrize(
        "alias",
        ["ailiance-gemma2", "ailiance-reasoning-r1", "ailiance-ministral-reasoning"],
    )
    def test_reasoning_content_populated(self, client: httpx.Client, alias: str):
        resp = client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": alias,
                "messages": [{"role": "user", "content": "Say only the word OK."}],
                "max_tokens": 200,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200, resp.text
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        assert len(content) > 0, f"{alias} returned empty content (normalize broken)"
        # No leaked tags in the visible content.
        assert "[THINK]" not in content.upper()


# ---------------------------------------------------------------------------
# Tour 4 — SSE streaming normalize
# ---------------------------------------------------------------------------


class TestTour4Stream:
    def _consume_sse(self, resp: httpx.Response) -> tuple[str, int]:
        content_parts: list[str] = []
        events = 0
        for line in resp.iter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            events += 1
            for c in obj.get("choices", []):
                d = c.get("delta") or c.get("message") or {}
                if d.get("content"):
                    content_parts.append(d["content"])
        return "".join(content_parts), events

    def test_qwen_stream_regression(self, client: httpx.Client):
        with client.stream(
            "POST",
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance-qwen",
                "messages": [{"role": "user", "content": "Say only OK."}],
                "max_tokens": 10,
                "temperature": 0,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            content, events = self._consume_sse(resp)
        assert events >= 1
        assert content.strip() != ""

    def test_reasoning_stream_content_populated(self, client: httpx.Client):
        with client.stream(
            "POST",
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance-reasoning-r1",
                "messages": [{"role": "user", "content": "Reply OK."}],
                "max_tokens": 50,
                "temperature": 0,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            content, _ = self._consume_sse(resp)
        # Reasoning streams produce many chunks; assert non-empty.
        assert content.strip() != "", "reasoning stream content empty (normalize broken)"


# ---------------------------------------------------------------------------
# Tour 5 — Vision (HTTP URL + data URL staging)
# ---------------------------------------------------------------------------


class TestTour5Vision:
    def test_http_url_image(self, client: httpx.Client):
        resp = client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What animal? One word."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/200px-PNG_transparency_demonstration_1.png"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 15,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "Pixtral" in body["model"], "multimodal auto-route to Pixtral broken"
        content = body["choices"][0]["message"]["content"].lower()
        assert any(w in content for w in ("cat", "horse", "animal")), \
            f"unexpected vision output: {content!r}"

    def test_data_url_image_staged(self, client: httpx.Client):
        blue = base64.b64encode(_solid_png(0, 0, 255)).decode()
        resp = client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What primary color? One word."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{blue}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 10,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "Pixtral" in body["model"]
        content = body["choices"][0]["message"]["content"].lower()
        assert "blue" in content, f"data URL staging broken: {content!r}"


# ---------------------------------------------------------------------------
# Tour 6 — File extract endpoint
# ---------------------------------------------------------------------------


class TestTour6FilesExtract:
    def test_pdf_text(self, client: httpx.Client):
        pdf = _minimal_pdf("E2E pdf extract probe")
        resp = client.post(
            f"{GATEWAY}/v1/files/extract",
            files={"file": ("probe.pdf", pdf, "application/pdf")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["format"] == "pdf"
        assert "E2E pdf extract probe" in body["markdown"]

    def test_image_ocr(self, client: httpx.Client):
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (400, 100), "white")
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        d.text((10, 30), "OCR works", fill="black", font=font)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            f"{GATEWAY}/v1/files/extract",
            files={"file": ("ocr.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["format"] == "image"
        # OCR is fuzzy — accept any close match.
        text = body["markdown"].lower()
        assert "ocr" in text or "works" in text, f"OCR output unexpected: {text!r}"

    def test_unsupported_format_returns_400(self, client: httpx.Client):
        resp = client.post(
            f"{GATEWAY}/v1/files/extract",
            files={"file": ("a.exe", b"\x00binary", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "unsupported_format"


# ---------------------------------------------------------------------------
# Tour 7 — Inline input_file in chat
# ---------------------------------------------------------------------------


class TestTour7InlineFile:
    def test_pdf_data_url_inline(self, client: httpx.Client):
        pdf = _minimal_pdf("Sovereign EU AI ships 2026")
        b64 = base64.b64encode(pdf).decode()
        resp = client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Quote the document verbatim."},
                            {
                                "type": "input_file",
                                "file": {
                                    "url": f"data:application/pdf;base64,{b64}",
                                    "filename": "doc.pdf",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 60,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200, resp.text
        content = resp.json()["choices"][0]["message"]["content"]
        # Model should echo / reference the document text.
        assert "Sovereign" in content or "ships" in content, \
            f"inline file not extracted into prompt: {content!r}"


# ---------------------------------------------------------------------------
# Tour 8 — Inference defaults
# ---------------------------------------------------------------------------


class TestTour8InferenceDefaults:
    def test_qwen_short_answer(self, client: httpx.Client):
        # With enable_thinking=False default, 2+2 should return in
        # very few tokens (~2-5).
        resp = client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance-qwen",
                "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                "max_tokens": 30,
            },
        )
        assert resp.status_code == 200
        usage = resp.json()["usage"]
        assert usage["completion_tokens"] < 15, \
            f"qwen thinking not disabled — completion={usage['completion_tokens']}"

    def test_caller_max_tokens_wins(self, client: httpx.Client):
        # Caller-set max_tokens=5 must override per-alias defaults.
        resp = client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance-reasoning-r1",
                "messages": [{"role": "user", "content": "Anything"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["usage"]["completion_tokens"] <= 5


# ---------------------------------------------------------------------------
# Tour 9 — Cockpit Playground bundle
# ---------------------------------------------------------------------------


class TestTourAInventory:
    """Verify every response carries alias / base_model / LoRA info
    (PR #92 + #93). Without this surface, callers can't tell what
    actually served them — especially through the auto-router."""

    def test_explicit_alias_with_lora(self, client: httpx.Client):
        resp = client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance-kicad",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-ailiance-alias") == "ailiance-kicad"
        assert resp.headers.get("x-ailiance-lora") == "mascarade-kicad"
        body = resp.json()
        assert body.get("ailiance", {}).get("alias") == "ailiance-kicad"
        assert body["ailiance"]["lora"] == ["mascarade-kicad"]

    def test_explicit_alias_no_lora(self, client: httpx.Client):
        resp = client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance-pixtral",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-ailiance-alias") == "ailiance-pixtral"
        # No LoRA → header omitted entirely.
        assert "x-ailiance-lora" not in resp.headers
        body = resp.json()
        assert body["ailiance"]["lora"] == []

    def test_auto_router_resolves_to_served_alias(self, client: httpx.Client):
        # KiCad prompt should classify to kicad domain → mascarade alias.
        resp = client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": "ailiance",
                "messages": [
                    {"role": "user", "content": "Help me design a KiCad PCB schematic."}
                ],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        alias = resp.headers.get("x-ailiance-alias")
        # Either it resolved to ailiance-kicad (classifier picked kicad)
        # OR it stayed at ailiance (classifier picked an unmapped domain).
        # In both cases we get a value; quality of resolution is tested
        # in unit tests.
        assert alias is not None and alias.startswith("ailiance")


class TestTour9CockpitBundle:
    def test_homepage_serves(self, client: httpx.Client):
        resp = client.get(f"{COCKPIT}/")
        assert resp.status_code == 200

    def test_bundle_contains_upload_feature(self, client: httpx.Client):
        # Locate the JS bundle then grep for the key strings shipped
        # in the Playground upload + reasoning defaults PRs.
        home = client.get(f"{COCKPIT}/").text
        import re

        match = re.search(r"/assets/index-[A-Za-z0-9_-]+\.js", home)
        assert match, "bundle path not found in index.html"
        js = client.get(f"{COCKPIT}{match.group(0)}").text
        for needle in (
            "Paperclip",
            "Attach a file",
            "v1/files/extract",
            "ailiance-gemma2",
            "ailiance-reasoning-r1",
            "gateway.ailiance.fr",
        ):
            assert needle in js, f"bundle missing feature: {needle!r}"

    @pytest.mark.parametrize(
        "route", ["/catalog", "/bench", "/chat", "/transparency", "/api/public/telemetry"]
    )
    def test_route_serves(self, client: httpx.Client, route: str):
        resp = client.get(f"{COCKPIT}{route}")
        assert resp.status_code == 200
