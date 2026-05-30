"""Tests for the multimodal auto-route and the schema's content handling.

When a request arrives on the auto-router alias (``model: "ailiance"``)
with an ``image_url`` (or other multimodal) block, the gateway should
transparently redirect to :data:`_CANONICAL_VISION_ALIAS` so the user
gets a vision-capable worker without having to know the alias.

The schema's ``_flatten_content`` validator must preserve multimodal
list content (otherwise the image is silently destroyed at the
gateway-pydantic boundary).
"""

from __future__ import annotations

from src.gateway.server import (
    _CANONICAL_VISION_ALIAS,
    _maybe_route_to_vision,
    _public_base_url,
    _VISION_ALIASES,
    _request_has_images,
)
from src.worker.schemas import (
    _content_has_multimodal_block,
    _flatten_content,
    ChatCompletionRequest,
    ChatMessage,
)


class TestSchemaPreservesMultimodal:
    def test_text_only_list_flattens_to_string(self):
        out = _flatten_content([{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}])
        assert out == "hello\nworld"

    def test_string_passthrough(self):
        assert _flatten_content("hello") == "hello"

    def test_none_passthrough(self):
        assert _flatten_content(None) is None

    def test_image_url_list_preserved(self):
        blocks = [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
        ]
        out = _flatten_content(blocks)
        assert out == blocks  # raw list preserved

    def test_input_image_alias_preserved(self):
        # Some SDKs use "input_image" instead of "image_url".
        blocks = [{"type": "input_image", "image_url": "..."}]
        out = _flatten_content(blocks)
        assert out == blocks

    def test_audio_block_preserved(self):
        blocks = [{"type": "input_audio", "audio_url": "..."}]
        assert _flatten_content(blocks) == blocks

    def test_chat_message_validator_keeps_image_list(self):
        msg = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            ],
        )
        assert isinstance(msg.content, list)
        assert msg.content[1]["type"] == "image_url"


class TestRequestHasImages:
    def _req(self, content):
        return ChatCompletionRequest(model="ailiance", messages=[ChatMessage(role="user", content=content)])

    def test_string_content_no_image(self):
        assert _request_has_images(self._req("hello world")) is False

    def test_text_only_list_no_image(self):
        assert _request_has_images(self._req([{"type": "text", "text": "hi"}])) is False

    def test_image_url_detected(self):
        blocks = [
            {"type": "text", "text": "what?"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]
        assert _request_has_images(self._req(blocks)) is True

    def test_input_image_detected(self):
        assert _request_has_images(self._req([{"type": "input_image"}])) is True

    def test_empty_messages(self):
        req = ChatCompletionRequest(model="ailiance", messages=[])
        assert _request_has_images(req) is False


class TestVisionAliasContract:
    def test_canonical_is_in_set(self):
        assert _CANONICAL_VISION_ALIAS in _VISION_ALIASES

    def test_canonical_is_pixtral(self):
        # Pixtral-12B on omlx :8500 is the canonical vision worker (issue #133
        # resolved: the image-drop was a missing torch/torchvision image
        # processor, not fix_mistral_regex). gemma-4-E4B stays vision-capable
        # as the lightweight explicit alias. If the catalog changes this test
        # forces a deliberate update.
        assert _CANONICAL_VISION_ALIAS == "ailiance-pixtral"


class TestMultimodalHelper:
    def test_helper_matches_flatten_decision(self):
        # The two helpers must agree on what counts as multimodal so a
        # request preserved by the schema is also detected by the
        # router.
        blocks = [{"type": "image_url", "image_url": {"url": "..."}}]
        assert _content_has_multimodal_block(blocks) is True
        text_blocks = [{"type": "text", "text": "hi"}]
        assert _content_has_multimodal_block(text_blocks) is False


class TestVisionRouting:
    """Any image-bearing request must reach the canonical vision worker
    (Pixtral-12B on omlx), regardless of the requested alias."""

    def test_image_request_routes_any_alias_to_vision(self):
        for m in ("ailiance", "ailiance-coder", "ailiance-qwen", "some-domain-model"):
            assert _maybe_route_to_vision(m, True) == "ailiance-pixtral"

    def test_vision_alias_passes_through(self):
        # Already a vision alias → no rewrite (canonical and the pixtral
        # follow-up alias both pass through).
        assert _maybe_route_to_vision("ailiance-gemma4-omlx", True) == "ailiance-gemma4-omlx"
        assert _maybe_route_to_vision("ailiance-pixtral", True) == "ailiance-pixtral"

    def test_text_request_keeps_requested_alias(self):
        assert _maybe_route_to_vision("ailiance-coder", False) == "ailiance-coder"
        assert _maybe_route_to_vision("ailiance", False) == "ailiance"
        assert _maybe_route_to_vision("ailiance-gemma4-omlx", False) == "ailiance-gemma4-omlx"


class TestImageFetchBaseUrl:
    """The worker fetches staged images from this URL, so it must prefer a
    worker-reachable address over the public (website-fronted) domain."""

    def test_prefers_dedicated_fetch_url(self, monkeypatch):
        monkeypatch.setenv("AILIANCE_IMAGE_FETCH_BASE_URL", "http://100.78.191.52:9300")
        monkeypatch.setenv("AILIANCE_PUBLIC_BASE_URL", "https://gateway.ailiance.fr")
        assert _public_base_url() == "http://100.78.191.52:9300"

    def test_falls_back_to_public(self, monkeypatch):
        monkeypatch.delenv("AILIANCE_IMAGE_FETCH_BASE_URL", raising=False)
        monkeypatch.setenv("AILIANCE_PUBLIC_BASE_URL", "https://gateway.ailiance.fr")
        assert _public_base_url() == "https://gateway.ailiance.fr"

    def test_local_default(self, monkeypatch):
        monkeypatch.delenv("AILIANCE_IMAGE_FETCH_BASE_URL", raising=False)
        monkeypatch.delenv("AILIANCE_PUBLIC_BASE_URL", raising=False)
        assert _public_base_url() == "http://localhost:9300"
