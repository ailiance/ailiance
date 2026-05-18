"""Tests for the alias inventory registry + response stamping."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.gateway.alias_inventory import (
    AliasInventory,
    get_alias_inventory,
    inventory_or_unknown,
    known_aliases,
    resolve_effective_alias,
    to_dict,
    to_headers,
)


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


class TestRegistryContract:
    def test_pixtral_base_and_no_lora(self):
        inv = get_alias_inventory("ailiance-pixtral")
        assert inv is not None
        assert "Pixtral" in inv.base_model
        assert inv.lora == ()

    def test_mascarade_kicad_has_lora(self):
        inv = get_alias_inventory("ailiance-kicad")
        assert inv is not None
        assert inv.lora == ("mascarade-kicad",)
        assert inv.base_model.startswith("Qwen3-4B")

    def test_devstral_python_has_lora(self):
        inv = get_alias_inventory("ailiance-python")
        assert inv is not None
        assert inv.lora == ("devstral-python",)
        assert "Devstral" in inv.base_model

    def test_apertus_math_has_lora(self):
        inv = get_alias_inventory("ailiance-apertus-math")
        assert inv is not None
        assert inv.lora == ("apertus-math",)
        assert "Apertus" in inv.base_model

    def test_auto_router_is_registered(self):
        inv = get_alias_inventory("ailiance")
        assert inv is not None
        assert inv.base_model == "auto-router"
        assert inv.lora == ()

    def test_unknown_alias_returns_none(self):
        assert get_alias_inventory("ailiance-no-such-thing") is None

    def test_no_lora_aliases_have_empty_tuple(self):
        # Pure base-model aliases never expose a phantom LoRA.
        for alias in (
            "ailiance",
            "ailiance-mistral-medium",
            "ailiance-pixtral",
            "ailiance-reasoning-r1",
            "ailiance-gemma",
            "ailiance-embed",
        ):
            inv = get_alias_inventory(alias)
            assert inv is not None and inv.lora == (), f"{alias} unexpectedly has LoRA"


class TestInventoryOrUnknown:
    def test_returns_real_for_known(self):
        inv = inventory_or_unknown("ailiance-pixtral")
        assert inv.alias == "ailiance-pixtral"

    def test_returns_placeholder_for_unknown(self):
        inv = inventory_or_unknown("ailiance-mystery")
        assert inv.alias == "ailiance-mystery"
        assert inv.base_model == "unknown"
        assert inv.lora == ()

    def test_none_input_yields_unknown(self):
        inv = inventory_or_unknown(None)
        assert inv.alias == "unknown"


class TestSerializers:
    def test_to_dict_shape(self):
        inv = inventory_or_unknown("ailiance-kicad")
        d = to_dict(inv)
        assert d == {
            "alias": "ailiance-kicad",
            "base_model": inv.base_model,
            "lora": ["mascarade-kicad"],
            "worker_host": inv.worker_host,
        }

    def test_to_headers_without_lora(self):
        inv = inventory_or_unknown("ailiance-pixtral")
        h = to_headers(inv)
        assert h["X-Ailiance-Alias"] == "ailiance-pixtral"
        assert "Pixtral" in h["X-Ailiance-Base-Model"]
        assert "X-Ailiance-LoRA" not in h  # omitted when empty

    def test_to_headers_with_lora(self):
        inv = inventory_or_unknown("ailiance-kicad")
        h = to_headers(inv)
        assert h["X-Ailiance-LoRA"] == "mascarade-kicad"

    def test_to_headers_multi_lora(self):
        inv = AliasInventory(alias="x", base_model="b", lora=("a", "b", "c"))
        h = to_headers(inv)
        assert h["X-Ailiance-LoRA"] == "a,b,c"


# ---------------------------------------------------------------------------
# Catalog completeness
# ---------------------------------------------------------------------------


class TestCatalogCoverage:
    def test_every_listed_chat_model_has_inventory(self):
        # Every alias that /v1/models advertises should have an
        # inventory entry, otherwise we ship "unknown" headers in prod.
        from src.gateway.server import make_gateway_app
        from fastapi.testclient import TestClient

        app = make_gateway_app(skip_router_load=True)
        client = TestClient(app)
        resp = client.get("/v1/models")
        listed = {m["id"] for m in resp.json()["data"]}
        registered = known_aliases()
        missing = listed - registered
        assert not missing, f"aliases in /v1/models without inventory: {sorted(missing)}"

    def test_force_map_aliases_have_inventory(self):
        from src.gateway.server import MODEL_FORCE_MAP

        registered = known_aliases()
        missing = set(MODEL_FORCE_MAP) - registered
        # Some force-map entries are pure aliases (eurollm, granite-30b);
        # the test catches drift. We allow a small known set of "alias-
        # only" mappings (cascade or planned aliases not yet exposed
        # via /v1/models).
        known_alias_only = {"ailiance-devstral"}  # legacy alias
        unexpected = missing - known_alias_only
        assert not unexpected, f"MODEL_FORCE_MAP entries without inventory: {sorted(unexpected)}"


# ---------------------------------------------------------------------------
# Response stamping via TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from src.gateway.server import make_gateway_app

    return TestClient(make_gateway_app(skip_router_load=True))


class TestResolveEffectiveAlias:
    def test_cascade_wins(self):
        assert (
            resolve_effective_alias(
                "ailiance", cascade_alias="ailiance-reasoning-r1", domain="python"
            )
            == "ailiance-reasoning-r1"
        )

    def test_explicit_alias_passthrough(self):
        assert resolve_effective_alias("ailiance-pixtral") == "ailiance-pixtral"

    def test_auto_router_mascarade_kicad(self):
        # Mascarade domain → ailiance-<domain> auto-derived.
        assert resolve_effective_alias("ailiance", domain="kicad") == "ailiance-kicad"

    def test_auto_router_mascarade_spice(self):
        assert resolve_effective_alias("ailiance", domain="spice") == "ailiance-spice"

    def test_auto_router_explicit_map_math_reasoning(self):
        assert (
            resolve_effective_alias("ailiance", domain="math-reasoning")
            == "ailiance-apertus-math-reasoning"
        )

    def test_auto_router_python_domain(self):
        # Code domains go to Devstral hot-swap aliases.
        assert resolve_effective_alias("ailiance", domain="python") == "ailiance-python"

    def test_auto_router_general_falls_to_flagship(self):
        assert (
            resolve_effective_alias("ailiance", domain="general")
            == "ailiance-mistral-medium"
        )

    def test_auto_router_unknown_domain_returns_ailiance(self):
        assert resolve_effective_alias("ailiance", domain="no-such-domain") == "ailiance"

    def test_auto_router_no_domain(self):
        assert resolve_effective_alias("ailiance") == "ailiance"


class TestResponseStamping:
    """We can't easily run a full chat against a TestClient (workers
    aren't reachable), but we can verify the inventory module is
    imported by the server module — a smoke check against the wiring."""

    def test_server_imports_inventory(self):
        # If the imports break, this raises at collection.
        from src.gateway.server import inventory_or_unknown as srv_inv

        assert srv_inv("ailiance-kicad").lora == ("mascarade-kicad",)
