"""Single source of truth for what each ``ailiance-*`` alias serves.

Every alias maps to a triple:

* ``alias`` — the public ailiance-* name the caller used (after any
  gateway-side rewrite: multimodal auto-route, cascade, etc.).
* ``base_model`` — the **logical** base model name, **not** the
  filesystem path returned by the worker. e.g. ``Qwen3-Next-80B-Q4-MoE``
  rather than ``/home/.../models/qwen-q4.gguf``.
* ``lora`` — tuple of LoRA adapter names stacked on the base, in
  application order. Empty tuple means no LoRA (raw base model).

Used by the chat-completions handler to stamp three things onto every
response:

* HTTP headers ``X-Ailiance-Alias``, ``X-Ailiance-Base-Model``,
  ``X-Ailiance-LoRA`` (comma-separated) — visible on streaming and
  non-streaming responses alike.
* The non-streaming JSON body's ``ailiance`` field — handy for
  clients that don't surface response headers (most OpenAI SDKs hide
  them).
* Logs and telemetry — so a request can be re-played knowing exactly
  which LoRA stack served it.

The registry must stay aligned with:
* ``MODEL_FORCE_MAP`` (alias → worker port)
* ``ALIAS_MODEL_REWRITES`` (alias → upstream model field)
* ``/v1/models`` listing
* ``configs/models-display.yaml``

When adding a new alias, register it here too — a missing entry falls
back to ``unknown`` instead of a stack trace, but observability
degrades silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from src.router.domain_map import (
    DOMAIN_TO_OMLX_MODEL,
    DOMAIN_TO_QWEN36,
    OMLX_PORT,
    QWEN36_PORT,
    QWEN36_PORT_B,
)

_QWEN36_PORTS = frozenset({QWEN36_PORT, QWEN36_PORT_B})


@dataclass(frozen=True)
class AliasInventory:
    """What an alias actually serves, in logical-model terms."""

    alias: str
    base_model: str
    lora: tuple[str, ...] = field(default_factory=tuple)
    # Free-form, optional. Surfaced in the body's ``ailiance`` dict for
    # introspection — useful when the same alias points at different
    # workers across hosts.
    worker_host: str | None = None


# ---------------------------------------------------------------------------
# Registry. Names match CLAUDE.md infrastructure section + /v1/models.
# ---------------------------------------------------------------------------

# Base model strings used across multiple aliases — keep them as
# constants so a rename only touches one place.
BASE_MISTRAL_MEDIUM = "Mistral-Medium-3.5-128B-MLX-Q8"
BASE_QWEN3_NEXT_80B = "Qwen3-Next-80B-A3B-MoE-Q4_K_M"
BASE_QWEN3_5_9B_MLX = "Qwen3.5-9B-MLX-4bit"
BASE_QWEN3_6_35B = "Qwen3.6-35B-A3B-MLX-BF16"
BASE_GRANITE_4_30B = "Granite-4.1-30B-Q4_K_M"
BASE_GRANITE_4_30B_MLX = "Granite-4.1-30B-4bit-MLX"
BASE_GEMMA_3_4B = "Gemma-3-4B-IT-Q4-GGUF"
BASE_GEMMA_3_4B_MLX = "Gemma-3-4B-IT-4bit-MLX"
BASE_GEMMA_4_E4B = "Gemma-4-E4B-IT-4bit-MLX"
BASE_GEMMA_4_E2B = "Gemma-4-E2B-IT-4bit-MLX"
BASE_MINISTRAL_3_14B = "Ministral-3-14B-Instruct-2512-4bit-MLX"
BASE_MINISTRAL_3_14B_REASONING = "Ministral-3-14B-Reasoning-2512-4bit-MLX"
BASE_APERTUS_70B = "Apertus-70B-Instruct-2509-4bit-MLX"
BASE_DEVSTRAL_24B = "Devstral-Small-2-24B-Instruct-4bit-MLX"
BASE_DEEPSEEK_R1_DISTILL = "DeepSeek-R1-Distill-Qwen-32B-4bit-MLX"
BASE_LLAMA_3_3_70B = "Llama-3.3-70B-Instruct-4bit-MLX"
BASE_PIXTRAL_12B = "Pixtral-12B-4bit-MLX"
BASE_MISTRAL_SMALL_24B = "Mistral-Small-3.1-24B-Instruct-4bit-MLX"
BASE_QWEN_CODER_30B = "Qwen3-Coder-30B-A3B-Instruct-4bit-MLX"
BASE_QWEN_235B = "Qwen3-235B-A22B-MoE-4bit-MLX"
BASE_MIXTRAL_8X22B = "Mixtral-8x22B-Instruct-v0.1-4bit-MLX"
BASE_BGE_M3 = "bge-m3"
BASE_MASCARADE_4B = "Qwen3-4B-Q4_K_M"  # Ollama mascarade base (Tower)
BASE_EUROLLM_22B = "EuroLLM-22B-Instruct-2512"


_REGISTRY: dict[str, AliasInventory] = {
    # ----- Auto-router pseudo-alias (resolved at request time) -----
    "ailiance": AliasInventory(
        alias="ailiance",
        base_model="auto-router",
        lora=(),
        worker_host="classifier-dispatched",
    ),
    # ----- Mac Studio MLX workers -----
    "ailiance-mistral-medium": AliasInventory(
        alias="ailiance-mistral-medium",
        base_model=BASE_MISTRAL_MEDIUM,
        worker_host="studio:9301",
    ),
    "ailiance-mistral": AliasInventory(
        alias="ailiance-mistral",
        base_model=BASE_MISTRAL_MEDIUM,
        worker_host="studio:9301",
    ),
    "ailiance-apertus": AliasInventory(
        alias="ailiance-apertus",
        base_model=BASE_MISTRAL_MEDIUM,  # legacy alias → routes to Mistral now
        worker_host="studio:9301",
    ),
    "ailiance-eurollm": AliasInventory(
        alias="ailiance-eurollm",
        base_model=BASE_EUROLLM_22B,
        worker_host="studio:9303",
    ),
    "ailiance-reasoning-r1": AliasInventory(
        alias="ailiance-reasoning-r1",
        base_model=BASE_DEEPSEEK_R1_DISTILL,
        worker_host="studio:9323",
    ),
    "ailiance-llama": AliasInventory(
        alias="ailiance-llama",
        base_model=BASE_LLAMA_3_3_70B,
        worker_host="studio:9324",
    ),
    "ailiance-pixtral": AliasInventory(
        alias="ailiance-pixtral",
        base_model=BASE_PIXTRAL_12B,
        worker_host="studio:9325",
    ),
    "ailiance-mistral-small": AliasInventory(
        alias="ailiance-mistral-small",
        base_model=BASE_MISTRAL_SMALL_24B,
        worker_host="studio:9326",
    ),
    "ailiance-coder-pro": AliasInventory(
        alias="ailiance-coder-pro",
        base_model=BASE_QWEN_CODER_30B,
        worker_host="studio:9327",
    ),
    "ailiance-flagship": AliasInventory(
        alias="ailiance-flagship",
        base_model=BASE_QWEN_235B,
        worker_host="studio:9328",
    ),
    "ailiance-qwen-235b": AliasInventory(
        alias="ailiance-qwen-235b",
        base_model=BASE_QWEN_235B,
        worker_host="studio:9328",
    ),
    "ailiance-mixtral": AliasInventory(
        alias="ailiance-mixtral",
        base_model=BASE_MIXTRAL_8X22B,
        worker_host="studio:9329",
    ),
    "ailiance-mixtral-8x22b": AliasInventory(
        alias="ailiance-mixtral-8x22b",
        base_model=BASE_MIXTRAL_8X22B,
        worker_host="studio:9350",
    ),
    "ailiance-qwen36": AliasInventory(
        alias="ailiance-qwen36",
        base_model=BASE_QWEN3_6_35B,
        worker_host="studio:9305",
    ),
    # ----- macM1 mlx_lm.server :8502 (14 multi-model variants) -----
    "ailiance-gemma4": AliasInventory(
        alias="ailiance-gemma4",
        base_model=BASE_GEMMA_4_E4B,
        lora=("ailiance-curriculum",),
        worker_host="macm1:8502",
    ),
    "ailiance-gemma4-mascarade": AliasInventory(
        alias="ailiance-gemma4-mascarade",
        base_model=BASE_GEMMA_4_E4B,
        lora=("gemma4-e4b-mascarade",),
        worker_host="studio:9335",
    ),
    "ailiance-gemma4-aggro": AliasInventory(
        alias="ailiance-gemma4-aggro",
        base_model=BASE_GEMMA_4_E4B,
        lora=("gemma4-e4b-aggro",),
        worker_host="studio:9335",
    ),
    "ailiance-gemma4-kicad9plus": AliasInventory(
        alias="ailiance-gemma4-kicad9plus",
        base_model=BASE_GEMMA_4_E4B,
        lora=("gemma4-e4b-kicad9plus",),
        worker_host="studio:9335",
    ),
    "ailiance-gemma2": AliasInventory(
        alias="ailiance-gemma2",
        base_model=BASE_GEMMA_4_E2B,
        worker_host="macm1:8502",
    ),
    "ailiance-ministral": AliasInventory(
        alias="ailiance-ministral",
        base_model=BASE_MINISTRAL_3_14B,
        worker_host="macm1:8502",
    ),
    "ailiance-ministral-reasoning": AliasInventory(
        alias="ailiance-ministral-reasoning",
        base_model=BASE_MINISTRAL_3_14B_REASONING,
        worker_host="macm1:8502",
    ),
    "ailiance-granite": AliasInventory(
        alias="ailiance-granite",
        base_model=BASE_GRANITE_4_30B_MLX,
        worker_host="macm1:8502",
    ),
    "ailiance-qwen-mlx": AliasInventory(
        alias="ailiance-qwen-mlx",
        base_model=BASE_QWEN3_5_9B_MLX,
        worker_host="macm1:8502",
    ),
    # ----- Tower llama.cpp :9304 -----
    "ailiance-gemma": AliasInventory(
        alias="ailiance-gemma",
        base_model=BASE_GEMMA_3_4B,
        worker_host="tower:9304",
    ),
    # ----- kxkm-ai llama.cpp :18888 + :18889 (via SSH tunnel) -----
    "ailiance-qwen": AliasInventory(
        alias="ailiance-qwen",
        base_model=BASE_QWEN3_NEXT_80B,
        worker_host="kxkm-ai:18888 via electron-server:8002",
    ),
    "ailiance-granite-30b": AliasInventory(
        alias="ailiance-granite-30b",
        base_model=BASE_GRANITE_4_30B,
        worker_host="kxkm-ai:18889 via electron-server:8003",
    ),
    # ----- Tower Ollama :11434 — mascarade fine-tunes -----
    "ailiance-kicad": AliasInventory(
        alias="ailiance-kicad",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-kicad",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-spice": AliasInventory(
        alias="ailiance-spice",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-spice",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-stm32": AliasInventory(
        alias="ailiance-stm32",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-stm32",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-emc": AliasInventory(
        alias="ailiance-emc",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-emc",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-embedded": AliasInventory(
        alias="ailiance-embedded",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-embedded",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-platformio": AliasInventory(
        alias="ailiance-platformio",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-platformio",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-freecad": AliasInventory(
        alias="ailiance-freecad",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-freecad",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-dsp": AliasInventory(
        alias="ailiance-dsp",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-dsp",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-iot": AliasInventory(
        alias="ailiance-iot",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-iot",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-power": AliasInventory(
        alias="ailiance-power",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-power",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-components-review": AliasInventory(
        alias="ailiance-components-review",
        base_model=BASE_MASCARADE_4B,
        lora=("mascarade-components-review",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-coder": AliasInventory(
        alias="ailiance-coder",
        base_model="Qwen2.5-Coder-3B",
        lora=("mascarade-coder-v2",),
        worker_host="tower:11434 via electron-server:8004",
    ),
    "ailiance-embed": AliasInventory(
        alias="ailiance-embed",
        base_model=BASE_BGE_M3,
        worker_host="tower:11434 via electron-server:8004",
    ),
    # ----- Devstral Studio :9316-9321 (1 base + 5 LoRA hot-swap) -----
    "ailiance-devstral-base": AliasInventory(
        alias="ailiance-devstral-base",
        base_model=BASE_DEVSTRAL_24B,
        worker_host="studio:9316",
    ),
    "ailiance-python": AliasInventory(
        alias="ailiance-python",
        base_model=BASE_DEVSTRAL_24B,
        lora=("devstral-python",),
        worker_host="studio:9330",
    ),
    "ailiance-cpp": AliasInventory(
        alias="ailiance-cpp",
        base_model=BASE_DEVSTRAL_24B,
        lora=("devstral-cpp",),
        worker_host="studio:9330",
    ),
    "ailiance-rust-emb": AliasInventory(
        alias="ailiance-rust-emb",
        base_model=BASE_DEVSTRAL_24B,
        lora=("devstral-rust-emb",),
        worker_host="studio:9330",
    ),
    "ailiance-html": AliasInventory(
        alias="ailiance-html",
        base_model=BASE_DEVSTRAL_24B,
        lora=("devstral-html",),
        worker_host="studio:9330",
    ),
    "ailiance-ml-training": AliasInventory(
        alias="ailiance-ml-training",
        base_model=BASE_DEVSTRAL_24B,
        lora=("devstral-ml-training",),
        worker_host="studio:9330",
    ),
    # ----- Apertus Studio :9322 (1 base + 9 LoRA hot-swap) -----
    "ailiance-apertus-real": AliasInventory(
        alias="ailiance-apertus-real",
        base_model=BASE_APERTUS_70B,
        worker_host="studio:9322",
    ),
    "ailiance-apertus-electronics-hw": AliasInventory(
        alias="ailiance-apertus-electronics-hw",
        base_model=BASE_APERTUS_70B,
        lora=("apertus-electronics-hw",),
        worker_host="studio:9322",
    ),
    "ailiance-apertus-math-reasoning": AliasInventory(
        alias="ailiance-apertus-math-reasoning",
        base_model=BASE_APERTUS_70B,
        lora=("apertus-math-reasoning",),
        worker_host="studio:9322",
    ),
    "ailiance-apertus-math-gsm8k": AliasInventory(
        alias="ailiance-apertus-math-gsm8k",
        base_model=BASE_APERTUS_70B,
        lora=("apertus-math-gsm8k",),
        worker_host="studio:9322",
    ),
    "ailiance-apertus-math": AliasInventory(
        alias="ailiance-apertus-math",
        base_model=BASE_APERTUS_70B,
        lora=("apertus-math",),
        worker_host="studio:9322",
    ),
    "ailiance-apertus-security-fenrir": AliasInventory(
        alias="ailiance-apertus-security-fenrir",
        base_model=BASE_APERTUS_70B,
        lora=("apertus-security-fenrir",),
        worker_host="studio:9322",
    ),
    "ailiance-apertus-spice-sim": AliasInventory(
        alias="ailiance-apertus-spice-sim",
        base_model=BASE_APERTUS_70B,
        lora=("apertus-spice-sim",),
        worker_host="studio:9322",
    ),
    "ailiance-apertus-emc-dsp-power": AliasInventory(
        alias="ailiance-apertus-emc-dsp-power",
        base_model=BASE_APERTUS_70B,
        lora=("apertus-emc-dsp-power",),
        worker_host="studio:9322",
    ),
    "ailiance-apertus-embedded": AliasInventory(
        alias="ailiance-apertus-embedded",
        base_model=BASE_APERTUS_70B,
        lora=("apertus-embedded",),
        worker_host="studio:9322",
    ),
}


def served_model_for(*, alias: str, domain: str | None, worker_port: int) -> str:
    """Derive the model/adapter actually serving a routed request.

    Observability only (X-Ailiance-Served-Model header + audit stamp) —
    never used for routing. Never raises; returns "unknown" only when
    given no usable input.
    """
    if domain:
        if worker_port in _QWEN36_PORTS:
            adapter = DOMAIN_TO_QWEN36.get(domain)
            if adapter:
                return adapter
        elif worker_port == OMLX_PORT:
            model = DOMAIN_TO_OMLX_MODEL.get(domain)
            if model:
                return model
    inv = _REGISTRY.get(alias)
    if inv and inv.base_model:
        return inv.base_model
    return alias or "unknown"


def get_alias_inventory(alias: str) -> AliasInventory | None:
    """Return registry entry for ``alias`` or ``None`` if unknown."""
    return _REGISTRY.get(alias)


def inventory_or_unknown(alias: str | None) -> AliasInventory:
    """Always return an :class:`AliasInventory`; unknowns get a placeholder.

    Used by the response stamper which wants to emit *something* even
    when the registry is out of date — better to ship an ``unknown``
    label than to skip observability entirely.
    """
    if alias and alias in _REGISTRY:
        return _REGISTRY[alias]
    return AliasInventory(
        alias=alias or "unknown",
        base_model="unknown",
        lora=(),
        worker_host=None,
    )


def to_dict(inv: AliasInventory) -> dict:
    """Serialise for the response body's ``ailiance`` field."""
    return {
        "alias": inv.alias,
        "base_model": inv.base_model,
        "lora": list(inv.lora),
        "worker_host": inv.worker_host,
    }


def to_headers(inv: AliasInventory) -> dict[str, str]:
    """Serialise for HTTP response headers (X-Ailiance-*)."""
    headers = {
        "X-Ailiance-Alias": inv.alias,
        "X-Ailiance-Base-Model": inv.base_model,
    }
    if inv.lora:
        headers["X-Ailiance-LoRA"] = ",".join(inv.lora)
    if inv.worker_host:
        headers["X-Ailiance-Worker-Host"] = inv.worker_host
    return headers


def known_aliases() -> frozenset[str]:
    """Read-only view of registered alias names. Used by tests."""
    return frozenset(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Routing → alias resolution
# ---------------------------------------------------------------------------

# Reverse: classifier domain → canonical ailiance-* alias. Lets the
# auto-router (req.model == "ailiance") expose the *actually served*
# alias in response headers/body instead of the generic "ailiance".
#
# Mascarade and Apertus aliases follow a strict naming convention
# (``ailiance-<domain>`` / ``ailiance-apertus-<domain>``) so the map
# below only enumerates the non-trivial cases.
_DOMAIN_TO_ALIAS_EXPLICIT: dict[str, str] = {}  # consolidated to omlx 2026-05-29

# Domains served by the mascarade Tower Ollama ensemble. The alias is
# always ``ailiance-<domain>`` so no enumeration needed; we just check
# membership.
_MASCARADE_DOMAINS: frozenset[str] = frozenset()  # consolidated to omlx 2026-05-29


def resolve_effective_alias(
    req_model: str,
    *,
    cascade_alias: str | None = None,
    domain: str | None = None,
) -> str:
    """Determine the alias actually served by a routed request.

    Resolution order (first match wins):

    1. ``cascade_alias`` — explicit complexity-cascade override.
    2. ``req_model`` if it's a concrete ailiance-* alias (not the
       auto-router pseudo-alias ``ailiance``).
    3. Auto-router (``req_model == "ailiance"``):
       a. ``ailiance-<domain>`` for mascarade domains.
       b. The explicit domain → alias map for specialised routes.
       c. ``ailiance-mistral-medium`` as the general-purpose default.
       d. The raw ``"ailiance"`` literal as last resort (also covers
          requests with no classifier domain at all).

    Returns a string that is *always* a valid registry key for known
    aliases, or the raw req_model when no rewrite applies.
    """
    if cascade_alias:
        return cascade_alias
    if req_model != "ailiance":
        return req_model
    # Auto-router: lift to the actually-served alias.
    if domain:
        if domain in _MASCARADE_DOMAINS:
            candidate = f"ailiance-{domain}"
            if candidate in _REGISTRY:
                return candidate
        explicit = _DOMAIN_TO_ALIAS_EXPLICIT.get(domain)
        if explicit and explicit in _REGISTRY:
            return explicit
    return "ailiance"
