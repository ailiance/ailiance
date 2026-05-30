# src/router/domain_map.py
"""Static mapping of auto-router domains to backends.

Since the 2026-05-29 consolidation, every domain is served by one of two
backends on the MacStudio omlx node (Tailscale 100.116.92.12):

- **omlx :8500** — consolidated multi-model server. The per-domain model is
  resolved via ``DOMAIN_TO_OMLX_MODEL`` (apertus-70b specialists, Qwen3-Coder,
  DeepSeek-R1, EuroLLM, gemma-4, Mistral-Small, fused gemma-4 LoRA…).
- **qwen36 :9360 / :9361** — two instances of the multi-LoRA Qwen3.6-35B
  server (256k ctx, hot-swap adapters). The per-domain adapter is resolved via
  ``DOMAIN_TO_QWEN36``; the :9361 split (``QWEN36_DOMAINS_B``) carries the
  code/web/devops/language domains to cut adapter thrash.

``DOMAIN_TO_WORKER`` (domain → port) is derived from those two maps and the
server.py override cascade picks the concrete model/adapter:
``ALIAS_MODEL_REWRITES > qwen36 (9360/9361) > omlx (8500)``.

All the former per-port workers (Apertus :9301, Devstral :9302, EuroLLM :9303,
Gemma :9304, Mascarade :9340, eu-kiki macm1 :8502, the Studio per-model ports
9323-9329) are no longer routed to — their LoRA/specialists are fused into the
omlx + qwen36 backends. Removed here 2026-05-30 (were dead since the
consolidation; every domain is overwritten to omlx/qwen36).
"""

OMLX_PORT = 8500  # consolidated omlx multi-model server (Tailscale 100.116.92.12:8500)
QWEN36_PORT = 9360  # multi-LoRA Qwen3.6-35B server (256k ctx) on Studio
QWEN36_PORT_B = 9361  # second multi-LoRA instance (load split, less adapter thrash)

# Aliases for label drift between training and runtime: the router was
# trained on slightly different surface forms than the domain keys below.
DOMAIN_ALIASES: dict[str, str] = {
    # Hardware family
    "electronics": "electronics-hw",
    "electronique": "electronics-hw",
    "hardware": "electronics-hw",
    "hw": "electronics-hw",
    "elec": "electronics-hw",
    # Multilingual / chat (router often emits these short forms)
    "translation": "traduction-tech",
    "traduction": "traduction-tech",
    "fr": "chat-fr",
    "francais": "chat-fr",
    "french": "chat-fr",
    "multilingual": "redaction-multilingue",
    # Code variants
    "ts": "typescript",
    "js": "typescript",
    "javascript": "typescript",
    "c++": "cpp",
    "py": "python",
    "bash": "shell",
    "sh": "shell",
    # Misc
    # NOTE: kicad-pcb was previously aliased to "kicad". Removed 2026-05-11 —
    # kicad-pcb is a first-class label with its own omlx model.
    "ml": "ml-training",
    "embedded-c": "embedded",
}

# Domain → omlx model name (resolved by server.py when worker_port == OMLX_PORT).
# This is the canonical domain set: ALL_DOMAINS is derived from its keys.
DOMAIN_TO_OMLX_MODEL: dict[str, str] = {
    "electronics-hw": "apertus-70b-electronics-hw-8bit",
    "embedded": "apertus-70b-embedded-8bit",
    "stm32": "apertus-70b-embedded-8bit",
    "platformio": "apertus-70b-embedded-8bit",
    "iot": "apertus-70b-embedded-8bit",
    "misra-c": "apertus-70b-embedded-8bit",
    "autosar-cert": "apertus-70b-embedded-8bit",
    "emc": "apertus-70b-emc-dsp-power-8bit",
    "dsp": "apertus-70b-emc-dsp-power-8bit",
    "power": "apertus-70b-emc-dsp-power-8bit",
    "spice": "apertus-70b-spice-sim-8bit",
    "security": "apertus-70b-security-fenrir-8bit",
    "math": "apertus-70b-math-8bit",
    "calcul-normatif": "apertus-70b-math-8bit",
    "freecad": "apertus-70b-electronics-hw-8bit",
    "kicad": "gemma-4-e4b-mascarade-fused",
    "kicad-dsl": "gemma-4-e4b-eukiki-fused",
    "kicad-pcb": "gemma-4-e4b-eukiki-fused",
    "reasoning": "DeepSeek-R1-Distill-Qwen-32B-MLX-4bit",
    "python": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "cpp": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "rust": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "typescript": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "shell": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "html-css": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "sql": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "web-backend": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "web-frontend": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "docker": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "devops": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "yaml-json": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "llm-ops": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "llm-orch": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "ml-training": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "lua-upy": "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "chat-fr": "EuroLLM-22B-Instruct-2512",
    "traduction-tech": "EuroLLM-22B-Instruct-2512",
    "redaction-multilingue": "EuroLLM-22B-Instruct-2512",
    "localisation-doc": "EuroLLM-22B-Instruct-2512",
    "general": "gemma-4-E4B-it-MLX-4bit",
    "quick": "gemma-4-E4B-it-MLX-4bit",
    "summarize": "gemma-4-E4B-it-MLX-4bit",
    "classification": "gemma-4-E4B-it-MLX-4bit",
    "tldr": "gemma-4-E4B-it-MLX-4bit",
    "normes-iec": "Mistral-Small-3.1-24B-Instruct-MLX-4bit",
    "doc-technique-ce": "Mistral-Small-3.1-24B-Instruct-MLX-4bit",
    "music-audio": "Mistral-Small-3.1-24B-Instruct-MLX-4bit",
}

# router domain -> qwen36 adapter name served by the :9360 / :9361 multi-LoRA
# servers. Subset of DOMAIN_TO_OMLX_MODEL keys: these domains route to qwen36
# (overriding omlx). Excludes spice (numerically unreliable adapter) and
# kicad-pcb (broken output) -> those keep their omlx model. python / reasoning /
# generalists also stay on omlx.
DOMAIN_TO_QWEN36: dict[str, str] = {
    "emc": "qwen36-emc-dsp-power",
    "dsp": "qwen36-emc-dsp-power",
    "power": "qwen36-emc-dsp-power",
    "stm32": "qwen36-embedded",
    "embedded": "qwen36-embedded",
    "electronics-hw": "qwen36-embedded",
    "misra-c": "qwen36-embedded",
    "platformio": "qwen36-platformio",
    "iot": "qwen36-iot",
    "freecad": "qwen36-freecad",
    "security": "qwen36-security-fenrir",
    "math": "qwen36-math-reasoning",
    "calcul-normatif": "qwen36-math-reasoning",
    "kicad": "qwen36-kicad-dsl",
    "kicad-dsl": "qwen36-kicad-dsl",
    "cpp": "qwen36-cpp",
    "rust": "qwen36-rust",
    "typescript": "qwen36-typescript",
    "shell": "qwen36-shell",
    "sql": "qwen36-sql",
    "html-css": "qwen36-html-css",
    "web-backend": "qwen36-web-backend",
    "web-frontend": "qwen36-web-frontend",
    "docker": "qwen36-docker-devops",
    "devops": "qwen36-docker-devops",
    "yaml-json": "qwen36-yaml-json",
    "llm-ops": "qwen36",  # adapter leaks URLs -> base
    "llm-orch": "qwen36",  # adapter leaks URLs -> base
    "ml-training": "qwen36-ml-training",
    "lua-upy": "qwen36-lua-upy",
    "chat-fr": "qwen36-chat-fr",
    "traduction-tech": "qwen36-traduction-tech",
    "redaction-multilingue": "qwen36-multilingual-eu",
    "localisation-doc": "qwen36-multilingual-eu",
    "music-audio": "qwen36-music-audio",
    "spice": "qwen36",  # spice adapter ruins RC math; base correct
}

# Domains served by qwen36 instance B (:9361) — code / web / devops / ml /
# language. The rest of DOMAIN_TO_QWEN36 stays on instance A (:9360, hardware /
# EDA / math).
QWEN36_DOMAINS_B: frozenset[str] = frozenset({
    "cpp", "rust", "typescript", "shell", "sql", "html-css",
    "web-backend", "web-frontend", "docker", "devops", "yaml-json",
    "llm-ops", "llm-orch", "ml-training", "lua-upy",
    "chat-fr", "traduction-tech", "redaction-multilingue", "localisation-doc",
})

# The canonical 47-domain set the auto-router classifier predicts over.
ALL_DOMAINS = frozenset(DOMAIN_TO_OMLX_MODEL)

# Sanity: every qwen36-routed domain must have an omlx model to fall back to
# (qwen36 overrides omlx, never introduces a new label).
assert set(DOMAIN_TO_QWEN36) <= set(DOMAIN_TO_OMLX_MODEL), (
    "DOMAIN_TO_QWEN36 must be a subset of DOMAIN_TO_OMLX_MODEL "
    f"(extra: {set(DOMAIN_TO_QWEN36) - set(DOMAIN_TO_OMLX_MODEL)})"
)
assert QWEN36_DOMAINS_B <= set(DOMAIN_TO_QWEN36), (
    "QWEN36_DOMAINS_B must be a subset of DOMAIN_TO_QWEN36 "
    f"(extra: {QWEN36_DOMAINS_B - set(DOMAIN_TO_QWEN36)})"
)

# Domain -> worker port. Built from the two live backends: omlx for every
# domain, then the qwen36-routed subset overrides to :9360 / :9361.
DOMAIN_TO_WORKER: dict[str, int] = {d: OMLX_PORT for d in DOMAIN_TO_OMLX_MODEL}
for _d in DOMAIN_TO_QWEN36:
    DOMAIN_TO_WORKER[_d] = QWEN36_PORT_B if _d in QWEN36_DOMAINS_B else QWEN36_PORT


def get_worker_for_domain(domain: str | None) -> int | None:
    """Resolve a domain label (with alias) to its worker port."""
    if not domain:
        return None
    canonical = DOMAIN_ALIASES.get(domain, domain)
    return DOMAIN_TO_WORKER.get(canonical)
