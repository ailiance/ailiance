# src/router/domain_map.py
"""Static mapping of domains to worker ports.

Apertus (:9301) — reasoning, hardware, EU normative
Devstral (:9302) — code generation
EuroLLM  (:9303) — multilingual EU
Gemma    (:9304) — quick / fallback / multilingual short
"""

APERTUS_PORT = 9301
DEVSTRAL_PORT = 9302
EUROLLM_PORT = 9303
GEMMA_PORT = 9304

APERTUS_DOMAINS = frozenset({
    "electronics-hw", "emc", "dsp", "spice", "kicad", "stm32",
    "platformio", "iot", "embedded", "math", "reasoning",
    "security", "music-audio", "freecad", "power",
    "misra-c", "autosar-cert", "doc-technique-ce",
    "calcul-normatif", "normes-iec",
})

DEVSTRAL_DOMAINS = frozenset({
    "python", "rust", "typescript", "cpp", "shell", "html-css",
    "sql", "web-backend", "web-frontend", "docker", "devops",
    "yaml-json", "llm-ops", "llm-orch", "ml-training", "lua-upy",
})

EUROLLM_DOMAINS = frozenset({
    "chat-fr", "traduction-tech", "redaction-multilingue", "localisation-doc",
})

# Gemma 3 4B IT lives on tower as the quick / generalist worker.
# Used for short prompts, summaries, classification, and the default
# fallback when the router can't confidently match a labeled domain.
GEMMA_DOMAINS = frozenset({
    "general", "quick", "summarize", "classification", "tldr",
})

# Aliases for label drift between training and runtime: the router was
# trained on slightly different surface forms than DOMAIN_TO_WORKER keys.
# Map each known synonym → canonical domain. Updated 2026-05-05.
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
    "kicad-pcb": "kicad",
    "ml": "ml-training",
    "embedded-c": "embedded",
}

ALL_DOMAINS = APERTUS_DOMAINS | DEVSTRAL_DOMAINS | EUROLLM_DOMAINS | GEMMA_DOMAINS

DOMAIN_TO_WORKER: dict[str, int] = {}
for d in APERTUS_DOMAINS:
    DOMAIN_TO_WORKER[d] = APERTUS_PORT
for d in DEVSTRAL_DOMAINS:
    DOMAIN_TO_WORKER[d] = DEVSTRAL_PORT
for d in EUROLLM_DOMAINS:
    DOMAIN_TO_WORKER[d] = EUROLLM_PORT
for d in GEMMA_DOMAINS:
    DOMAIN_TO_WORKER[d] = GEMMA_PORT


def get_worker_for_domain(domain: str | None) -> int | None:
    """Resolve a domain label (with alias) to its worker port."""
    if not domain:
        return None
    canonical = DOMAIN_ALIASES.get(domain, domain)
    return DOMAIN_TO_WORKER.get(canonical)
