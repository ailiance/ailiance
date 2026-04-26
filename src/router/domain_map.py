# src/router/domain_map.py
"""Static mapping of domains to worker ports.

Apertus (:9201) — reasoning, hardware, EU normative
Devstral (:9202) — code generation
EuroLLM  (:9203) — multilingual EU
"""

APERTUS_PORT = 9201
DEVSTRAL_PORT = 9202
EUROLLM_PORT = 9203

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

ALL_DOMAINS = APERTUS_DOMAINS | DEVSTRAL_DOMAINS | EUROLLM_DOMAINS

DOMAIN_TO_WORKER: dict[str, int] = {}
for d in APERTUS_DOMAINS:
    DOMAIN_TO_WORKER[d] = APERTUS_PORT
for d in DEVSTRAL_DOMAINS:
    DOMAIN_TO_WORKER[d] = DEVSTRAL_PORT
for d in EUROLLM_DOMAINS:
    DOMAIN_TO_WORKER[d] = EUROLLM_PORT


def get_worker_for_domain(domain: str) -> int | None:
    return DOMAIN_TO_WORKER.get(domain)
