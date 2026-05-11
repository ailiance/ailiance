# src/router/domain_map.py
"""Static mapping of domains to worker ports.

Apertus    (:9301) — hardware, EU normative, math, music
Devstral   (:9302) — code generation
EuroLLM    (:9303) — multilingual EU
Gemma      (:9304) — quick / fallback / generalist short
Qwen3-Next (:8002) — reasoning (80B sparse MoE, served via tunnel to kxkm-ai)
Mascarade  (:8004) — domain LoRA specialists via Tower Ollama tunnel
"""

APERTUS_PORT = 9301
DEVSTRAL_PORT = 9302
EUROLLM_PORT = 9303
GEMMA_PORT = 9304
QWEN_PORT = 8002
MASCARADE_PORT = 8004

# `reasoning` moved off Apertus so the 80B sparse MoE handles complex
# multi-step reasoning — strictly more capable on benchmarks like GSM8K /
# AIME / MMLU-Pro reasoning subsets at the cost of ~3x lower throughput
# (CPU-side MoE expert offload). Math stays on Apertus (faster, sufficient
# for routine maths).
APERTUS_DOMAINS = frozenset({
    "electronics-hw", "emc", "dsp", "spice", "kicad", "stm32",
    "platformio", "iot", "embedded", "math",
    "security", "music-audio", "freecad", "power",
    "misra-c", "autosar-cert", "doc-technique-ce",
    "calcul-normatif", "normes-iec",
})

QWEN_DOMAINS = frozenset({"reasoning"})

DEVSTRAL_DOMAINS = frozenset({
    "python", "rust", "typescript", "cpp", "shell", "html-css",
    "sql", "web-backend", "web-frontend", "docker", "devops",
    "yaml-json", "llm-ops", "llm-orch", "ml-training", "lua-upy",
})

EUROLLM_DOMAINS = frozenset({
    "chat-fr", "traduction-tech", "redaction-multilingue", "localisation-doc",
})

# Gemma 3 4B IT lives on tower as the quick / generalist worker.
GEMMA_DOMAINS = frozenset({
    "general", "quick", "summarize", "classification", "tldr",
})

# Mascarade LoRA specialists (Qwen3 4B Q4_K_M base) on Tower Ollama,
# reachable via autossh tunnel localhost:8004 -> tower:11434.
# Each domain has a 1:1 mapping to a mascarade-<domain>:latest LoRA.
# Priority: MASCARADE > APERTUS for these 10 labels (override below).
# Rationale: ~20x throughput advantage (Tower 4B Q4 ~80 tok/s vs Studio
# Mistral-Medium 128B ~3 tok/s) AND domain-fine-tuned quality wins on
# narrow technical tasks. Apertus stays the fallback when Tower is down
# (server.py retry path) or when confidence is below threshold.
MASCARADE_DOMAINS = frozenset({
    "kicad", "spice", "stm32", "emc", "embedded",
    "platformio", "freecad", "dsp", "iot", "power",
})

# Aliases for label drift between training and runtime: the router was
# trained on slightly different surface forms than DOMAIN_TO_WORKER keys.
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

ALL_DOMAINS = (
    APERTUS_DOMAINS | DEVSTRAL_DOMAINS | EUROLLM_DOMAINS | GEMMA_DOMAINS
    | QWEN_DOMAINS
)
# Sanity: MASCARADE_DOMAINS must be a subset of APERTUS (we override, not extend)
assert MASCARADE_DOMAINS <= APERTUS_DOMAINS, (
    "MASCARADE_DOMAINS must be a subset of APERTUS_DOMAINS "
    f"(extra: {MASCARADE_DOMAINS - APERTUS_DOMAINS})"
)

DOMAIN_TO_WORKER: dict[str, int] = {}
for d in APERTUS_DOMAINS:
    DOMAIN_TO_WORKER[d] = APERTUS_PORT
for d in DEVSTRAL_DOMAINS:
    DOMAIN_TO_WORKER[d] = DEVSTRAL_PORT
# EuroLLM (:9303) on Studio is DOWN as of 2026-05-11 (plist refuses
# bootstrap via SSH, requires GUI session). Temporarily reroute all
# EUROLLM_DOMAINS to Gemma (:9304, Tower llama.cpp) which is the closest
# fit for short chat-fr / translation prompts. Revert by flipping the
# next loop back to EUROLLM_PORT once :9303 is healthy.
EUROLLM_LIVE = False  # set to True when Studio :9303 is back up
_eurollm_target = EUROLLM_PORT if EUROLLM_LIVE else GEMMA_PORT
for d in EUROLLM_DOMAINS:
    DOMAIN_TO_WORKER[d] = _eurollm_target
for d in GEMMA_DOMAINS:
    DOMAIN_TO_WORKER[d] = GEMMA_PORT
for d in QWEN_DOMAINS:
    DOMAIN_TO_WORKER[d] = QWEN_PORT
# Override APERTUS for the 10 mascarade-specialized domains.
# Must come LAST so dict-write-last-wins selects Tower over Studio.
for d in MASCARADE_DOMAINS:
    DOMAIN_TO_WORKER[d] = MASCARADE_PORT


# Minimum classifier score to route to a Mascarade specialist. Below this
# threshold we fall back to the bigger Apertus generalist, which is more
# robust on ambiguous prompts. Tuned conservatively at 0.85 to start —
# observed top-1 scores for clear-domain prompts are typically >0.99.
MASCARADE_MIN_CONFIDENCE = 0.85


def get_worker_for_domain(domain: str | None) -> int | None:
    """Resolve a domain label (with alias) to its worker port."""
    if not domain:
        return None
    canonical = DOMAIN_ALIASES.get(domain, domain)
    return DOMAIN_TO_WORKER.get(canonical)


def get_worker_for_domain_with_confidence(
    domain: str | None,
    score: float,
    *,
    mascarade_min_score: float = MASCARADE_MIN_CONFIDENCE,
) -> int | None:
    """Resolve domain → worker port with confidence-gated Mascarade routing.

    For domains in MASCARADE_DOMAINS, route to Tower Ollama (:8004) only
    when classifier confidence is >= mascarade_min_score. Below threshold
    we fall back to APERTUS (Mistral-Medium 128B on Studio), which is more
    forgiving on ambiguous prompts at the cost of ~20x lower throughput.

    Non-Mascarade domains use the static DOMAIN_TO_WORKER mapping
    unchanged (this preserves the legacy contract for tests/callers
    that don't yet pass scores).
    """
    if not domain:
        return None
    canonical = DOMAIN_ALIASES.get(domain, domain)
    if canonical in MASCARADE_DOMAINS and score < mascarade_min_score:
        return APERTUS_PORT
    return DOMAIN_TO_WORKER.get(canonical)
