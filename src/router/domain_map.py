# src/router/domain_map.py
"""Static mapping of domains to worker ports.

Apertus       (:9301) — hardware, EU normative, math, music
Qwen3-Coder   (:9327) — code generation (Studio MLX, MoE 4-bit; replaces
                        Devstral :9302 decommissioned 2026-05-10)
EuroLLM       (:9303) — multilingual EU
Gemma         (:9304) — quick / fallback / generalist short
DeepSeek-R1   (:9323) — reasoning (Studio MLX 32B 4-bit local; replaces
                        Qwen3-Next :8002 kxkm-ai tunnel, unreliable)
Mascarade     (:9340) — domain LoRA specialists, MLX bf16 on Studio
eu-kiki       (:8502) — Gemma-4 E4B + ailiance curriculum LoRA on macm1

Studio MLX worker ports declared but not yet wired into DOMAIN_TO_WORKER
(reachable via direct model alias only — see ALIAS_MODEL_REWRITES in
server.py): FLAGSHIP_PORT (9328), MIXTRAL_PORT (9329), LLAMA_PORT (9324),
MISTRAL_SMALL_PORT (9326), PIXTRAL_PORT (9325).
"""

APERTUS_PORT = 9301
# Devstral :9302 decommissioned 2026-05-10 — kept as named constant for
# audit/grep, but no domain maps here. DEVSTRAL_DOMAINS now route to
# QWEN_CODER_PORT (Studio Qwen3-Coder-30B MoE 4-bit).
DEVSTRAL_PORT = 9302  # DEAD — do not route to this port
EUROLLM_PORT = 9303
GEMMA_PORT = 9304
QWEN_PORT = 8002  # Qwen3-Next 80B (kxkm-ai tunnel) — unreliable, see below
# Studio MLX :9340 — the 10 qwen3-4b-mascarade experts merged into
# Qwen3-4B-Instruct-2507 and served as MLX bf16. Replaces the Tower
# Ollama :8004 Q4_K_M path: same fine-tunes, no quantization loss.
MASCARADE_PORT = 9340
# macm1 mlx_lm.server (alias `ailiance-gemma4`): Gemma-4 E4B + eu-kiki
# ailiance curriculum LoRA. Bench ailiance/ailiance-bench Phase 6
# (commit 46801af 2026-05-11) confirms this is the P1 generation champion:
# kicad-dsl +55 pts, kicad-pcb +42 pts vs prior fallbacks.
AILIANCE_MACM1_PORT = 8502

# Studio MLX workers (post-2026-05-12). Live on MacStudio M3 Ultra,
# served by mlx_lm.server. All 4-bit quantized.
QWEN_CODER_PORT = 9327       # Qwen3-Coder-30B-A3B MoE 4-bit
DEEPSEEK_R1_PORT = 9323      # DeepSeek-R1-Distill-Qwen-32B 4-bit
FLAGSHIP_PORT = 9328         # Qwen3-235B-A22B MoE 4-bit (max capability)
MIXTRAL_PORT = 9329          # Mixtral-8x22B-Instruct 4-bit
LLAMA_PORT = 9324            # Llama-3.3-70B-Instruct 4-bit
PIXTRAL_PORT = 9325          # Pixtral-12B 4-bit (multimodal, not yet wired)
MISTRAL_SMALL_PORT = 9326    # Mistral-Small-3.1-24B 4-bit

OMLX_PORT = 8500  # consolidated omlx multi-model server (Tailscale 100.116.92.12:8500)

# Math stays on Apertus (faster, sufficient for routine maths).
# `reasoning` moved off Apertus to a dedicated reasoning worker — see
# QWEN_DOMAINS / QWEN_LIVE below.
APERTUS_DOMAINS = frozenset({
    "electronics-hw", "emc", "dsp", "spice", "kicad", "stm32",
    "platformio", "iot", "embedded", "math",
    "security", "music-audio", "freecad", "power",
    "misra-c", "autosar-cert", "doc-technique-ce",
    "calcul-normatif", "normes-iec",
})

# `reasoning` — was Qwen3-Next 80B (:8002, kxkm-ai tunnel). Tunnel observed
# unreliable (60s timeout 2026-05-12). Reroute to DeepSeek-R1-Distill-Qwen-32B
# on Studio :9323 (local MLX, no tunnel hop). Flip QWEN_LIVE=True to restore
# the 80B sparse MoE if the tunnel stabilises.
QWEN_LIVE = False
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

# Mascarade LoRA specialists: each domain LoRA merged into
# Qwen3-4B-Instruct-2507 and served as MLX bf16 on the Mac Studio
# (:9340), reached via autossh tunnel localhost:9340. Replaces the
# former Tower Ollama Q4_K_M path — same fine-tunes, no quantization.
# Priority: MASCARADE > APERTUS for these labels (override below).
# Apertus stays the fallback when the Studio worker is down (server.py
# retry path) or when classifier confidence is below threshold.
MASCARADE_DOMAINS = frozenset({
    "kicad", "stm32", "emc", "embedded",
    "platformio", "freecad", "dsp", "iot", "power",
})
# Note 2026-05-11: `spice` removed from mascarade override after bench
# Phase 6 (ailiance/ailiance-bench commit 46801af) showed mascarade-spice
# regresses -25 pts on spice-sim vs base Gemma-E4B. Apertus Mistral-Medium
# 128B (generalist fallback) outperforms mascarade-spice on this task.

# eu-kiki P1 champion domains: Gemma-4 E4B + ailiance curriculum LoRA
# on macm1 :8502. Bench Phase 6 (commit 46801af) shows eu-kiki wins
# P1 generation decisively (+55 kicad-dsl, +42 kicad-pcb) against both
# Mascarade-kicad (0 lift on generation) and Apertus (prior PCB target).
# These are *generation* labels — distinct from mascarade-kicad which
# wins only on P3 extraction (+48). Override applied last (last-write-wins).
AILIANCE_MACM1_DOMAINS = frozenset({"kicad-dsl", "kicad-pcb"})

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
    # NOTE: kicad-pcb was previously aliased to "kicad" (→ Mascarade :8004).
    # Removed 2026-05-11 — kicad-pcb is now a first-class EUKIKI_DOMAIN
    # routing to macm1 :8502 (bench Phase 6 shows +42 pts vs mascarade).
    "ml": "ml-training",
    "embedded-c": "embedded",
}

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

ALL_DOMAINS = (
    APERTUS_DOMAINS | DEVSTRAL_DOMAINS | EUROLLM_DOMAINS | GEMMA_DOMAINS
    | QWEN_DOMAINS | AILIANCE_MACM1_DOMAINS
)
# Sanity: MASCARADE_DOMAINS must be a subset of APERTUS (we override, not extend)
assert MASCARADE_DOMAINS <= APERTUS_DOMAINS, (
    "MASCARADE_DOMAINS must be a subset of APERTUS_DOMAINS "
    f"(extra: {MASCARADE_DOMAINS - APERTUS_DOMAINS})"
)
# Sanity: AILIANCE_MACM1_DOMAINS must be disjoint from MASCARADE and APERTUS core
# (these are *new* labels, not overrides of existing ones).
assert AILIANCE_MACM1_DOMAINS.isdisjoint(APERTUS_DOMAINS), (
    "AILIANCE_MACM1_DOMAINS must be disjoint from APERTUS_DOMAINS "
    f"(overlap: {AILIANCE_MACM1_DOMAINS & APERTUS_DOMAINS})"
)

DOMAIN_TO_WORKER: dict[str, int] = {}
for d in APERTUS_DOMAINS:
    DOMAIN_TO_WORKER[d] = APERTUS_PORT
# Code domains: route to Studio Qwen3-Coder-30B MoE 4-bit (:9327).
# Devstral :9302 decommissioned 2026-05-10. For LoRA-specific code
# routing (python/cpp/rust-emb/html-css/ml-training), use direct model
# alias resolved by ALIAS_MODEL_REWRITES → devstral multi-LoRA :9330.
for d in DEVSTRAL_DOMAINS:
    DOMAIN_TO_WORKER[d] = QWEN_CODER_PORT
# EuroLLM-22B removed from Studio fleet 2026-05-12 (RAM-expensive,
# ~22 GB warm). The 4 EUROLLM_DOMAINS (chat-fr, traduction-tech,
# redaction-multilingue, localisation-doc) now route to Apertus
# (Mistral-Medium-128B Q8 :9301) which handles FR-strong tasks well.
# EUROLLM_PORT constant kept for clarity but no longer wired.
for d in EUROLLM_DOMAINS:
    DOMAIN_TO_WORKER[d] = APERTUS_PORT
for d in GEMMA_DOMAINS:
    DOMAIN_TO_WORKER[d] = GEMMA_PORT
# Reasoning: prefer Qwen3-Next 80B (:8002, kxkm-ai tunnel) when QWEN_LIVE,
# else DeepSeek-R1-Distill-Qwen-32B on Studio :9323 (local, stable).
_reasoning_target = QWEN_PORT if QWEN_LIVE else DEEPSEEK_R1_PORT
for d in QWEN_DOMAINS:
    DOMAIN_TO_WORKER[d] = _reasoning_target
# Override APERTUS for the 10 mascarade-specialized domains.
for d in MASCARADE_DOMAINS:
    DOMAIN_TO_WORKER[d] = MASCARADE_PORT
# eu-kiki P1 KiCad generation domains — must come LAST (last-write-wins).
# kicad-dsl and kicad-pcb are first-class labels (not aliased); the
# kicad-pcb alias to "kicad" was removed to prevent the alias resolution
# from bypassing this override in get_worker_for_domain().
for d in AILIANCE_MACM1_DOMAINS:
    DOMAIN_TO_WORKER[d] = AILIANCE_MACM1_PORT

# Consolidation 2026-05-29: every domain with an omlx specialist routes to
# the single omlx :8500 server; the per-domain model is resolved in server.py
# via DOMAIN_TO_OMLX_MODEL. Legacy per-port workers are dead.
for _d in DOMAIN_TO_OMLX_MODEL:
    DOMAIN_TO_WORKER[_d] = OMLX_PORT


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

    For domains in MASCARADE_DOMAINS, route to the Studio MLX worker
    (:9340) only when confidence is >= mascarade_min_score. Below threshold
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

QWEN36_PORT = 9360  # multi-LoRA Qwen3.6-35B server (256k ctx) on Studio
QWEN36_PORT_B = 9361  # second multi-LoRA instance (load split, less adapter thrash)
# Domains served by instance B (code / web / devops / ml / language). The rest
# of DOMAIN_TO_QWEN36 stays on instance A (9360, hardware / EDA / math).
QWEN36_DOMAINS_B: frozenset[str] = frozenset({
    "cpp", "rust", "typescript", "shell", "sql", "html-css",
    "web-backend", "web-frontend", "docker", "devops", "yaml-json",
    "llm-ops", "llm-orch", "ml-training", "lua-upy",
    "chat-fr", "traduction-tech", "redaction-multilingue", "localisation-doc",
})

# router domain -> qwen36 adapter name served by the :9360 multi-LoRA server.
# Excludes spice (numerically unreliable) and kicad-pcb (broken output) -> those
# stay on omlx via DOMAIN_TO_OMLX_MODEL. python/reasoning/generalists also stay on omlx.
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
for _d in DOMAIN_TO_QWEN36:
    DOMAIN_TO_WORKER[_d] = QWEN36_PORT_B if _d in QWEN36_DOMAINS_B else QWEN36_PORT
