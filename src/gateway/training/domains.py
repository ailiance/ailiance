"""medium35 campaign domain order and verified 3-phase hyperparameters."""
from __future__ import annotations

# Hardware-first order. emc-dsp-power (crashed iter 1) and embedded (phase 2
# done) are incomplete; the resume sentinels phaseN_done rejoin them at the
# right phase, so they sit naturally in the hardware tier.
HARDWARE_DOMAINS: tuple[str, ...] = (
    "kicad-dsl", "kicad-pcb", "platformio", "rust-embedded", "spice-sim",
    "iot", "freecad", "emc-dsp-power", "embedded",
)
GENERAL_DOMAINS: tuple[str, ...] = (
    "python", "rust", "typescript", "web-backend", "web-frontend", "sql",
    "shell", "yaml-json", "html-css", "lua-upy", "ml-training", "llm-ops",
    "llm-orch", "math-gsm8k", "math-reasoning", "music-audio",
    "multilingual-eu", "traduction-tech", "security-fenrir",
)
CAMPAIGN_DOMAINS: tuple[str, ...] = HARDWARE_DOMAINS + GENERAL_DOMAINS

# Already trained healthily (2026-05-19) — never re-run.
DONE_DOMAINS: frozenset[str] = frozenset({"chat-fr", "cpp", "docker-devops"})

# Verified against the 3 healthy medium35 runs (config-phaseN.yaml +
# adapter_config.json, 2026-05-19). MLX-LM 0.31.2 reads only rank/scale/
# dropout; the inert `alpha: 32` field is not set here.
LORA_RANK = 16
LORA_SCALE = 32.0
LORA_DROPOUT = 0.01
NUM_LAYERS = -1  # all layers

HOURS_PER_DOMAIN = 17  # measured on medium35-cpp

# Iterations per curriculum phase (medium35_train_domain.sh PHASE table).
PHASE_ITERS: dict[int, int] = {1: 500, 2: 800, 3: 500}
