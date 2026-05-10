#!/usr/bin/env python3
"""
AILIANCE Evaluation Framework — v1 (Devstral+Apertus+EuroLLM) vs v2 (Qwen3.6+Medium3.5)

Evaluates LoRA adapters across 4 dimensions:
  1. Perplexity (cross-entropy loss on validation data)
  2. Generation quality (sample outputs + speed)
  3. Adapter efficiency (size, params, training metrics)
  4. Inference speed benchmark (prompt/gen tok/s, peak memory)

Usage:
  python eval_framework.py --mode compare
  python eval_framework.py --mode v1-only
  python eval_framework.py --mode v2-only
  python eval_framework.py --mode compare --quick

EU AI Act Art. 53(1)(d): evaluation methodology documented for transparency.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx  # module-level handle for monkey-patchable probe (kept here so tests can monkeypatch eval_framework.mx)

# Wired-memory budget for MLX. Stays under macOS iogpu.wired_limit_mb=458752
# (= 448 GiB) hard cap with 8 GiB headroom for kernel + non-Metal allocations.
# Going above the wired cap triggers kernel SIGKILL at MLX import time.
WIRED_MEMORY_BUDGET_GIB = 440

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EU_KIKI = Path.home() / "eu-kiki"
KIKI_TUNNER = Path.home() / "KIKI-Mac_tunner"
HF_DATA = EU_KIKI / "data" / "hf-traced"
ADAPTERS_V1 = EU_KIKI / "output" / "adapters"
ADAPTERS_V2 = EU_KIKI / "output" / "adapters-v2"
EVAL_OUTPUT = EU_KIKI / "output" / "eval"
RAW_OUTPUT = EVAL_OUTPUT / "raw"
LOG_DIR = EU_KIKI / "output" / "training-logs"

# Inject mlx_lm_fork before mlx_lm
sys.path.insert(0, str(KIKI_TUNNER / "lib"))

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "devstral": {
        "path": str(KIKI_TUNNER / "models" / "Devstral-Small-2-24B-MLX-4bit"),
        "version": "v1",
        "short": "Devstral 24B",
        "params_b": 24,
        "license": "Apache-2.0",
    },
    "apertus": {
        "path": str(KIKI_TUNNER / "models" / "Apertus-70B-Instruct-2509"),
        "version": "v1",
        "short": "Apertus 70B",
        "params_b": 70,
        "license": "Apache-2.0",
    },
    "eurollm": {
        "path": str(KIKI_TUNNER / "models" / "EuroLLM-22B-Instruct-2512"),
        "version": "v1",
        "short": "EuroLLM 22B",
        "params_b": 22,
        "license": "Apache-2.0",
    },
    "qwen36": {
        "path": str(KIKI_TUNNER / "models" / "Qwen3.6-35B-A3B"),
        "version": "v2",
        "short": "Qwen3.6 35B-A3B",
        "params_b": 35,
        "active_b": 3,
        "license": "Apache-2.0",
    },
    "medium35": {
        "path": str(KIKI_TUNNER / "models" / "Mistral-Medium-3.5-128B-BF16"),
        "version": "v2",
        "short": "Medium 3.5 128B",
        "params_b": 128,
        "license": "Modified-MIT",
    },
}

# ---------------------------------------------------------------------------
# Domain → model mapping  (which model trains which domain)
# ---------------------------------------------------------------------------
V1_DOMAIN_MAP: dict[str, str] = {}

_DEVSTRAL_DOMAINS = [
    "python", "rust", "typescript", "cpp", "shell", "html-css", "sql",
    "web-backend", "web-frontend", "docker-devops", "llm-ops", "llm-orch",
    "ml-training", "lua-upy", "platformio", "iot", "music-audio", "freecad",
    "kicad-dsl", "kicad-pcb", "rust-embedded", "yaml-json",
]
for _d in _DEVSTRAL_DOMAINS:
    V1_DOMAIN_MAP[_d] = "devstral"

_APERTUS_DOMAINS = [
    "electronics", "embedded", "emc-dsp-power", "math-gsm8k",
    "math-reasoning", "security-fenrir", "spice-sim",
]
for _d in _APERTUS_DOMAINS:
    V1_DOMAIN_MAP[_d] = "apertus"

_EUROLLM_DOMAINS = ["chat-fr", "multilingual-eu", "traduction-tech"]
for _d in _EUROLLM_DOMAINS:
    V1_DOMAIN_MAP[_d] = "eurollm"

V2_DOMAIN_MAP: dict[str, str] = {}

_QWEN36_DOMAINS = [
    "python", "rust", "typescript", "sql", "shell", "html-css",
    "docker-devops", "llm-ops", "ml-training", "web-backend",
    "web-frontend", "yaml-json", "llm-orch", "iot", "lua-upy",
    "platformio", "music-audio", "freecad", "cpp", "rust-embedded",
    "kicad-dsl", "kicad-pcb",
]
for _d in _QWEN36_DOMAINS:
    V2_DOMAIN_MAP[_d] = "qwen36"

_MEDIUM35_DOMAINS = [
    "electronics", "math-gsm8k", "math-reasoning", "embedded",
    "emc-dsp-power", "security-fenrir", "spice-sim",
    "chat-fr", "multilingual-eu", "traduction-tech",
]
for _d in _MEDIUM35_DOMAINS:
    V2_DOMAIN_MAP[_d] = "medium35"

# All unique domains across both versions
ALL_DOMAINS = sorted(set(list(V1_DOMAIN_MAP.keys()) + list(V2_DOMAIN_MAP.keys())))

# Domains where both v1 and v2 have adapters (for direct comparison)
COMPARABLE_DOMAINS = sorted(
    set(V1_DOMAIN_MAP.keys()) & set(V2_DOMAIN_MAP.keys())
)

# ---------------------------------------------------------------------------
# Test prompts per domain category
# ---------------------------------------------------------------------------
DOMAIN_PROMPTS: dict[str, list[str]] = {
    # -- coding --
    "python": [
        "Write a function that implements a least-recently-used (LRU) cache with O(1) get and put operations.",
        "Write a Python async generator that yields paginated API results with exponential backoff on rate limits.",
        "Write a function to parse nested JSON-like bracket expressions into a tree structure.",
        "Implement a context manager that captures and replays stdout for testing purposes.",
        "Write a decorator that retries a function on specific exception types with configurable delay.",
    ],
    "rust": [
        "Write a function that parses a CSV line handling quoted fields with embedded commas.",
        "Implement a thread-safe ring buffer using Arc and Mutex.",
        "Write a Rust function that validates UTF-8 byte sequences without the standard library.",
        "Implement an iterator adapter that batches elements into fixed-size chunks.",
        "Write a generic binary search function that works on any Ord type.",
    ],
    "typescript": [
        "Write a type-safe event emitter with TypeScript generics that enforces event payload types.",
        "Implement a debounce function with proper TypeScript typing that returns a cancellable promise.",
        "Write a recursive type that deeply makes all properties of an object optional.",
        "Implement a type-safe builder pattern for constructing complex configuration objects.",
        "Write a function that merges two objects with full type inference on the result.",
    ],
    "cpp": [
        "Write a C++ template function for a compile-time Fibonacci sequence using constexpr.",
        "Implement a simple smart pointer similar to unique_ptr with move semantics.",
        "Write a lock-free single-producer single-consumer queue using atomics.",
        "Implement a memory pool allocator for fixed-size objects.",
        "Write a variadic template function that prints all arguments with type info.",
    ],
    "sql": [
        "Write a SQL query to find the top 3 customers by total order value per quarter using window functions.",
        "Write a recursive CTE to traverse a hierarchical category tree and compute depth.",
        "Write a query to detect gaps in a sequential ID column.",
        "Write a SQL query to compute a 7-day rolling average of daily revenue.",
        "Write a query to pivot monthly sales data from rows to columns.",
    ],
    "shell": [
        "Write a bash function that safely creates a temporary directory and cleans it up on exit.",
        "Write a shell script that monitors disk usage and sends an alert when usage exceeds 90%.",
        "Write a bash function to parse command-line arguments with both short and long options.",
        "Write a script that performs parallel execution of commands with a configurable concurrency limit.",
        "Write a bash function to validate an IPv4 address.",
    ],
    "html-css": [
        "Write HTML and CSS for a responsive card grid that transitions from 1 to 3 columns.",
        "Create a pure CSS animated loading spinner with a gradient ring effect.",
        "Write a CSS-only dropdown menu with smooth height transitions.",
        "Create an accessible form with floating labels using only HTML and CSS.",
        "Write CSS for a sticky header that shrinks on scroll using CSS custom properties.",
    ],
    # -- reasoning --
    "embedded": [
        "Write firmware to read an I2C sensor on STM32 using HAL, with error handling and timeout.",
        "Explain the design considerations for a watchdog timer in safety-critical embedded systems.",
        "Write a bare-metal UART driver for ARM Cortex-M4 with interrupt-driven receive.",
        "Explain how to implement DMA-based ADC sampling with double buffering on STM32.",
        "Write a state machine for debouncing multiple GPIO inputs on a microcontroller.",
    ],
    "electronics": [
        "Explain the design considerations for a low-noise power supply for an ADC front-end.",
        "Design a voltage divider circuit for a 3.3V ADC reading a 12V battery with protection.",
        "Explain the tradeoffs between linear regulators and switching regulators for battery-powered IoT.",
        "Design an RC low-pass filter with a 1kHz cutoff frequency and explain component selection.",
        "Explain ground plane design considerations for a mixed-signal PCB.",
    ],
    "math-gsm8k": [
        "A store sells notebooks for $4 each. If you buy 5 or more, you get a 20% discount. How much do 7 notebooks cost?",
        "A train travels 120 km in 1.5 hours. It then slows down and covers 80 km in 2 hours. What is the average speed?",
        "If 3 workers can complete a job in 12 days, how many days will it take 4 workers?",
        "A rectangular garden is 15m long and 8m wide. If you add a 2m border path around it, what is the total area including the path?",
        "A cistern can be filled by pipe A in 6 hours and pipe B in 8 hours. How long to fill it with both pipes?",
    ],
    "math-reasoning": [
        "Explain the design considerations for choosing between Euler and Runge-Kutta methods for ODE solving.",
        "Prove that the sum of the first n odd numbers equals n squared.",
        "Explain why the harmonic series diverges while the p-series with p=2 converges.",
        "Derive the formula for the area of a circle using integration.",
        "Explain the intuition behind eigenvalues and their physical interpretation in vibration analysis.",
    ],
    "emc-dsp-power": [
        "Explain EMI filtering techniques for a switching power supply to meet EN 55032 Class B.",
        "Design a digital low-pass FIR filter with 50 taps for audio signal processing at 44.1kHz.",
        "Explain the tradeoffs between synchronous and asynchronous buck converter topologies.",
        "Describe common-mode and differential-mode noise suppression techniques for power lines.",
        "Explain how to size a transformer for a flyback converter with 12V/2A output.",
    ],
    "security-fenrir": [
        "Explain buffer overflow protections in modern embedded systems (stack canaries, ASLR, MPU).",
        "Describe secure boot chain verification for an ARM Cortex-M microcontroller.",
        "Explain how to implement secure firmware update over-the-air with rollback protection.",
        "Describe the OWASP IoT Top 10 vulnerabilities and mitigation strategies.",
        "Explain how to implement certificate pinning for TLS in an embedded MQTT client.",
    ],
    "spice-sim": [
        "Write a SPICE netlist for a common-emitter amplifier with bypass capacitor and bias network.",
        "Write a SPICE simulation for a buck converter with PWM control and output LC filter.",
        "Create a SPICE model for a simple op-amp based active low-pass filter.",
        "Write a SPICE transient analysis for an RC oscillator circuit.",
        "Create a SPICE AC analysis sweep for a bandpass filter from 100Hz to 100kHz.",
    ],
    # -- multilingual --
    "chat-fr": [
        "Explique en français les avantages et inconvénients de l'architecture microservices.",
        "Décris le fonctionnement d'un transformateur en apprentissage automatique.",
        "Explique le principe de fonctionnement d'un amplificateur opérationnel.",
        "Compare les architectures ARM et RISC-V pour les systèmes embarqués.",
        "Explique le protocole MQTT et ses cas d'usage en IoT.",
    ],
    "multilingual-eu": [
        "Traduis en français technique: 'The watchdog timer resets the microcontroller if the main loop stalls.'",
        "Traduis en allemand: 'The PCB layout requires careful impedance matching for high-speed signals.'",
        "Explique en français la différence entre I2C et SPI pour les capteurs embarqués.",
        "Traduis en espagnol: 'The firmware update procedure uses a dual-bank flash memory scheme.'",
        "Traduis en italien: 'The power supply design must comply with CE marking requirements.'",
    ],
    "traduction-tech": [
        "Traduis en français technique: 'The differential pair routing must maintain 100 ohm impedance.'",
        "Traduis en français: 'Stack canaries detect buffer overflow attacks at runtime.'",
        "Traduis en français technique le paragraphe suivant sur le protocole CAN bus.",
        "Traduis en anglais: 'Le condensateur de découplage doit être placé au plus près du pin d'alimentation.'",
        "Traduis en français: 'The MOSFET gate driver requires a bootstrap capacitor for high-side switching.'",
    ],
    # -- kicad --
    "kicad-dsl": [
        "Create a KiCad symbol for a 4-pin connector with VCC, GND, SDA, and SCL pins.",
        "Write a KiCad footprint for a SOT-23-5 package with the correct pad dimensions.",
        "Create a KiCad symbol for an LDO voltage regulator with EN, IN, OUT, and GND pins.",
        "Write the KiCad S-expression for a 0805 capacitor footprint.",
        "Create a KiCad hierarchical sheet pin list for an I2C sensor module.",
    ],
    "kicad-pcb": [
        "Describe the PCB layout strategy for a 4-layer mixed-signal board with analog and digital sections.",
        "Explain via stitching patterns for ground plane integrity in a multilayer PCB.",
        "Describe component placement strategy for a USB-C connector with ESD protection.",
        "Explain how to route differential pairs for USB 2.0 on a 2-layer PCB.",
        "Describe thermal relief pad design considerations for power components.",
    ],
    # -- other coding --
    "docker-devops": [
        "Write a multi-stage Dockerfile for a Python FastAPI application with minimal final image.",
        "Write a docker-compose.yml for a 3-service stack: API, PostgreSQL, and Redis.",
        "Write a GitHub Actions workflow for building, testing, and pushing a Docker image.",
        "Write a Dockerfile that uses BuildKit cache mounts for pip dependencies.",
        "Write a docker-compose health check for a PostgreSQL service.",
    ],
    "web-backend": [
        "Write a FastAPI endpoint with Pydantic validation, error handling, and pagination.",
        "Implement rate limiting middleware for an Express.js API.",
        "Write a database migration script that adds an index without downtime.",
        "Implement a webhook receiver with signature verification and idempotency.",
        "Write a background task queue consumer with retry logic and dead letter handling.",
    ],
    "web-frontend": [
        "Write a React custom hook for infinite scroll with intersection observer.",
        "Implement a virtualized list component for rendering 10,000 items efficiently.",
        "Write a React context provider for theme switching with system preference detection.",
        "Implement a form component with real-time validation using React Hook Form.",
        "Write a service worker registration script with update notification.",
    ],
    "llm-ops": [
        "Write a Python script to measure LLM inference latency across different batch sizes.",
        "Implement a simple prompt caching layer using Redis for an LLM API gateway.",
        "Write a monitoring dashboard configuration for tracking LLM token usage and costs.",
        "Implement a fallback chain for LLM providers with automatic failover.",
        "Write a script to benchmark VRAM usage for different quantization levels.",
    ],
    "llm-orch": [
        "Design an LLM routing pipeline that selects models based on task complexity.",
        "Implement a multi-agent conversation framework with shared context.",
        "Write an orchestration layer for chaining LLM calls with intermediate validation.",
        "Design a RAG pipeline with hybrid search combining dense and sparse retrieval.",
        "Implement a prompt template system with variable injection and version control.",
    ],
    "ml-training": [
        "Write a PyTorch training loop with gradient accumulation and mixed-precision training.",
        "Implement early stopping with patience and model checkpointing in a training script.",
        "Write a custom dataset class for loading JSONL conversation data for fine-tuning.",
        "Implement a learning rate scheduler with warmup and cosine annealing.",
        "Write a distributed training script using PyTorch DDP with proper initialization.",
    ],
    "yaml-json": [
        "Write a YAML schema for validating Kubernetes deployment manifests.",
        "Write a JSON Schema for validating OpenAPI 3.1 operation objects.",
        "Create a YAML configuration file for a multi-environment CI/CD pipeline.",
        "Write a JSON transformation script that flattens nested objects with dot notation keys.",
        "Create a YAML template with anchors and aliases for a microservices configuration.",
    ],
    "iot": [
        "Write an MQTT client in Python that publishes sensor data with QoS 1 and handles reconnection.",
        "Design a message format for IoT telemetry data with efficient binary encoding.",
        "Write firmware for an ESP32 to connect to WiFi, read a DHT22 sensor, and publish via MQTT.",
        "Implement an IoT gateway that aggregates data from multiple BLE sensors.",
        "Write a CoAP server for a constrained IoT device with observable resources.",
    ],
    "lua-upy": [
        "Write a MicroPython driver for an SSD1306 OLED display over I2C.",
        "Implement a cooperative task scheduler in MicroPython for ESP32.",
        "Write a Lua script for a NodeMCU that controls an LED strip via MQTT.",
        "Implement a MicroPython web server on ESP32 for configuring WiFi credentials.",
        "Write a Lua coroutine-based event loop for handling multiple sensor readings.",
    ],
    "platformio": [
        "Write a PlatformIO configuration for an ESP32 project with OTA update support.",
        "Create a platformio.ini with multiple environments for STM32 and ESP32 targets.",
        "Write a PlatformIO custom script for pre-build version injection.",
        "Configure PlatformIO for unit testing with Unity framework on embedded targets.",
        "Write a PlatformIO library.json manifest for a reusable sensor driver library.",
    ],
    "rust-embedded": [
        "Write a Rust embedded HAL driver for reading an SPI accelerometer on STM32.",
        "Implement a no_std ring buffer for UART DMA reception in embedded Rust.",
        "Write a Rust RTIC task for periodic sensor sampling with shared resources.",
        "Implement a Rust embedded driver for an I2C temperature sensor using embedded-hal traits.",
        "Write a Rust no_std state machine for a motor controller with error recovery.",
    ],
    "music-audio": [
        "Write a Python script to generate a sine wave WAV file at a given frequency and duration.",
        "Implement a simple audio effects chain with reverb and delay in Python.",
        "Write a real-time audio FFT visualizer using PyAudio and NumPy.",
        "Implement a MIDI note parser that converts MIDI messages to frequency values.",
        "Write a script to detect BPM from an audio file using onset detection.",
    ],
    "freecad": [
        "Write a FreeCAD Python script to create a parametric box with fillets.",
        "Create a FreeCAD macro to generate a gear profile with configurable parameters.",
        "Write a FreeCAD script to export all bodies in a document to STEP format.",
        "Implement a FreeCAD workbench toolbar button that creates a predefined assembly.",
        "Write a FreeCAD Python script to create a sheet metal bracket with bends.",
    ],
}

# Fallback for domains without specific prompts
DEFAULT_PROMPTS = [
    "Explain the key concepts and best practices in this domain.",
    "Write a comprehensive example demonstrating core functionality.",
    "What are common pitfalls and how to avoid them?",
    "Describe the architecture of a production-grade system in this area.",
    "Write a tutorial-style explanation with code examples.",
]

# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class PerplexityResult:
    domain: str
    model_key: str
    model_name: str
    version: str
    val_loss: float
    perplexity: float
    n_samples: int
    eval_time_s: float


@dataclass
class GenerationResult:
    domain: str
    model_key: str
    model_name: str
    version: str
    prompt: str
    response: str
    tokens_generated: int
    generation_time_s: float
    tokens_per_sec: float


@dataclass
class AdapterEfficiency:
    domain: str
    model_key: str
    version: str
    adapter_size_mb: float
    trainable_params: Optional[int]
    final_train_loss: Optional[float]
    final_val_loss: Optional[float]
    training_time_min: Optional[float]


@dataclass
class SpeedBenchmark:
    model_key: str
    model_name: str
    prompt_tokens: int
    prompt_speed_tps: float
    gen_tokens: int
    gen_speed_tps: float
    peak_memory_gb: float


@dataclass
class EvalResults:
    perplexity: list[PerplexityResult] = field(default_factory=list)
    generations: list[GenerationResult] = field(default_factory=list)
    efficiency: list[AdapterEfficiency] = field(default_factory=list)
    speed: list[SpeedBenchmark] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility: MLX model loading / unloading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, adapter_path: str | None = None):
    """Load an MLX model with optional LoRA adapter. Returns (model, tokenizer)."""
    import mlx.core as mx

    mx.set_memory_limit(WIRED_MEMORY_BUDGET_GIB * 1024**3)
    mx.set_cache_limit(32 * 1024**3)

    from mlx_lm_fork import load as mlx_load
    from mlx_lm_fork.tuner.utils import load_adapters

    model, tokenizer = mlx_load(model_path)

    if adapter_path and Path(adapter_path).exists():
        adapter_file = Path(adapter_path) / "adapters.safetensors"
        if adapter_file.exists():
            model = load_adapters(model, str(adapter_path))
            print(f"  Applied LoRA adapter from {adapter_path}")

    return model, tokenizer


def unload_model():
    """Free GPU memory between model loads."""
    import mlx.core as mx

    gc.collect()
    mx.metal.clear_cache()
    time.sleep(1)
    gc.collect()


def _assert_within_budget(budget_gib: int = WIRED_MEMORY_BUDGET_GIB) -> None:
    """Abort cleanly with RuntimeError if Metal peak memory has exceeded the
    configured budget. Called between every model transition in
    sequential-strict mode so an overrun produces a structured error
    instead of a kernel SIGKILL.
    """
    peak_b = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()
    peak_gib = peak_b / (1024 ** 3)
    if peak_gib > budget_gib:
        raise RuntimeError(
            f"peak memory {peak_gib:.1f} GiB exceeds budget {budget_gib} GiB"
        )


# ---------------------------------------------------------------------------
# 1. Perplexity evaluation
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    valid_path: Path,
    max_samples: int = 50,
) -> tuple[float, int]:
    """Compute average cross-entropy loss on validation data.

    Returns (avg_loss, n_samples).
    """
    import mlx.core as mx
    import mlx.nn as nn

    records = []
    with open(valid_path) as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
            if len(records) >= max_samples:
                break

    if not records:
        return float("nan"), 0

    total_loss = 0.0
    total_tokens = 0
    n_evaluated = 0

    for rec in records:
        messages = rec.get("messages", [])
        if not messages:
            continue

        # Build full conversation text
        text_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            text_parts.append(f"<|{role}|>\n{content}")
        full_text = "\n".join(text_parts)

        tokens = tokenizer.encode(full_text)
        if len(tokens) < 2:
            continue

        # Truncate to avoid OOM
        max_len = 2048
        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        input_ids = mx.array(tokens[:-1])[None, :]
        target_ids = mx.array(tokens[1:])

        logits = model(input_ids)
        logits = logits.squeeze(0)

        loss = nn.losses.cross_entropy(logits, target_ids, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += len(tokens) - 1
        n_evaluated += 1

    if total_tokens == 0:
        return float("nan"), 0

    avg_loss = total_loss / total_tokens
    return avg_loss, n_evaluated


def eval_perplexity_for_adapter(
    model_key: str,
    domain: str,
    version: str,
    max_samples: int = 50,
) -> PerplexityResult | None:
    """Evaluate perplexity for a single model+domain adapter pair."""
    model_info = MODELS[model_key]
    adapters_base = ADAPTERS_V1 if version == "v1" else ADAPTERS_V2
    adapter_path = adapters_base / model_key / domain

    valid_path = HF_DATA / domain / "valid.jsonl"
    if not valid_path.exists():
        print(f"  SKIP {domain} — no valid.jsonl")
        return None

    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.exists():
        print(f"  SKIP {version}/{model_key}/{domain} — adapter not found")
        return None

    print(f"  Evaluating perplexity: {version}/{model_key}/{domain}")
    t0 = time.time()

    model, tokenizer = load_model_and_tokenizer(
        model_info["path"], str(adapter_path)
    )
    avg_loss, n_samples = compute_perplexity(
        model, tokenizer, valid_path, max_samples=max_samples
    )
    elapsed = time.time() - t0

    del model, tokenizer
    unload_model()

    ppl = math.exp(avg_loss) if not math.isnan(avg_loss) else float("nan")

    return PerplexityResult(
        domain=domain,
        model_key=model_key,
        model_name=model_info["short"],
        version=version,
        val_loss=round(avg_loss, 4),
        perplexity=round(ppl, 2),
        n_samples=n_samples,
        eval_time_s=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# 2. Generation quality
# ---------------------------------------------------------------------------

def generate_sample(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
) -> tuple[str, int, float]:
    """Generate text and return (response, n_tokens, elapsed_seconds)."""
    from mlx_lm_fork import generate as mlx_generate

    t0 = time.time()
    response = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    elapsed = time.time() - t0

    # Count response tokens
    response_tokens = tokenizer.encode(response)
    n_tokens = len(response_tokens)

    return response, n_tokens, elapsed


def eval_generation_for_adapter(
    model_key: str,
    domain: str,
    version: str,
    max_tokens: int = 200,
) -> list[GenerationResult]:
    """Generate sample outputs for a model+domain adapter pair."""
    model_info = MODELS[model_key]

    # Medium 3.5 BF16 is too slow for generation — skip
    if model_key == "medium35":
        print(f"  SKIP generation for {model_key} (BF16 too slow, use Q4 for speed test)")
        return []

    adapters_base = ADAPTERS_V1 if version == "v1" else ADAPTERS_V2
    adapter_path = adapters_base / model_key / domain

    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.exists():
        return []

    prompts = DOMAIN_PROMPTS.get(domain, DEFAULT_PROMPTS)[:5]

    print(f"  Generating samples: {version}/{model_key}/{domain} ({len(prompts)} prompts)")

    model, tokenizer = load_model_and_tokenizer(
        model_info["path"], str(adapter_path)
    )

    results = []
    for prompt in prompts:
        response, n_tokens, elapsed = generate_sample(
            model, tokenizer, prompt, max_tokens=max_tokens
        )
        tps = n_tokens / elapsed if elapsed > 0 else 0.0
        results.append(GenerationResult(
            domain=domain,
            model_key=model_key,
            model_name=model_info["short"],
            version=version,
            prompt=prompt,
            response=response,
            tokens_generated=n_tokens,
            generation_time_s=round(elapsed, 2),
            tokens_per_sec=round(tps, 1),
        ))

    del model, tokenizer
    unload_model()

    return results


# ---------------------------------------------------------------------------
# 3. Adapter efficiency
# ---------------------------------------------------------------------------

def parse_training_log(log_path: Path) -> dict:
    """Extract final train loss, val loss, and training time from a log file."""
    result = {
        "final_train_loss": None,
        "final_val_loss": None,
        "training_time_min": None,
        "peak_memory_gb": None,
    }
    if not log_path.exists():
        return result

    content = log_path.read_text()

    # Find last "Train loss X.XXX"
    train_losses = re.findall(r"Train loss ([\d.]+)", content)
    if train_losses:
        result["final_train_loss"] = float(train_losses[-1])

    # Find last "Val loss X.XXX"
    val_losses = re.findall(r"Val loss ([\d.]+)", content)
    if val_losses:
        result["final_val_loss"] = float(val_losses[-1])

    # Find "Done in X.X min"
    done_match = re.search(r"Done in ([\d.]+) min", content)
    if done_match:
        result["training_time_min"] = float(done_match.group(1))

    # Find peak memory
    peak_mems = re.findall(r"Peak mem ([\d.]+) GB", content)
    if peak_mems:
        result["peak_memory_gb"] = float(peak_mems[-1])

    return result


def eval_adapter_efficiency(
    model_key: str,
    domain: str,
    version: str,
) -> AdapterEfficiency | None:
    """Measure adapter file size and extract training metrics."""
    adapters_base = ADAPTERS_V1 if version == "v1" else ADAPTERS_V2
    adapter_path = adapters_base / model_key / domain
    adapter_file = adapter_path / "adapters.safetensors"

    if not adapter_file.exists():
        return None

    size_mb = adapter_file.stat().st_size / (1024 * 1024)

    # Determine trainable params from safetensors metadata
    trainable_params = None
    try:
        from safetensors import safe_open

        with safe_open(str(adapter_file), framework="mlx") as f:
            total_params = 0
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                total_params += math.prod(shape)
            trainable_params = total_params
    except Exception:
        pass

    # Find corresponding training log
    log_candidates = [
        LOG_DIR / f"batch9-qwen36-{domain}.log",
        LOG_DIR / f"batch10-medium35-{domain}.log",
        LOG_DIR / f"devstral-{domain}-bf16.log",
        LOG_DIR / f"devstral-{domain}.log",
        LOG_DIR / f"devstral-{domain}-curriculum.log",
        LOG_DIR / f"devstral-{domain}-fullseq.log",
        LOG_DIR / f"apertus-{domain}.log",
        LOG_DIR / f"apertus-{domain}-curriculum.log",
        LOG_DIR / f"eurollm-{domain}.log",
    ]

    log_info = {"final_train_loss": None, "final_val_loss": None, "training_time_min": None}
    for log_path in log_candidates:
        if log_path.exists():
            log_info = parse_training_log(log_path)
            if log_info.get("final_train_loss") is not None:
                break

    return AdapterEfficiency(
        domain=domain,
        model_key=model_key,
        version=version,
        adapter_size_mb=round(size_mb, 1),
        trainable_params=trainable_params,
        final_train_loss=(
            round(log_info["final_train_loss"], 4)
            if log_info["final_train_loss"] is not None
            else None
        ),
        final_val_loss=(
            round(log_info["final_val_loss"], 4)
            if log_info["final_val_loss"] is not None
            else None
        ),
        training_time_min=(
            round(log_info["training_time_min"], 1)
            if log_info["training_time_min"] is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# 4. Inference speed benchmark
# ---------------------------------------------------------------------------

SPEED_PROMPT = (
    "You are a helpful AI assistant. Explain the concept of transfer learning "
    "in machine learning, including its benefits and common approaches."
)


def benchmark_model_speed(model_key: str) -> SpeedBenchmark | None:
    """Benchmark raw model speed without adapters."""
    model_info = MODELS[model_key]

    # Skip Medium 3.5 BF16 generation benchmark (too slow), but still measure prompt
    skip_gen = model_key == "medium35"

    print(f"  Benchmarking speed: {model_key} ({model_info['short']})")

    import mlx.core as mx

    model, tokenizer = load_model_and_tokenizer(model_info["path"])

    # Measure prompt processing
    tokens = tokenizer.encode(SPEED_PROMPT)
    input_ids = mx.array(tokens)[None, :]

    t0 = time.time()
    logits = model(input_ids)
    mx.eval(logits)
    prompt_time = time.time() - t0
    prompt_tps = len(tokens) / prompt_time if prompt_time > 0 else 0.0

    # Measure generation speed
    gen_tokens = 0
    gen_tps = 0.0
    if not skip_gen:
        from mlx_lm_fork import generate as mlx_generate

        t0 = time.time()
        output = mlx_generate(
            model, tokenizer, prompt=SPEED_PROMPT, max_tokens=100, verbose=False
        )
        gen_time = time.time() - t0
        gen_tokens = len(tokenizer.encode(output))
        gen_tps = gen_tokens / gen_time if gen_time > 0 else 0.0

    # Peak memory
    peak_mem = mx.metal.get_peak_memory() / (1024**3)

    del model, tokenizer
    unload_model()

    return SpeedBenchmark(
        model_key=model_key,
        model_name=model_info["short"],
        prompt_tokens=len(tokens),
        prompt_speed_tps=round(prompt_tps, 1),
        gen_tokens=gen_tokens,
        gen_speed_tps=round(gen_tps, 1),
        peak_memory_gb=round(peak_mem, 1),
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: EvalResults, mode: str) -> str:
    """Generate the markdown comparison report."""
    lines = [
        "# AILIANCE Eval Report: v1 vs v2",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Mode**: {mode}",
        "",
        "**EU AI Act Art. 53(1)(d)**: This evaluation documents adapter quality metrics",
        "for transparency and reproducibility. All training data has provenance tracking.",
        "",
    ]

    # ---- Summary table ----
    if mode == "compare":
        lines.append("## Summary Table — Comparable Domains")
        lines.append("")
        lines.append(
            "| Domain | v1 Model | v1 Loss | v1 PPL | v2 Model | v2 Loss | v2 PPL | Winner | Delta |"
        )
        lines.append(
            "|--------|----------|---------|--------|----------|---------|--------|--------|-------|"
        )

        ppl_by_key = {}
        for r in results.perplexity:
            ppl_by_key[(r.version, r.domain)] = r

        for domain in COMPARABLE_DOMAINS:
            v1 = ppl_by_key.get(("v1", domain))
            v2 = ppl_by_key.get(("v2", domain))

            v1_model = v1.model_name if v1 else "—"
            v1_loss = f"{v1.val_loss:.4f}" if v1 else "—"
            v1_ppl = f"{v1.perplexity:.2f}" if v1 else "—"
            v2_model = v2.model_name if v2 else "—"
            v2_loss = f"{v2.val_loss:.4f}" if v2 else "—"
            v2_ppl = f"{v2.perplexity:.2f}" if v2 else "—"

            if v1 and v2 and not math.isnan(v1.val_loss) and not math.isnan(v2.val_loss):
                if v2.val_loss < v1.val_loss:
                    winner = "v2"
                    delta = f"-{((v1.val_loss - v2.val_loss) / v1.val_loss * 100):.1f}%"
                elif v1.val_loss < v2.val_loss:
                    winner = "v1"
                    delta = f"+{((v2.val_loss - v1.val_loss) / v1.val_loss * 100):.1f}%"
                else:
                    winner = "tie"
                    delta = "0%"
            else:
                winner = "—"
                delta = "—"

            lines.append(
                f"| {domain} | {v1_model} | {v1_loss} | {v1_ppl} | "
                f"{v2_model} | {v2_loss} | {v2_ppl} | {winner} | {delta} |"
            )

        lines.append("")

    # ---- Per-domain details ----
    lines.append("## Per-Domain Details")
    lines.append("")

    ppl_by_key = {}
    for r in results.perplexity:
        ppl_by_key[(r.version, r.domain)] = r

    gen_by_key: dict[tuple[str, str], list[GenerationResult]] = {}
    for g in results.generations:
        key = (g.version, g.domain)
        gen_by_key.setdefault(key, []).append(g)

    eff_by_key = {}
    for e in results.efficiency:
        eff_by_key[(e.version, e.domain)] = e

    for domain in ALL_DOMAINS:
        v1_ppl = ppl_by_key.get(("v1", domain))
        v2_ppl = ppl_by_key.get(("v2", domain))

        if not v1_ppl and not v2_ppl:
            continue

        lines.append(f"### {domain}")
        lines.append("")

        for ver_label, ppl_r in [("v1", v1_ppl), ("v2", v2_ppl)]:
            if not ppl_r:
                continue
            eff_r = eff_by_key.get((ver_label, domain))
            size_str = f", adapter={eff_r.adapter_size_mb:.0f}MB" if eff_r else ""
            train_loss_str = (
                f", train_loss={eff_r.final_train_loss:.4f}"
                if eff_r and eff_r.final_train_loss is not None
                else ""
            )
            lines.append(
                f"- **{ver_label}** ({ppl_r.model_name}): "
                f"loss={ppl_r.val_loss:.4f}, ppl={ppl_r.perplexity:.2f}, "
                f"n={ppl_r.n_samples}, eval={ppl_r.eval_time_s:.0f}s"
                f"{size_str}{train_loss_str}"
            )

        # Sample generations
        for ver_label in ["v1", "v2"]:
            gens = gen_by_key.get((ver_label, domain), [])
            if gens:
                avg_tps = sum(g.tokens_per_sec for g in gens) / len(gens)
                lines.append(f"- {ver_label} generation speed: {avg_tps:.1f} tok/s avg")
                lines.append(f"  - Sample prompt: *{gens[0].prompt[:80]}...*")
                response_preview = gens[0].response[:200].replace("\n", " ")
                lines.append(f"  - Sample response: {response_preview}...")

        lines.append("")

    # ---- Model speed comparison ----
    if results.speed:
        lines.append("## Model Speed Comparison (no adapters)")
        lines.append("")
        lines.append(
            "| Model | Prompt (tok/s) | Gen (tok/s) | Peak Memory (GB) |"
        )
        lines.append(
            "|-------|---------------|-------------|-------------------|"
        )
        for s in results.speed:
            gen_str = f"{s.gen_speed_tps:.1f}" if s.gen_tokens > 0 else "skipped (BF16)"
            lines.append(
                f"| {s.model_name} | {s.prompt_speed_tps:.1f} | {gen_str} | {s.peak_memory_gb:.1f} |"
            )
        lines.append("")

    # ---- Adapter efficiency ----
    if results.efficiency:
        lines.append("## Adapter Efficiency Summary")
        lines.append("")
        lines.append(
            "| Version | Model | Domains | Avg Size (MB) | Avg Train Loss | Avg Val Loss |"
        )
        lines.append(
            "|---------|-------|---------|---------------|----------------|--------------|"
        )

        from collections import defaultdict

        by_model: dict[str, list[AdapterEfficiency]] = defaultdict(list)
        for e in results.efficiency:
            by_model[f"{e.version}/{e.model_key}"].append(e)

        for key, effs in sorted(by_model.items()):
            ver, mk = key.split("/")
            n_domains = len(effs)
            avg_size = sum(e.adapter_size_mb for e in effs) / n_domains
            train_losses = [e.final_train_loss for e in effs if e.final_train_loss is not None]
            val_losses = [e.final_val_loss for e in effs if e.final_val_loss is not None]
            avg_tl = f"{sum(train_losses) / len(train_losses):.4f}" if train_losses else "—"
            avg_vl = f"{sum(val_losses) / len(val_losses):.4f}" if val_losses else "—"

            lines.append(
                f"| {ver} | {MODELS[mk]['short']} | {n_domains} | "
                f"{avg_size:.0f} | {avg_tl} | {avg_vl} |"
            )

        lines.append("")

    # ---- Recommendation ----
    lines.append("## Recommendation")
    lines.append("")

    if mode == "compare" and results.perplexity:
        v1_wins = 0
        v2_wins = 0
        for domain in COMPARABLE_DOMAINS:
            v1_r = ppl_by_key.get(("v1", domain))
            v2_r = ppl_by_key.get(("v2", domain))
            if v1_r and v2_r and not math.isnan(v1_r.val_loss) and not math.isnan(v2_r.val_loss):
                if v2_r.val_loss < v1_r.val_loss:
                    v2_wins += 1
                elif v1_r.val_loss < v2_r.val_loss:
                    v1_wins += 1

        total = v1_wins + v2_wins
        lines.append(
            f"Out of {total} comparable domains: "
            f"v1 wins {v1_wins}, v2 wins {v2_wins}."
        )
        lines.append("")
        lines.append(
            "Key factors for final decision: perplexity improvement, generation speed "
            "(MoE advantage for Qwen3.6), memory footprint, and EU sovereignty requirements."
        )
        lines.append("")
        lines.append("**Note on Medium 3.5**: At 128B BF16 (~238 GB), generation is slow. "
                      "Consider Q4 quantization for production serving or keep only "
                      "for reasoning-heavy domains where quality justifies the cost.")
    else:
        lines.append("Single-version eval complete. Run `--mode compare` for v1 vs v2 comparison.")

    lines.append("")

    # ---- Eval methodology ----
    lines.append("## Evaluation Methodology (EU AI Act Transparency)")
    lines.append("")
    lines.append("- **Perplexity**: Average cross-entropy loss on held-out validation data")
    lines.append("  (up to 50 samples per domain, truncated to 2048 tokens)")
    lines.append("- **Generation**: 5 domain-specific prompts, 200 tokens max per response")
    lines.append("- **Speed**: Measured on Apple M3 Ultra (512 GB unified memory)")
    lines.append("- **Adapter efficiency**: File size, trainable params, training metrics from logs")
    lines.append("- **All training data**: Provenance-tracked per EU AI Act Art. 53(1)(d)")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_raw_results(results: EvalResults, mode: str) -> None:
    """Save raw results as JSON for later analysis."""
    RAW_OUTPUT.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M")

    if results.perplexity:
        path = RAW_OUTPUT / f"perplexity_{mode}_{timestamp}.json"
        path.write_text(json.dumps(
            [asdict(r) for r in results.perplexity], indent=2
        ))
        print(f"  Saved: {path}")

    if results.generations:
        path = RAW_OUTPUT / f"generations_{mode}_{timestamp}.json"
        path.write_text(json.dumps(
            [asdict(r) for r in results.generations], indent=2
        ))
        print(f"  Saved: {path}")

    if results.efficiency:
        path = RAW_OUTPUT / f"efficiency_{mode}_{timestamp}.json"
        path.write_text(json.dumps(
            [asdict(r) for r in results.efficiency], indent=2
        ))
        print(f"  Saved: {path}")

    if results.speed:
        path = RAW_OUTPUT / f"speed_{mode}_{timestamp}.json"
        path.write_text(json.dumps(
            [asdict(r) for r in results.speed], indent=2
        ))
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def get_domains_to_eval(
    version: str,
) -> list[tuple[str, str]]:
    """Return list of (domain, model_key) pairs for a version."""
    domain_map = V1_DOMAIN_MAP if version == "v1" else V2_DOMAIN_MAP
    adapters_base = ADAPTERS_V1 if version == "v1" else ADAPTERS_V2

    pairs = []
    for domain, model_key in domain_map.items():
        adapter_file = adapters_base / model_key / domain / "adapters.safetensors"
        if adapter_file.exists():
            pairs.append((domain, model_key))
        else:
            print(f"  [pending] {version}/{model_key}/{domain} — adapter not yet trained")

    return pairs


def _strict_iteration_order(
    load_groups: dict[tuple[str, str], list[str]],
) -> list[tuple[tuple[str, str], list[str]]]:
    """Return (version, model_key) -> [domains] entries in an order that
    consumes every adapter for one base model before any adapter of another
    base model. Stable on dict insertion order so identical inputs always
    produce identical sequences (reproducibility for the bench history)."""
    return list(load_groups.items())


def run_eval(
    mode: str = "compare",
    quick: bool = False,
    skip_generation: bool = False,
    skip_speed: bool = False,
) -> None:
    """Run the full evaluation pipeline."""
    EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.mkdir(parents=True, exist_ok=True)

    max_ppl_samples = 5 if quick else 50
    results = EvalResults()

    versions_to_eval = []
    if mode in ("v1-only", "compare", "sequential-strict"):
        versions_to_eval.append("v1")
    if mode in ("v2-only", "compare", "sequential-strict"):
        versions_to_eval.append("v2")

    # ---- Phase 1: Adapter efficiency (no model loading needed) ----
    print("\n" + "=" * 60)
    print("Phase 1: Adapter Efficiency (metadata only)")
    print("=" * 60)

    for version in versions_to_eval:
        domain_map = V1_DOMAIN_MAP if version == "v1" else V2_DOMAIN_MAP
        for domain, model_key in domain_map.items():
            eff = eval_adapter_efficiency(model_key, domain, version)
            if eff:
                results.efficiency.append(eff)
                print(f"  {version}/{model_key}/{domain}: {eff.adapter_size_mb:.0f} MB")

    # ---- Phase 2: Perplexity evaluation ----
    # Group by model to minimize model loads
    print("\n" + "=" * 60)
    print(f"Phase 2: Perplexity Evaluation (max {max_ppl_samples} samples/domain)")
    print("=" * 60)

    # Build load order: group domains by (version, model_key) to load each model once
    load_groups: dict[tuple[str, str], list[str]] = {}
    for version in versions_to_eval:
        pairs = get_domains_to_eval(version)
        for domain, model_key in pairs:
            key = (version, model_key)
            load_groups.setdefault(key, []).append(domain)

    for (version, model_key), domains in _strict_iteration_order(load_groups):
        model_info = MODELS[model_key]
        print(f"\n  Loading {model_key} ({model_info['short']}) for {len(domains)} domains...")

        for domain in domains:
            ppl_result = eval_perplexity_for_adapter(
                model_key, domain, version, max_samples=max_ppl_samples
            )
            if ppl_result:
                results.perplexity.append(ppl_result)
                print(
                    f"    {domain}: loss={ppl_result.val_loss:.4f}, "
                    f"ppl={ppl_result.perplexity:.2f}"
                )
        if mode == "sequential-strict":
            unload_model()
            _assert_within_budget(budget_gib=WIRED_MEMORY_BUDGET_GIB)

    # ---- Phase 3: Generation quality ----
    if not skip_generation and not quick:
        print("\n" + "=" * 60)
        print("Phase 3: Generation Quality")
        print("=" * 60)

        for (version, model_key), domains in _strict_iteration_order(load_groups):
            if model_key == "medium35":
                print(f"  SKIP generation for {model_key} (BF16 too slow)")
                # Skipping the budget probe below is safe: Phase 2's
                # sequential-strict epilogue already unloaded medium35.
                continue

            for domain in domains:
                gen_results = eval_generation_for_adapter(
                    model_key, domain, version
                )
                results.generations.extend(gen_results)
                if gen_results:
                    avg_tps = sum(g.tokens_per_sec for g in gen_results) / len(gen_results)
                    print(f"    {domain}: {avg_tps:.1f} tok/s avg")
            if mode == "sequential-strict":
                unload_model()
                _assert_within_budget(budget_gib=WIRED_MEMORY_BUDGET_GIB)

    # ---- Phase 4: Speed benchmark ----
    if not skip_speed and not quick:
        print("\n" + "=" * 60)
        print("Phase 4: Inference Speed Benchmark (base models, no adapters)")
        print("=" * 60)

        models_to_bench = set()
        for version in versions_to_eval:
            domain_map = V1_DOMAIN_MAP if version == "v1" else V2_DOMAIN_MAP
            for model_key in set(domain_map.values()):
                models_to_bench.add(model_key)

        for model_key in sorted(models_to_bench):
            speed_result = benchmark_model_speed(model_key)
            if speed_result:
                results.speed.append(speed_result)
                print(
                    f"    {model_key}: prompt={speed_result.prompt_speed_tps:.1f} tok/s, "
                    f"gen={speed_result.gen_speed_tps:.1f} tok/s, "
                    f"mem={speed_result.peak_memory_gb:.1f} GB"
                )

    # ---- Save results ----
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)

    save_raw_results(results, mode)

    report = generate_report(results, mode)
    report_path = EVAL_OUTPUT / "eval_report_v1_vs_v2.md"
    report_path.write_text(report)
    print(f"\n  Report saved: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVAL COMPLETE")
    print("=" * 60)
    print(f"  Perplexity evals: {len(results.perplexity)}")
    print(f"  Generation samples: {len(results.generations)}")
    print(f"  Adapter efficiency: {len(results.efficiency)}")
    print(f"  Speed benchmarks: {len(results.speed)}")
    print(f"  Report: {report_path}")
    print(f"  Raw data: {RAW_OUTPUT}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AILIANCE Eval Framework — v1 vs v2 adapter comparison"
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "v1-only", "v2-only", "sequential-strict"],
        default="compare",
        help="Which adapter versions to evaluate (default: compare)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 5 records per domain, skip generation and speed",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation quality evaluation",
    )
    parser.add_argument(
        "--skip-speed",
        action="store_true",
        help="Skip inference speed benchmark",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Evaluate only these domains (space-separated)",
    )

    args = parser.parse_args()

    # Filter domains if specified
    if args.domains:
        global V1_DOMAIN_MAP, V2_DOMAIN_MAP, ALL_DOMAINS, COMPARABLE_DOMAINS
        V1_DOMAIN_MAP = {k: v for k, v in V1_DOMAIN_MAP.items() if k in args.domains}
        V2_DOMAIN_MAP = {k: v for k, v in V2_DOMAIN_MAP.items() if k in args.domains}
        ALL_DOMAINS = sorted(set(list(V1_DOMAIN_MAP.keys()) + list(V2_DOMAIN_MAP.keys())))
        COMPARABLE_DOMAINS = sorted(
            set(V1_DOMAIN_MAP.keys()) & set(V2_DOMAIN_MAP.keys())
        )

    run_eval(
        mode=args.mode,
        quick=args.quick,
        skip_generation=args.skip_generation,
        skip_speed=args.skip_speed,
    )


if __name__ == "__main__":
    main()
