"""Curated router training prompts for under-represented niche domains.

After router-v5, 13 domains have <10 rows in the training set, which causes
mis-classification (the encoder either collapses them into a neighbour class
or scatters them with low confidence). This module provides a hand-curated
scaffold of high-confidence prompts per domain so the router has at least
~10 anchor points per niche.

Integration:
    `scripts/rebuild_router_dataset.py` imports `NICHE_DOMAIN_PROMPTS` and
    appends one row per (domain, prompt) pair to the consolidated dataset,
    tagged with::

        source  = "L'Électron Rare internal (niche curation)"
        license = "apache-2.0"

    The dict key is the domain label as it appears in the router schema
    (e.g. "kicad-pcb"); the value is a list of free-form user prompts.

This file is meant to be edited collaboratively — a human curator adds more
prompts in batches as router quality regressions are observed. Keep each
list a flat top-level Python list (no f-strings, no comprehensions) so that
diff-review stays trivial.

Conventions per prompt:
    * 8 to 300 characters
    * FR or EN (mix is fine — the router sees both)
    * Mentions domain-specific tools / components / workflows
    * Avoid strings that could plausibly belong to two domains
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# kicad-pcb — KiCad PCB layout, DRC, Gerber, footprint placement
# ---------------------------------------------------------------------------
KICAD_PCB = [
    "Comment configurer les règles DRC dans KiCad pour un PCB 4 couches impédance contrôlée ?",
    "Génère les fichiers Gerber et le drill map pour fabrication chez JLCPCB depuis KiCad 8.",
    "My KiCad PCB has unrouted ratlines on GND — how do I add a stitched ground pour properly?",
    "Quelle largeur de piste pour 3 A en cuivre 1 oz extérieur sur un PCB KiCad ?",
    "How do I import a STEP body for a custom footprint in KiCad's 3D viewer?",
    "Configure des keep-out zones autour d'une antenne PCB chip dans pcbnew.",
    "KiCad DRC complains about courtyard overlap between two QFN packages — how to fix without moving them?",
    "Comment exporter une netlist BOM CSV depuis KiCad pour PCBA chez PCBWay ?",
    "Set up differential pair routing for USB 2.0 D+/D- in KiCad with length matching.",
    "Why does my KiCad zone refill change copper pour after every save? Is there a stable workflow?",
    "Place les vias de découplage autour d'un BGA 0.8 mm pitch dans KiCad pcbnew.",
    "Génère un panneau v-cut 2x3 pour mes PCB KiCad avec rails de fabrication.",
]

# ---------------------------------------------------------------------------
# kicad-dsl — KiCad schematic editor, hierarchy, symbols, ERC
# ---------------------------------------------------------------------------
KICAD_DSL = [
    "Crée une feuille hiérarchique dans KiCad eeschema pour isoler la section alimentation.",
    "How do I define a custom schematic symbol with multiple units (e.g. quad opamp) in KiCad?",
    "ERC dans KiCad signale 'pin not driven' sur mon rail 3V3 — comment ajouter un PWR_FLAG ?",
    "Migrate an Altium schematic library to KiCad symbol format — what's the cleanest path?",
    "Annote automatiquement tous les composants d'un schéma KiCad multi-feuilles.",
    "How to link a datasheet PDF to a KiCad symbol so it opens from eeschema?",
    "Configure des bus labels et bus entries pour un schéma de mémoire SDRAM dans KiCad.",
    "Why does KiCad's ERC report unconnected hierarchical pins even though they are wired?",
    "Add a custom field 'LCSC' to all symbols in a KiCad schematic and export it to BOM.",
    "Différence entre un net label local et un global label dans eeschema ?",
    "Crée un symbole KiCad pour un connecteur Molex 5566 14 pôles avec les pin numbers corrects.",
]

# ---------------------------------------------------------------------------
# stm32 — STM32 / ARM Cortex-M firmware (HAL, CubeIDE, FreeRTOS)
# ---------------------------------------------------------------------------
STM32 = [
    "Configure un timer TIM2 en PWM 20 kHz sur STM32F411 avec STM32CubeMX et HAL.",
    "Why does HAL_UART_Receive_IT only fire once on my STM32G0? Need continuous reception.",
    "Set up FreeRTOS with two tasks on STM32H7 using CubeIDE and CMSIS-RTOS v2 wrapper.",
    "Comment lire un capteur I2C avec DMA sur STM32L4 sans bloquer le scheduler RTOS ?",
    "STM32 hard fault handler — how to decode the stacked PC and LR to find the offending line?",
    "Configure le low-power stop mode sur STM32L0 avec wakeup via RTC alarm toutes les 10 s.",
    "Migrate from HAL to LL drivers on STM32F4 to shrink flash footprint — what's the gotcha list?",
    "Implémente un bootloader custom STM32F103 avec saut vers application en 0x08008000.",
    "ADC injected channel + DMA circular on STM32G4 — sample 4 channels at 50 kHz, how?",
    "Pourquoi mon CAN bus STM32F7 ne reçoit aucune trame malgré un bitrate correct ?",
    "Set up SWO trace on STM32H743 for printf debugging via ST-Link without a UART.",
    "STM32CubeIDE rebuilds everything on every save — how to fix incremental build?",
]

# ---------------------------------------------------------------------------
# embedded — Generic embedded C, drivers, ISR, low-level (non-STM32-specific)
# ---------------------------------------------------------------------------
EMBEDDED = [
    "Write a circular ring buffer in C99 safe for single-producer single-consumer ISR context.",
    "Pourquoi mon ISR se déclenche en boucle ? J'ai oublié de clear le pending flag, où le faire ?",
    "Implement a non-blocking debounce state machine for a mechanical switch in pure C.",
    "How do I avoid a race condition between main loop and an ISR that share a 32-bit counter?",
    "Crée un driver SPI bit-bang en C pour un MCU sans périphérique SPI matériel.",
    "Memory-mapped register access in C — when do I need volatile and when do I need a memory barrier?",
    "Aligne une structure C sur 4 octets pour un MCU Cortex-M0 sans __attribute__((packed)).",
    "Embedded C: how to detect stack overflow at runtime without an MMU?",
    "Implémente une CRC-16 CCITT table-driven en C pour un protocole série custom.",
    "What's the right way to expose a hardware peripheral as a singleton driver in C?",
    "Write a minimal cooperative scheduler in 200 lines of C for a Cortex-M0+ without RTOS.",
]

# ---------------------------------------------------------------------------
# dsp — Digital signal processing (FIR/IIR, FFT, scipy.signal)
# ---------------------------------------------------------------------------
DSP = [
    "Design a 4th-order Butterworth bandpass IIR filter 300-3400 Hz in scipy.signal.",
    "Calcule la FFT d'un signal audio 48 kHz et identifie le pic principal en Python avec numpy.",
    "Why does my FIR filter introduce group delay and how do I compensate it for real-time use?",
    "Implement an overlap-add convolution in numpy for a 4096-tap FIR filter on streaming audio.",
    "Conçois un filtre coupe-bande 50 Hz pour ECG avec scipy.signal et trace la réponse fréquentielle.",
    "What's the difference between a Hamming and a Blackman window for spectral leakage reduction?",
    "Compute the magnitude and phase response of an IIR biquad given its 5 coefficients.",
    "Resample a 44.1 kHz signal to 48 kHz using polyphase filtering in scipy without aliasing.",
    "Détecte le pitch fondamental d'une voix par autocorrélation en Python — algorithme YIN ?",
    "Apply a Kalman filter to fuse accelerometer and gyroscope for tilt estimation.",
    "Why does my STFT spectrogram show vertical bands at every window boundary?",
]

# ---------------------------------------------------------------------------
# iot — MQTT, BLE, LoRaWAN, ESP32 networking
# ---------------------------------------------------------------------------
IOT = [
    "Set up an ESP32 MQTT client with TLS to HiveMQ Cloud using ESP-IDF v5.4 and mbedTLS.",
    "Configure une node LoRaWAN classe A en OTAA sur RAK3172 avec keys de The Things Network.",
    "Why does my BLE GATT notification stop after 30 seconds on ESP32 NimBLE stack?",
    "Implement a Wi-Fi provisioning flow over BLE for ESP32 with smartphone companion app.",
    "MQTT QoS 1 vs QoS 2 — when does the latency cost actually matter for an IoT sensor fleet?",
    "Configure ESP32 deep sleep wakeup via GPIO and RTC timer to last 6 months on a CR2032.",
    "Crée une advertising packet BLE custom de 31 octets contenant température et humidité.",
    "Bridge a Zigbee Hue bulb to MQTT via Zigbee2MQTT on a Raspberry Pi.",
    "How to handle MQTT reconnection backoff in firmware when the broker is unreachable?",
    "Implémente OTA update sécurisé sur ESP32 via HTTPS depuis un bucket S3 signé.",
    "Why does my LoRaWAN gateway drop packets above SF10? Duty cycle or RX window issue?",
]

# ---------------------------------------------------------------------------
# music-audio — Audio synthesis, VST, MIDI, JUCE
# ---------------------------------------------------------------------------
MUSIC_AUDIO = [
    "Build a polyphonic subtractive synthesizer VST3 plugin in JUCE 7 with 8-voice voice stealing.",
    "Comment router un signal MIDI clock depuis Ableton vers un module Eurorack via interface USB ?",
    "Implement a Karplus-Strong plucked string algorithm in C++ for real-time audio at 96 kHz.",
    "JUCE AudioProcessor: where do I allocate buffers safely without locking the audio thread?",
    "Crée un patch Pure Data pour un générateur de drone évolutif avec 3 oscillateurs FM.",
    "Why does my VST plugin click when I change a parameter? Need parameter smoothing — how?",
    "Map MIDI CC 74 to a low-pass filter cutoff in a JUCE synth using AudioProcessorValueTreeState.",
    "Conçois un compresseur audio sidechain en C++ avec attaque 5 ms et release 100 ms.",
    "What's the difference between AU, VST3, and CLAP plugin formats for cross-DAW compatibility?",
    "Implémente un convolution reverb en JUCE avec impulse response chargée depuis un WAV.",
    "Build a granular sampler in SuperCollider with grain density and pitch jitter controls.",
]

# ---------------------------------------------------------------------------
# platformio — PlatformIO build system, multi-env, debugger
# ---------------------------------------------------------------------------
PLATFORMIO = [
    "Configure platformio.ini avec deux environnements debug et release pour ESP32 et nrf52.",
    "How do I add a custom upload_flags entry in PlatformIO for a J-Link probe on STM32?",
    "PlatformIO library.json — déclare une dépendance privée Git avec un tag spécifique.",
    "Why does PlatformIO rebuild the whole framework every time I change a single source file?",
    "Set up unit tests with PlatformIO Unity framework running both on host and on hardware.",
    "Switch PlatformIO from Arduino framework to ESP-IDF for the same ESP32 board — what changes?",
    "Crée une custom board JSON dans PlatformIO pour un Cortex-M0+ non listé officiellement.",
    "Configure platformio debug avec OpenOCD et un ST-Link v2 pour breakpoints dans VS Code.",
    "Override the linker script in PlatformIO for a custom flash partition layout.",
    "Why does pio run --target upload fail with 'no device found' but esptool.py works fine?",
    "Add pre and post build scripts in platformio.ini to embed git hash into firmware.",
]

# ---------------------------------------------------------------------------
# ml-training — PyTorch, Hugging Face PEFT/LoRA, training loops
# ---------------------------------------------------------------------------
ML_TRAINING = [
    "Fine-tune Llama 3.1 8B with LoRA rank 16 on a custom JSONL dataset using Hugging Face PEFT.",
    "Pourquoi ma loss explose à l'epoch 3 ? Gradient clipping ou learning rate trop élevé ?",
    "Set up DDP multi-GPU training in PyTorch 2.4 with mixed precision and gradient accumulation.",
    "Implement a custom torch.utils.data.Dataset for variable-length tokenized sequences with padding.",
    "Compare full fine-tuning vs QLoRA 4-bit on a 13B model — VRAM and quality tradeoffs?",
    "Configure HuggingFace TrainingArguments avec early stopping et best checkpoint sur eval_loss.",
    "How do I freeze the embedding layer of a transformer in PyTorch while training only the head?",
    "Why is my GPU at 30% utilization during training? DataLoader workers or pin_memory issue?",
    "Implémente un focal loss en PyTorch pour classification multi-label déséquilibrée.",
    "Convert a fine-tuned model to GGUF and quantize Q4_K_M for llama.cpp inference.",
    "Set up Weights & Biases logging with a PyTorch Lightning training loop and log gradient histograms.",
    "Pourquoi le merge des LoRA adapters change-t-il la performance sur le benchmark ?",
]

# ---------------------------------------------------------------------------
# llm-orch — LiteLLM, LangChain, vLLM, RAG
# ---------------------------------------------------------------------------
LLM_ORCH = [
    "Configure LiteLLM proxy with model fallback from gpt-4o to claude-3.5-sonnet on rate limit.",
    "Set up a RAG pipeline with LangChain, Chroma, and OpenAI embeddings for a 10k-doc corpus.",
    "Pourquoi vLLM serve crash avec OOM sur un Llama 70B en bf16 sur 2x A100 80GB ?",
    "Build a multi-agent workflow with LangGraph where a researcher agent feeds a writer agent.",
    "How do I stream tokens from a LiteLLM router to a FastAPI SSE endpoint?",
    "Implémente une stratégie de chunking sémantique pour RAG sur des PDF techniques avec tables.",
    "Compare LlamaIndex vs LangChain for a production RAG with hybrid search (BM25 + dense).",
    "Configure vLLM avec speculative decoding et tensor parallelism sur 4 GPU H100.",
    "Why does my LangChain agent enter an infinite tool-calling loop on simple questions?",
    "Set up Ollama as a backend behind LiteLLM with custom model aliasing and per-key budgets.",
    "Add a re-ranker (Cohere or BGE) to my LangChain RAG retrieval pipeline — where in the chain?",
    "Implémente un cache Redis pour les embeddings OpenAI afin de réduire les coûts en RAG.",
]

# ---------------------------------------------------------------------------
# web-backend — FastAPI, Express, NestJS, REST API design
# ---------------------------------------------------------------------------
WEB_BACKEND = [
    "Implémente une route FastAPI POST /users avec validation Pydantic v2 et SQLAlchemy 2.0 async.",
    "Set up JWT authentication with refresh tokens in NestJS using Passport and @nestjs/jwt.",
    "Why does my Express middleware not catch async errors? Need express-async-handler or v5?",
    "Design a paginated REST endpoint with cursor-based pagination — query params and response shape?",
    "Configure FastAPI avec SQLAlchemy async, Alembic migrations et dependency injection des sessions.",
    "Add rate limiting per API key in a FastAPI app using slowapi and Redis backend.",
    "How do I structure a NestJS monorepo with shared DTOs between microservices?",
    "Implement WebSocket broadcast in FastAPI with a connection manager and Redis pub/sub.",
    "Pourquoi mon endpoint Express renvoie 502 derrière nginx ? Timeout ou keep-alive ?",
    "Compare gRPC vs REST for an internal service-to-service API with strong typing requirements.",
    "Set up OpenAPI 3.1 schema generation with examples in FastAPI and validate against Spectral.",
    "Implémente un upload de fichier streaming en FastAPI sans charger le fichier complet en RAM.",
]

# ---------------------------------------------------------------------------
# web-frontend — React, Vite, Tailwind, TanStack Router
# ---------------------------------------------------------------------------
WEB_FRONTEND = [
    "Set up TanStack Router with file-based routes in a Vite + React 19 + TypeScript project.",
    "Pourquoi mon composant React re-render à chaque keystroke ? useMemo ou React.memo manquant ?",
    "Configure Tailwind v4 avec un thème dark/light basé sur prefers-color-scheme et CSS variables.",
    "Implement an infinite scroll list in React with TanStack Query useInfiniteQuery and IntersectionObserver.",
    "How do I share state between sibling components in React without prop drilling — Context vs Zustand?",
    "Migrate a Create React App project to Vite — what config gotchas should I expect?",
    "Crée un formulaire React Hook Form avec validation Zod et erreurs traduites en français.",
    "Why does my Tailwind class not apply in production build but works in dev? PurgeCSS issue?",
    "Set up Storybook 8 with Vite and Tailwind for a component library with MDX docs.",
    "Implement optimistic updates in TanStack Query for a todo list with rollback on error.",
    "Pourquoi mon useEffect se déclenche deux fois au mount en React 18 ? StrictMode dev mode ?",
    "Build a drag-and-drop kanban board in React with dnd-kit and persistent column state.",
]

# ---------------------------------------------------------------------------
# yaml-json — Kubernetes manifests, OpenAPI, GitHub Actions YAML
# ---------------------------------------------------------------------------
YAML_JSON = [
    "Write a Kubernetes Deployment YAML with rolling update strategy, resource limits, and a liveness probe.",
    "Crée un workflow GitHub Actions YAML qui build une image Docker et la push sur GHCR à chaque tag.",
    "Why does my Kubernetes ConfigMap not reload in the pod after kubectl apply? Need a restart trigger?",
    "Convert this OpenAPI 3.0 spec to OpenAPI 3.1 and fix the nullable / type array changes.",
    "Set up a GitHub Actions matrix build for Node 18, 20, 22 across ubuntu and macos runners.",
    "Écris un Helm chart values.yaml override pour activer ingress TLS avec cert-manager.",
    "Validate a Kubernetes manifest against the schema with kubeconform before applying — CI step?",
    "Explain the difference between a Kubernetes Service ClusterIP, NodePort, and LoadBalancer in YAML.",
    "Add a reusable composite GitHub Actions workflow callable from multiple repos.",
    "Pourquoi mon manifest Kubernetes échoue avec 'invalid value for field spec.template.spec.containers' ?",
    "Generate an OpenAPI schema with discriminated unions for a polymorphic 'event' object.",
    "Write a docker-compose.yml with healthchecks, depends_on conditions, and a named volume for postgres.",
]

# ---------------------------------------------------------------------------
# Aggregate — consumed by scripts/rebuild_router_dataset.py
# ---------------------------------------------------------------------------
NICHE_DOMAIN_PROMPTS: dict[str, list[str]] = {
    "kicad-pcb": KICAD_PCB,
    "kicad-dsl": KICAD_DSL,
    "stm32": STM32,
    "embedded": EMBEDDED,
    "dsp": DSP,
    "iot": IOT,
    "music-audio": MUSIC_AUDIO,
    "platformio": PLATFORMIO,
    "ml-training": ML_TRAINING,
    "llm-orch": LLM_ORCH,
    "web-backend": WEB_BACKEND,
    "web-frontend": WEB_FRONTEND,
    "yaml-json": YAML_JSON,
}
