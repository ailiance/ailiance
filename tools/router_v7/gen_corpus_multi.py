#!/usr/bin/env python3
"""Multi-model corpus generator for router v7.

Generates training prompts per domain by querying multiple LLMs through the
ailiance gateway (:9300). Sequential per domain (avoid Studio OOM), parallel
per model within a domain (cap ~3).

Output: data/router-v7-multimodel-raw/<domain>.jsonl
"""
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx

GATEWAY = "http://localhost:9300/v1/chat/completions"
OUT_DIR = Path("/home/electron/ailiance/data/router-v7-multimodel-raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / "_gen.log"

# Models that are FAST + reliable.  Avoid Mistral-Medium 128B (3 tok/s),
# Apertus (slow), Pixtral (multimodal).
FAST_FR = "ailiance-mistral-small"            # 24B 4-bit
FAST_EN = "ailiance-gemma"                    # Gemma-3-4B Tower
GEMMA4 = "ailiance-gemma4"                    # macm1
MINISTRAL = "ailiance-ministral"              # 14B macm1
QWEN_CODER = "ailiance-coder-pro"             # Qwen3-Coder-30B
GRANITE = "ailiance-granite"                  # Granite-30B
R1 = "ailiance-reasoning-r1"                  # DeepSeek-R1
QWEN = "ailiance-qwen"                        # Qwen3-Next-80B
LLAMA = "ailiance-llama"

# Devstral LoRA (code-domain specialists)
PY = "ailiance-python"
CPP = "ailiance-cpp"
RUST = "ailiance-rust-emb"
HTML = "ailiance-html"
ML = "ailiance-ml-training"

# Mascarade hardware LoRA
KICAD = "ailiance-kicad"
SPICE = "ailiance-spice"
STM32 = "ailiance-stm32"
EMC = "ailiance-emc"
EMBEDDED = "ailiance-embedded"
PLATFORMIO = "ailiance-platformio"
FREECAD = "ailiance-freecad"
DSP = "ailiance-dsp"
IOT = "ailiance-iot"
POWER = "ailiance-power"

# Per-domain config: description + list of (model, n) tuples
N_DEFAULT = 60  # examples per model per domain

DOMAIN_CONFIG = {
    # ----- FR / multilingue -----
    "chat-fr": {
        "desc": "Conversation française informelle, salutations, questions générales",
        "models": [(FAST_FR, 80), (MINISTRAL, 80), (FAST_EN, 60)],
        "lang": "French",
    },
    "traduction-tech": {
        "desc": "Translate technical documents EN<->FR, terminology requests, glossary lookups",
        "models": [(FAST_FR, 70), (MINISTRAL, 70), (GRANITE, 60)],
        "lang": "French and English mixed",
    },
    "redaction-multilingue": {
        "desc": "Write multilingual marketing copy, technical reports in FR/EN/DE/ES",
        "models": [(FAST_FR, 70), (MINISTRAL, 60), (GRANITE, 60)],
        "lang": "French and English",
    },
    "localisation-doc": {
        "desc": "Localize software UI strings, i18n keys, locale resource files",
        "models": [(FAST_FR, 60), (GRANITE, 60), (QWEN_CODER, 60)],
        "lang": "French and English",
    },
    # ----- code -----
    "python": {
        "desc": "Python code: scripts, debugging, libraries (pandas, numpy, fastapi, django)",
        "models": [(QWEN_CODER, 80), (PY, 80), (R1, 60)],
        "lang": "English with occasional French",
    },
    "cpp": {
        "desc": "C++ code: STL, templates, embedded C++, memory management",
        "models": [(QWEN_CODER, 70), (CPP, 70), (R1, 60)],
        "lang": "English",
    },
    "rust": {
        "desc": "Rust code: ownership, lifetimes, tokio, async, cargo",
        "models": [(QWEN_CODER, 70), (RUST, 70), (R1, 60)],
        "lang": "English",
    },
    "typescript": {
        "desc": "TypeScript: types, generics, React with TS, node + ts, tsconfig",
        "models": [(QWEN_CODER, 70), (FAST_EN, 60), (GEMMA4, 60)],
        "lang": "English",
    },
    "shell": {
        "desc": "Shell scripts: bash, zsh, awk, sed, pipelines, sysadmin one-liners",
        "models": [(QWEN_CODER, 70), (FAST_EN, 60), (GRANITE, 60)],
        "lang": "English with French",
    },
    "sql": {
        "desc": "SQL queries: postgres, mysql, joins, CTEs, window functions, schema design",
        "models": [(QWEN_CODER, 70), (GRANITE, 60), (FAST_EN, 60)],
        "lang": "English",
    },
    "web-backend": {
        "desc": "Backend web: REST APIs, FastAPI, Express, Django, auth, middleware, ORMs",
        "models": [(QWEN_CODER, 70), (PY, 60), (GRANITE, 60)],
        "lang": "English",
    },
    "web-frontend": {
        "desc": "Frontend web: React, Vue, Svelte, state mgmt, hooks, components",
        "models": [(QWEN_CODER, 70), (HTML, 60), (GEMMA4, 60)],
        "lang": "English",
    },
    "html-css": {
        "desc": "HTML markup and CSS styling: flexbox, grid, animations, responsive design",
        "models": [(HTML, 80), (QWEN_CODER, 60), (FAST_EN, 60)],
        "lang": "English",
    },
    "docker": {
        "desc": "Docker: Dockerfiles, compose, networking, volumes, multi-stage builds",
        "models": [(QWEN_CODER, 70), (GRANITE, 60), (FAST_EN, 60)],
        "lang": "English",
    },
    "devops": {
        "desc": "DevOps: CI/CD, GitHub Actions, Kubernetes, Terraform, ansible, monitoring",
        "models": [(QWEN_CODER, 70), (GRANITE, 60), (FAST_EN, 60)],
        "lang": "English",
    },
    "yaml-json": {
        "desc": "YAML/JSON configuration files, schemas, validation, parsing",
        "models": [(QWEN_CODER, 60), (FAST_EN, 60), (GEMMA4, 50)],
        "lang": "English",
    },
    "llm-ops": {
        "desc": "LLM operations: serving (vllm, llama.cpp), quantization, evaluation, monitoring",
        "models": [(QWEN_CODER, 60), (ML, 60), (GRANITE, 60)],
        "lang": "English",
    },
    "llm-orch": {
        "desc": "LLM orchestration: LangChain, LangGraph, agents, tool use, RAG pipelines",
        "models": [(QWEN_CODER, 60), (R1, 60), (ML, 60)],
        "lang": "English",
    },
    "ml-training": {
        "desc": "ML training: PyTorch, transformers, LoRA fine-tuning, distillation, optimizer",
        "models": [(ML, 70), (QWEN_CODER, 60), (R1, 60)],
        "lang": "English",
    },
    "lua-upy": {
        "desc": "Lua scripts and MicroPython (upy) for embedded devices, neovim configs",
        "models": [(QWEN_CODER, 60), (EMBEDDED, 60), (FAST_EN, 50)],
        "lang": "English",
    },
    # ----- reasoning / math -----
    "reasoning": {
        "desc": "Multi-step logical reasoning, deductive puzzles, step-by-step problem analysis",
        "models": [(R1, 80), (QWEN, 70), (MINISTRAL, 60)],
        "lang": "English and French",
    },
    "math": {
        "desc": "Mathematical problems: algebra, calculus, linear algebra, probability, proofs",
        "models": [(R1, 80), (QWEN, 70), (GRANITE, 60)],
        "lang": "English and French",
    },
    # ----- hardware mascarade -----
    "kicad": {
        "desc": "KiCad ECAD: schematic and PCB questions, library, footprints, netlists",
        "models": [(KICAD, 100), (FAST_FR, 60), (MINISTRAL, 60)],
        "lang": "English and French",
    },
    "kicad-dsl": {
        "desc": "KiCad text/S-expression DSL: parsing .kicad_sch, .kicad_pcb files programmatically",
        "models": [(KICAD, 80), (GEMMA4, 60), (QWEN_CODER, 60)],
        "lang": "English",
    },
    "kicad-pcb": {
        "desc": "KiCad PCB layout specifics: routing, trace width, vias, layers, copper pours, DRC",
        "models": [(KICAD, 80), (GEMMA4, 60), (FAST_FR, 60)],
        "lang": "English and French",
    },
    "spice": {
        "desc": "SPICE simulation: ngspice, LTspice, analog circuit netlists, transient analysis",
        "models": [(SPICE, 100), (FAST_FR, 60), (R1, 60)],
        "lang": "English and French",
    },
    "stm32": {
        "desc": "STM32 microcontroller: HAL, LL drivers, CubeMX, peripherals, freertos on STM32",
        "models": [(STM32, 100), (EMBEDDED, 60), (CPP, 60)],
        "lang": "English",
    },
    "emc": {
        "desc": "EMC/EMI: pre-compliance, FCC/CISPR, ground planes, shielding, ferrites, filtering",
        "models": [(EMC, 100), (FAST_FR, 60), (MINISTRAL, 60)],
        "lang": "English and French",
    },
    "embedded": {
        "desc": "Embedded systems: RTOS, bare-metal, interrupts, DMA, low-power, drivers",
        "models": [(EMBEDDED, 100), (STM32, 60), (CPP, 60)],
        "lang": "English",
    },
    "platformio": {
        "desc": "PlatformIO build system: platformio.ini, libraries, boards, frameworks",
        "models": [(PLATFORMIO, 100), (EMBEDDED, 60), (QWEN_CODER, 60)],
        "lang": "English",
    },
    "freecad": {
        "desc": "FreeCAD mechanical CAD: sketches, parts, assemblies, PartDesign workbench, python macros",
        "models": [(FREECAD, 100), (FAST_FR, 60), (MINISTRAL, 60)],
        "lang": "English and French",
    },
    "dsp": {
        "desc": "DSP: FIR/IIR filters, FFT, audio signal processing, sample rate conversion",
        "models": [(DSP, 100), (R1, 60), (CPP, 60)],
        "lang": "English",
    },
    "iot": {
        "desc": "IoT: MQTT, LoRaWAN, BLE, Zigbee, ESP32 wifi, edge sensors, home automation",
        "models": [(IOT, 100), (EMBEDDED, 60), (FAST_EN, 60)],
        "lang": "English and French",
    },
    "power": {
        "desc": "Power electronics: SMPS, LDOs, buck/boost, MOSFET drivers, battery management, PFC",
        "models": [(POWER, 100), (FAST_FR, 60), (MINISTRAL, 60)],
        "lang": "English and French",
    },
    # ----- technical norms / certif -----
    "electronics-hw": {
        "desc": "General hardware electronics: op-amps, logic gates, PCB design rules, components",
        "models": [(FAST_FR, 70), (MINISTRAL, 70), (GRANITE, 60)],
        "lang": "English and French",
    },
    "security": {
        "desc": "Cybersecurity: pentest, OWASP, crypto, TLS, secure coding, vulnerability analysis",
        "models": [(QWEN_CODER, 70), (GRANITE, 60), (R1, 60)],
        "lang": "English",
    },
    "music-audio": {
        "desc": "Music + audio: DAWs, MIDI, synthesis, mixing, mastering, plugins, SuperCollider",
        "models": [(FAST_FR, 70), (MINISTRAL, 60), (FAST_EN, 60)],
        "lang": "English and French",
    },
    "autosar-cert": {
        "desc": "AUTOSAR automotive standard: BSW, RTE, ARXML, certification process, ASIL",
        "models": [(MINISTRAL, 80), (FAST_FR, 60), (GRANITE, 60)],
        "lang": "English and French",
    },
    "doc-technique-ce": {
        "desc": "CE technical documentation: dossier technique, EU declaration of conformity, harmonised standards",
        "models": [(FAST_FR, 80), (MINISTRAL, 70), (GRANITE, 50)],
        "lang": "French primarily",
    },
    "misra-c": {
        "desc": "MISRA C/C++ coding standard for safety-critical embedded software, rule compliance",
        "models": [(CPP, 80), (MINISTRAL, 60), (QWEN_CODER, 60)],
        "lang": "English",
    },
    "normes-iec": {
        "desc": "IEC standards: IEC 61508 functional safety, IEC 60601 medical, IEC 62443 cybersecurity",
        "models": [(FAST_FR, 70), (MINISTRAL, 70), (GRANITE, 60)],
        "lang": "French and English",
    },
    "calcul-normatif": {
        "desc": "Normative calculations: stress, EMC margin, derating, thermal, regulatory thresholds",
        "models": [(R1, 70), (FAST_FR, 60), (MINISTRAL, 60)],
        "lang": "French and English",
    },
    # ----- meta-tasks -----
    "general": {
        "desc": "General queries that don't fit any specialized domain: lifestyle, world facts, opinions",
        "models": [(FAST_EN, 80), (FAST_FR, 70), (GEMMA4, 60)],
        "lang": "English and French",
    },
    "quick": {
        "desc": "Very short factual questions, one-liners, quick lookups (capital of X, definition of Y)",
        "models": [(FAST_EN, 80), (GEMMA4, 70), (FAST_FR, 60)],
        "lang": "English and French",
    },
    "summarize": {
        "desc": "Requests to summarize text, articles, papers, meeting notes into bullet points",
        "models": [(FAST_FR, 70), (FAST_EN, 60), (MINISTRAL, 60)],
        "lang": "English and French",
    },
    "tldr": {
        "desc": "TL;DR requests: ultra-short condensed summary, key takeaways in 1-3 sentences",
        "models": [(FAST_EN, 70), (FAST_FR, 60), (GEMMA4, 60)],
        "lang": "English and French",
    },
    "classification": {
        "desc": "Text classification tasks: sentiment, topic labelling, intent detection requests",
        "models": [(GEMMA4, 70), (FAST_EN, 60), (GRANITE, 60)],
        "lang": "English",
    },
}

assert len(DOMAIN_CONFIG) == 47, f"Got {len(DOMAIN_CONFIG)} domains, expected 47"

META_PROMPT = """You are generating training data for a domain classifier (NOT answering questions).

Generate exactly {n} realistic user prompts that should be classified as domain "{domain}".

Domain "{domain}" means: {desc}

Language: {lang}. Vary length (5-200 words), tone (formal/informal/technical/curious), and format (question/imperative/statement). Include some short prompts (<10 words) and a few longer multi-sentence prompts.

OUTPUT FORMAT (strict): JSON Lines, one object per line, no markdown fences, no preamble, no commentary.
Each line: {{"text": "the prompt here", "label": "{domain}"}}

Generate {n} distinct prompts now:"""


def parse_jsonl(content: str, domain: str) -> list[dict]:
    out = []
    # Strip markdown fences if present
    content = re.sub(r"```(?:jsonl|json)?\s*\n?", "", content)
    content = content.replace("```", "")
    for line in content.split("\n"):
        line = line.strip().lstrip("- ").lstrip("* ").strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            ex = json.loads(line)
        except Exception:
            continue
        text = ex.get("text") or ex.get("prompt")
        if not text or not isinstance(text, str):
            continue
        text = text.strip()
        words = text.split()
        if len(words) < 3 or len(words) > 500:
            continue
        out.append({"text": text, "label": domain})
    return out


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


async def gen_one(client, domain: str, model: str, n: int, lang: str, attempt=1) -> list[dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": META_PROMPT.format(
            n=n, domain=domain, desc=DOMAIN_CONFIG[domain]["desc"], lang=lang,
        )}],
        "max_tokens": min(8000, 80 + n * 50),
        "temperature": 0.9,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    t0 = time.time()
    try:
        r = await client.post(GATEWAY, json=payload, timeout=400.0)
        if r.status_code != 200:
            log(f"  [{domain}] {model}: HTTP {r.status_code} {r.text[:100]}")
            return []
        content = r.json()["choices"][0]["message"]["content"]
        examples = parse_jsonl(content, domain)
        dt = time.time() - t0
        log(f"  [{domain}] {model}: {len(examples)}/{n} in {dt:.1f}s")
        if len(examples) < n * 0.3 and attempt == 1:
            # retry with a different temperature
            payload["temperature"] = 0.7
            await asyncio.sleep(1)
            r2 = await client.post(GATEWAY, json=payload, timeout=400.0)
            if r2.status_code == 200:
                c2 = r2.json()["choices"][0]["message"]["content"]
                ex2 = parse_jsonl(c2, domain)
                log(f"  [{domain}] {model}: retry +{len(ex2)}")
                examples.extend(ex2)
        # Tag with source model
        for e in examples:
            e["_model"] = model
        return examples
    except Exception as e:
        log(f"  [{domain}] {model}: EXC {type(e).__name__}: {str(e)[:120]}")
        return []


async def gen_domain(client, domain: str, cfg: dict) -> list[dict]:
    out_file = OUT_DIR / f"{domain}.jsonl"
    if out_file.exists() and out_file.stat().st_size > 1000:
        existing = [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
        if len(existing) >= 100:
            log(f"=== {domain}: SKIP (already {len(existing)} examples)")
            return existing
    log(f"=== {domain} ({len(cfg['models'])} models)")
    tasks = [gen_one(client, domain, m, n, cfg["lang"]) for (m, n) in cfg["models"]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_ex = []
    for r in results:
        if isinstance(r, list):
            all_ex.extend(r)
    # dedupe by text
    seen = set()
    unique = []
    for e in all_ex:
        t = e["text"].strip().lower()
        if t in seen:
            continue
        seen.add(t)
        unique.append(e)
    out_file.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in unique))
    target = sum(n for _, n in cfg["models"])
    log(f"=== {domain}: {len(unique)} unique (target {target})")
    return unique


async def main():
    log(f"=== START {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Domains: {len(DOMAIN_CONFIG)}")
    # Allow 4 concurrent domains × up to 3 models = up to 12 concurrent calls,
    # but the gateway routes them to different backends (Tower/Studio/macm1/kxkm).
    DOMAIN_CONCURRENCY = 2
    sem = asyncio.Semaphore(DOMAIN_CONCURRENCY)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=24)) as client:
        r = await client.get("http://localhost:9300/v1/models", timeout=10)
        log(f"Gateway: {r.status_code}, {len(r.json()['data'])} models exposed")
        order = list(DOMAIN_CONFIG.keys())

        async def _wrap(idx, domain):
            async with sem:
                log(f"--- [{idx+1}/{len(order)}] {domain} START")
                return await gen_domain(client, domain, DOMAIN_CONFIG[domain])

        tasks = [asyncio.create_task(_wrap(i, d)) for i, d in enumerate(order)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total = sum(len(r) for r in results if isinstance(r, list))
    log(f"\n=== DONE. Total {total} examples across {len(DOMAIN_CONFIG)} domains.")


if __name__ == "__main__":
    asyncio.run(main())
