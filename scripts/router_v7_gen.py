#!/usr/bin/env python3
"""Generate router v7 corpus (47-label) using ailiance-mistral-small via gateway.

Concurrency: 2 parallel streams against the gateway to amortize TTFT.
Writes one JSONL per domain in data/router-v7-raw/<domain>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:9300/v1/chat/completions")
MODEL = os.environ.get("GEN_MODEL", "ailiance-mistral-small")

# 47 domains with concise descriptions for the meta-prompt.
DOMAINS: dict[str, str] = {
    # APERTUS (19)
    "electronics-hw": "hardware electronics generic — discrete components, schematics, soldering, prototyping",
    "emc": "electromagnetic compatibility — EMC testing, EMI, CISPR, FCC, shielding, grounding",
    "dsp": "digital signal processing — FIR, IIR, FFT, audio, filtering, sampling theory",
    "spice": "SPICE simulation — ngspice, LTspice, .tran/.ac/.dc, netlists, opamp circuits",
    "kicad": "KiCad EDA generic — schematic capture, library, footprint, project setup",
    "stm32": "STM32 microcontroller — STM32CubeIDE, HAL, RTOS on STM32, peripherals",
    "platformio": "PlatformIO build system — platformio.ini, libraries, framework selection",
    "iot": "Internet of Things — ESP32, MQTT, LoRa, Zigbee, sensors, edge devices",
    "embedded": "embedded systems generic — bare-metal, RTOS, bootloader, hardware bring-up",
    "math": "mathematics — algebra, calculus, linear algebra, equations, proofs",
    "security": "cybersecurity — vulnerabilities, auth, crypto, pentest, OWASP, CVEs",
    "music-audio": "music and audio production — DAW, synths, mixing, mastering, audio engineering",
    "freecad": "FreeCAD CAD — parametric modeling, sketches, parts, drafts, workbenches",
    "power": "power electronics — DC-DC, LDO, buck/boost, PFC, battery management, MOSFETs",
    "misra-c": "MISRA-C compliance — coding standards for safety-critical C (rules, violations)",
    "autosar-cert": "AUTOSAR and automotive certification — ISO 26262, ASIL, SWC, CAN, ECU",
    "doc-technique-ce": "CE technical documentation — declarations of conformity, file CE, EU directives",
    "calcul-normatif": "normative calculations — code-prescribed formulas, regulatory engineering math",
    "normes-iec": "IEC standards — IEC 61010, IEC 60601, IEC 61508, standard interpretation",
    # QWEN (1)
    "reasoning": "multi-step logical reasoning, deduction, chain-of-thought problems, riddles",
    # DEVSTRAL (16)
    "python": "Python programming — scripts, libraries (numpy, pandas, requests, asyncio), debugging",
    "rust": "Rust programming — ownership, lifetimes, async, cargo, traits, no_std",
    "typescript": "TypeScript programming — types, generics, decorators, tsconfig, npm packages",
    "cpp": "C++ programming — templates, STL, RAII, modern C++, CMake, debugging",
    "shell": "shell scripting — bash, zsh, awk, sed, grep, pipes, POSIX tools",
    "html-css": "HTML and CSS — markup, layout, flexbox, grid, animations, responsive design",
    "sql": "SQL and database queries — SELECT, JOIN, indexes, query optimization, schema design",
    "web-backend": "web backend — REST APIs, FastAPI, Express, Django, authentication, server logic",
    "web-frontend": "web frontend — React, Vue, Svelte, components, state management, SPA",
    "docker": "Docker and containers — Dockerfile, docker-compose, images, volumes, networks",
    "devops": "DevOps and CI/CD — GitHub Actions, GitLab CI, Terraform, Ansible, Kubernetes",
    "yaml-json": "YAML and JSON config files — schema, validation, parsing, editing config",
    "llm-ops": "LLM operations — inference servers, vLLM, llama.cpp, quantization, serving",
    "llm-orch": "LLM orchestration — LangChain, LlamaIndex, agents, RAG pipelines, tool use",
    "ml-training": "machine learning training — PyTorch, datasets, optimizers, fine-tuning, GPUs",
    "lua-upy": "Lua and MicroPython — embedded scripting on ESP, Pico, OpenComputers",
    # EUROLLM (4)
    "chat-fr": "informal French conversation, daily-life questions in French, casual chitchat",
    "traduction-tech": "technical translation EN<->FR (or other), preserving terminology",
    "redaction-multilingue": "multilingual writing — drafting documents in multiple languages",
    "localisation-doc": "documentation localisation — i18n, l10n, .po files, technical doc translation",
    # GEMMA (5)
    "general": "general everyday questions — common knowledge, no specific technical domain",
    "quick": "quick short factual queries — define a word, simple lookup, one-liner",
    "summarize": "summarisation — TL;DR of an article, condense a text, executive summary",
    "classification": "classification tasks — categorize text, sentiment, label assignment",
    "tldr": "TL;DR requests — give me the short version, the gist, key takeaways",
    # AILIANCE_MACM1 (2)
    "kicad-dsl": "KiCad DSL — generate kicad_sch / kicad_pcb files from textual spec, atopile-like",
    "kicad-pcb": "KiCad PCB layout — placement, routing, copper pours, design rules, fab output",
}

assert len(DOMAINS) == 47, f"Expected 47 domains, got {len(DOMAINS)}"


META = """Generate EXACTLY 60 realistic user prompts that should be classified into the domain "{domain}".

Domain meaning: {desc}

Mix:
- 50% French, 50% English (vary)
- Lengths from 5 to 200 words (mostly short, some long)
- Tone: formal, informal, technical, urgent, polite — vary
- Forms: question, command, expression of need, partial sentence
- 20% should be ambiguous-but-still-clearly-{domain} edge cases

OUTPUT FORMAT: exactly 60 lines, each line a JSON object:
{{"text": "the user prompt", "label": "{domain}"}}

No preamble, no markdown fences, no commentary. Just 60 JSONL lines. Start now.
"""


def gen_one(domain: str, desc: str, out_dir: Path, n_target: int = 100) -> tuple[str, int]:
    """Generate up to n_target examples for a domain. Retries on parse failure."""
    out_file = out_dir / f"{domain}.jsonl"
    if out_file.exists() and sum(1 for _ in out_file.open()) >= n_target * 0.7:
        n = sum(1 for _ in out_file.open())
        return domain, n  # already done

    collected: list[dict] = []
    attempts = 0
    max_attempts = 3
    while len(collected) < n_target and attempts < max_attempts:
        attempts += 1
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": META.format(domain=domain, desc=desc)}],
            "max_tokens": 2500,
            "temperature": 0.9 if attempts == 1 else 0.95,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        try:
            r = httpx.post(GATEWAY, json=payload, timeout=600)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [{domain}] HTTP error attempt {attempts}: {e}", flush=True)
            continue

        for line in content.split("\n"):
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                # try to strip markdown
                m = re.search(r"\{.*?\}", line)
                if not m:
                    continue
                line = m.group(0)
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = ex.get("text") or ex.get("prompt") or ex.get("instruction")
            if not text or len(text) < 3:
                continue
            collected.append({"text": text.strip(), "label": domain})

        # dedup
        seen = set()
        deduped = []
        for ex in collected:
            if ex["text"] in seen:
                continue
            seen.add(ex["text"])
            deduped.append(ex)
        collected = deduped

    out_file.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in collected) + "\n")
    return domain, len(collected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/router-v7-raw")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-target", type=int, default=100, help="examples per domain (single LLM call delivers ~100)")
    ap.add_argument("--only", default=None, help="comma list to restrict")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(DOMAINS.items())
    if args.only:
        wanted = {d.strip() for d in args.only.split(",")}
        items = [(d, desc) for d, desc in items if d in wanted]

    t0 = time.time()
    results: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, d, desc, out_dir, args.n_target): d for d, desc in items}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                domain, n = fut.result()
            except Exception as e:
                print(f"[FAIL] {d}: {e}", flush=True)
                continue
            results.append((domain, n))
            elapsed = time.time() - t0
            print(f"[{len(results)}/{len(items)}] {domain}: {n} examples (elapsed {elapsed:.0f}s)", flush=True)

    print("\n=== Summary ===")
    total = sum(n for _, n in results)
    print(f"Total examples: {total} across {len(results)} domains")
    for d, n in sorted(results):
        marker = "OK" if n >= 50 else "LOW"
        print(f"  [{marker}] {d}: {n}")


if __name__ == "__main__":
    main()
