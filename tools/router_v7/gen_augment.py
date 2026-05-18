#!/usr/bin/env python3
"""Augment under-represented domains in router v7 corpus.

Serial execution, fast models only (gemma, ministral, gemma4), small batches.
Appends to existing per-domain jsonl files.
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
LOG = OUT_DIR / "_augment.log"

# Fast & reliable models only.
FAST_FR = "ailiance-gemma"        # Gemma-3-4B Tower, 100+ tok/s
GEMMA4 = "ailiance-gemma4"        # macm1
MINISTRAL = "ailiance-ministral"  # macm1

# Per-domain descriptions (subset that needs augmentation).
DOMAIN_DESC = {
    "platformio": "PlatformIO build system: platformio.ini, libraries, boards, frameworks",
    "normes-iec": "IEC standards: IEC 61508 functional safety, IEC 60601 medical, IEC 62443 cybersecurity",
    "math": "Mathematical problems: algebra, calculus, linear algebra, probability, proofs",
    "spice": "SPICE simulation: ngspice, LTspice, analog circuit netlists, transient analysis",
    "doc-technique-ce": "CE technical documentation: dossier technique, EU DoC, harmonised standards",
    "kicad-dsl": "KiCad text/S-expression DSL: parsing .kicad_sch, .kicad_pcb files programmatically",
    "stm32": "STM32 microcontroller: HAL, LL drivers, CubeMX, peripherals, freertos on STM32",
    "llm-orch": "LLM orchestration: LangChain, LangGraph, agents, tool use, RAG pipelines",
    "autosar-cert": "AUTOSAR automotive standard: BSW, RTE, ARXML, certification process, ASIL",
    "emc": "EMC/EMI: pre-compliance, FCC/CISPR, ground planes, shielding, ferrites, filtering",
    "chat-fr": "Conversation francaise informelle, salutations, questions generales",
    "kicad": "KiCad ECAD: schematic and PCB questions, library, footprints, netlists",
    "cpp": "C++ code: STL, templates, embedded C++, memory management",
    "redaction-multilingue": "Write multilingual marketing copy, technical reports in FR/EN/DE/ES",
    "misra-c": "MISRA C/C++ coding standard for safety-critical embedded software",
    "shell": "Shell scripts: bash, zsh, awk, sed, pipelines, sysadmin one-liners",
    "rust": "Rust code: ownership, lifetimes, tokio, async, cargo",
    # missing entirely:
    "summarize": "Requests to summarize text, articles, papers, meeting notes into bullet points",
    "tldr": "TL;DR requests: ultra-short condensed summary, key takeaways in 1-3 sentences",
    "classification": "Text classification tasks: sentiment, topic labelling, intent detection requests",
}

META = """You are generating training data for a domain classifier (NOT answering).

Generate exactly {n} short realistic user prompts for label "{domain}".

Domain means: {desc}

Language: mix English and French (~60/40). Vary length 5-80 words. Vary tone.

OUTPUT: JSONL only, one per line: {{"text":"...","label":"{domain}"}}. No fences, no preamble."""


def parse_jsonl(content: str, domain: str) -> list[dict]:
    out = []
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
        text = (ex.get("text") or ex.get("prompt") or "").strip()
        if not text:
            continue
        words = text.split()
        if len(words) < 3 or len(words) > 400:
            continue
        out.append({"text": text, "label": domain})
    return out


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


async def gen_call(client, domain, model, n):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": META.format(
            n=n, domain=domain, desc=DOMAIN_DESC[domain])}],
        "max_tokens": min(3500, 80 + n * 40),
        "temperature": 0.9,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    try:
        t0 = time.time()
        r = await client.post(GATEWAY, json=payload, timeout=180.0)
        if r.status_code != 200:
            log(f"  {domain}/{model}: HTTP {r.status_code}")
            return []
        content = r.json()["choices"][0]["message"]["content"]
        ex = parse_jsonl(content, domain)
        for e in ex:
            e["_model"] = model
        log(f"  {domain}/{model}: {len(ex)}/{n} in {time.time()-t0:.0f}s")
        return ex
    except Exception as e:
        log(f"  {domain}/{model}: {type(e).__name__}")
        return []


async def main():
    log(f"=== AUGMENT START {time.strftime('%H:%M:%S')}")
    # Target: bring each domain to >=150 examples
    TARGET = 150
    async with httpx.AsyncClient() as client:
        for domain, desc in DOMAIN_DESC.items():
            existing_file = OUT_DIR / f"{domain}.jsonl"
            existing = []
            if existing_file.exists():
                for line in existing_file.read_text().splitlines():
                    line = line.strip()
                    if line:
                        try:
                            existing.append(json.loads(line))
                        except Exception:
                            pass
            current = len(existing)
            need = TARGET - current
            if need <= 0:
                log(f"=== {domain}: already {current}, skip")
                continue
            log(f"=== {domain}: have {current}, need {need}")
            # Try 3 calls: gemma N=50, ministral N=50, gemma4 N=50 — strictly serial
            new_ex = []
            for model in [FAST_FR, MINISTRAL, GEMMA4]:
                if len(new_ex) >= need:
                    break
                n = min(60, need - len(new_ex) + 10)
                ex = await gen_call(client, domain, model, n)
                new_ex.extend(ex)
            # Merge + dedupe
            all_ex = existing + new_ex
            seen = set()
            unique = []
            for e in all_ex:
                t = (e.get("text") or e.get("prompt") or "").strip().lower()
                if not t or t in seen:
                    continue
                seen.add(t)
                # normalize key
                unique.append({"text": e.get("text") or e["prompt"], "label": domain,
                               "_model": e.get("_model", "unknown")})
            existing_file.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in unique))
            log(f"=== {domain}: {current} -> {len(unique)}")
    log("=== AUGMENT DONE")


if __name__ == "__main__":
    asyncio.run(main())
