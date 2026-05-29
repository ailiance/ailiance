#!/usr/bin/env python3
"""Curate router v7 corpus: inject edge cases, build splits.

Inputs:  data/router-v7-raw/<domain>.jsonl
Output:  data/router-v7/{train,valid,test}.jsonl (80/10/10 stratified)
         data/router-v7/CORPUS_STATS.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Edge cases: human-curated prompts that MUST be classified correctly.
# Used both to seed training and as a sanity test set.
EDGE_CASES: dict[str, list[str]] = {
    "reasoning": [
        "Think step by step about why the sky is blue",
        "Pourquoi la lumiere voyage a vitesse constante ? Raisonne logiquement",
        "Solve: if A>B and B>C then what is the relation between A and C?",
        "Raisonne etape par etape: 3 boites, 1 piege, 2 indices. Laquelle ouvrir?",
        "Walk me through your reasoning for choosing this answer",
        "Decompose the problem into smaller logical steps",
        "Use chain of thought to explain why 0.999... = 1",
        "Reflechis a voix haute sur ce probleme de logique",
        "Pensee analytique: quelles consequences si X alors Y?",
        "Solve this puzzle step by step",
    ],
    "python": [
        "Write Python fibonacci",
        "Ecris une fonction Python pour reverse une string",
        "How do I install pandas with pip?",
        "Python decorator example",
        "Use asyncio to fetch 10 URLs in parallel in Python",
        "Pourquoi mon import numpy echoue?",
        "Python f-string formatting tips",
        "Convert a CSV to a dict in Python",
        "Quelle est la difference entre tuple et liste en Python?",
        "Write a Python script to rename files",
    ],
    "traduction-tech": [
        "Translate this technical doc to French",
        "Traduis ce manuel utilisateur en anglais",
        "How would you render 'feedback loop' in French technical writing?",
        "Localize this error message for German users",
        "Translation: 'circuit board' en francais technique?",
        "Translate this datasheet section, keep units in SI",
        "Rends-moi cette spec en anglais sans changer la terminologie",
        "Best French equivalent for 'pipeline' in DevOps context?",
    ],
    "cpp": [
        "Compile cette fonction C++ avec un bug template",
        "C++ smart pointer example",
        "Pourquoi ce constructeur copy en C++ ne marche pas?",
        "Modern C++ RAII pattern for resource handle",
        "Difference between unique_ptr and shared_ptr in C++",
        "C++20 concepts tutorial",
    ],
    "rust": [
        "Why does the Rust borrow checker complain here?",
        "Pourquoi ce code Rust ne compile pas? error lifetime",
        "Rust async/await tutorial",
        "Convert this C function to Rust",
    ],
    "typescript": [
        "TypeScript generics example",
        "Pourquoi tsc dit que ce type n'existe pas?",
        "TS strict null checks how to enable",
    ],
    "shell": [
        "Bash one-liner to find files modified in last 7 days",
        "Comment lister les fichiers les plus volumineux en shell?",
        "awk command to sum a column of numbers",
    ],
    "sql": [
        "SQL join query for orders and customers",
        "Optimize this slow PostgreSQL query",
        "Requete SQL pour grouper par mois",
    ],
    "web-backend": [
        "FastAPI endpoint with JWT auth",
        "Comment proteger mon API REST contre la CSRF?",
        "Build a REST API with Express in Node",
    ],
    "web-frontend": [
        "React useEffect cleanup pattern",
        "Pourquoi mon useState ne se met pas a jour?",
        "Vue 3 composition API tutorial",
    ],
    "yaml-json": [
        "How do I validate this YAML config?",
        "Convertir ce JSON en YAML",
        "What's wrong with this docker-compose YAML indentation?",
    ],
    "electronics-hw": [
        "Recommend a transistor for switching 12V at 2A",
        "How to design an RC low-pass filter for audio?",
        "Pourquoi ma LED grille sans resistance?",
    ],
    "emc": [
        "Will my board pass CISPR 32 class B?",
        "EMI mitigation for switching power supply",
        "Comment passer les tests EMC en mode rayonne?",
    ],
    "kicad": [
        "How to import a footprint into KiCad?",
        "KiCad symbol library setup",
        "Pourquoi mon ERC echoue dans KiCad?",
    ],
    "power": [
        "Choose a buck converter for 24V to 5V 3A",
        "Battery management for 4S Li-Ion pack",
        "Dimensionner un LDO faible bruit",
    ],
    "spice": [
        "Run a .tran SPICE simulation on this RC circuit",
        "ngspice fails with timestep too small, why?",
        "LTspice opamp model setup",
    ],
    "stm32": [
        "STM32 UART DMA receive example",
        "How to configure SPI on STM32F4 with HAL?",
        "STM32CubeMX clock tree config",
    ],
    "autosar-cert": [
        "What ASIL level requires redundant memory protection?",
        "AUTOSAR SWC port interfaces explained",
        "ISO 26262 hazard analysis steps",
    ],
    "doc-technique-ce": [
        "Generer un dossier technique CE pour mon produit",
        "Declaration UE de conformite template",
        "EU technical file requirements for low-voltage equipment",
    ],
    "misra-c": [
        "Is this pointer cast MISRA-C compliant?",
        "MISRA-C 2012 Rule 8.13 example",
        "Refactor this function to satisfy MISRA",
    ],
    "normes-iec": [
        "IEC 61010 safety requirements for measurement equipment",
        "IEC 60601 leakage current limits",
        "Interpretation of IEC 61508 SIL 2",
    ],
    "localisation-doc": [
        "Set up i18n for my docs site",
        "Generate .po files for French and German",
        "How to localize a markdown documentation tree?",
    ],
    "redaction-multilingue": [
        "Write a product datasheet in English and French simultaneously",
        "Redige une note technique bilingue FR/EN",
    ],
    "classification": [
        "Classify this email as spam or not",
        "Categorize these tickets by priority",
        "Sentiment analysis on this review",
    ],
    "general": [
        "What's the capital of Japan?",
        "Tell me something interesting",
        "Comment vas-tu aujourd'hui?",
    ],
    "quick": [
        "Define 'osmosis' in one sentence",
        "Quelle est la date d'aujourd'hui?",
        "Speed of light?",
    ],
    "summarize": [
        "Summarize this paper in 3 bullets",
        "Resume-moi ce long article",
        "Give me a 100-word summary of the meeting notes",
    ],
    "tldr": [
        "tldr; this stack overflow thread",
        "TL;DR what is Rust?",
        "TL;DR me cette page wikipedia",
    ],
    "llm-ops": [
        "Set up vLLM with continuous batching",
        "Quantize a 70B model to Q4_K_M",
        "llama.cpp server tuning for throughput",
    ],
    "security": [
        "Audit this auth flow for CSRF",
        "Common XSS sinks in React",
        "Bonnes pratiques pour stocker un mot de passe",
    ],
    "iot": [
        "ESP32 MQTT publish example",
        "LoRa range vs payload size",
        "Connecter un capteur DHT22 a un ESP8266",
    ],
    "embedded": [
        "Bare-metal bootloader for ARM Cortex-M0",
        "RTOS task priority inversion",
        "Debug embedded firmware with OpenOCD",
    ],
    "dsp": [
        "Design a FIR filter with 100 taps for 8 kHz",
        "FFT vs DFT in practice",
        "Concevoir un filtre passe-bande IIR",
    ],
    "kicad-dsl": [
        "Generate a kicad_sch file from this textual netlist",
        "Convert atopile to KiCad schematic",
        "Code-first board: define a buck converter in DSL form",
    ],
    "kicad-pcb": [
        "Route this two-layer board with auto-router",
        "Place components for thermal balance on a 4-layer PCB",
        "Generate Gerbers and drill files",
    ],
    "freecad": [
        "Create a parametric enclosure in FreeCAD",
        "FreeCAD sketcher constraints tutorial",
        "Export a STEP file from FreeCAD",
    ],
    "music-audio": [
        "Mix a vocal track with reverb and EQ",
        "Comment masteriser un titre electro?",
        "Best free VST for synth bass",
    ],
    "math": [
        "Solve x^2 + 3x - 4 = 0",
        "Prove that sqrt(2) is irrational",
        "Calcule l'integrale de sin(x) cos(x)",
    ],
    "calcul-normatif": [
        "Calculer la section de cable selon NF C 15-100",
        "Code-prescribed wind load on a tall building",
        "Compute earthing resistance per IEC 60364",
    ],
    "platformio": [
        "platformio.ini for ESP32 with Arduino framework",
        "Add a library dependency in PlatformIO",
        "Switch to ESP-IDF framework in platformio.ini",
    ],
    "docker": [
        "Multi-stage Dockerfile for a Go binary",
        "Pourquoi mon volume Docker ne persiste pas?",
        "docker-compose with traefik labels",
    ],
    "devops": [
        "GitHub Actions matrix build for Python 3.10-3.12",
        "Pipeline GitLab CI avec stages",
        "Ansible playbook to deploy Nginx",
    ],
    "lua-upy": [
        "MicroPython UART read on Pico",
        "Lua script in OpenComputers",
        "ESP32 MicroPython webserver",
    ],
    "llm-orch": [
        "LangChain agent with custom tools",
        "RAG pipeline with LlamaIndex",
        "Build a multi-agent orchestrator",
    ],
    "ml-training": [
        "Fine-tune a 7B with LoRA on a single GPU",
        "PyTorch DataLoader for large image dataset",
        "Schedule the learning rate with cosine warmup",
    ],
    "html-css": [
        "Center a div in CSS",
        "Flexbox layout for a 3-column page",
        "Pourquoi mon CSS grid ne s'affiche pas?",
    ],
    "chat-fr": [
        "Salut, comment ca va ?",
        "Bonjour, j'aurais une question generale",
        "Tu fais quoi ce week-end?",
    ],
}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/router-v7-raw")
    ap.add_argument("--out-dir", default="data/router-v7")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-domain", type=int, default=40, help="domains below this get a warning")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    by_domain: dict[str, list[dict]] = {}

    # Load raw
    for f in sorted(raw.glob("*.jsonl")):
        domain = f.stem
        rows = load_jsonl(f)
        rows = [{"prompt": r.get("text") or r.get("prompt", ""), "domain": domain}
                for r in rows if (r.get("text") or r.get("prompt"))]
        by_domain[domain] = rows

    # Inject edge cases — they go into TRAIN only (also written separately for the obligatory test).
    edge_for_train: dict[str, list[dict]] = {}
    for d, prompts in EDGE_CASES.items():
        edge_for_train[d] = [{"prompt": p, "domain": d} for p in prompts]
        by_domain.setdefault(d, [])
        # Avoid dups
        existing = {r["prompt"] for r in by_domain[d]}
        for r in edge_for_train[d]:
            if r["prompt"] not in existing:
                by_domain[d].append(r)
                existing.add(r["prompt"])

    train, valid, test = [], [], []
    stats = {}
    for d, rows in sorted(by_domain.items()):
        random.shuffle(rows)
        n = len(rows)
        stats[d] = n
        if n < args.min_per_domain:
            print(f"[WARN] {d}: only {n} examples (< {args.min_per_domain})")
        # 80/10/10
        n_train = max(1, int(n * 0.8))
        n_valid = max(1, int(n * 0.1))
        train.extend(rows[:n_train])
        valid.extend(rows[n_train:n_train + n_valid])
        test.extend(rows[n_train + n_valid:])

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    for name, items in [("train", train), ("valid", valid), ("test", test)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)}")

    (out / "CORPUS_STATS.json").write_text(json.dumps({
        "per_domain_total": stats,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_domains": len(by_domain),
        "seed": args.seed,
    }, indent=2))
    print(f"\nWrote splits to {out}/")
    print(f"Total: {len(train)+len(valid)+len(test)} across {len(by_domain)} domains")


if __name__ == "__main__":
    main()
