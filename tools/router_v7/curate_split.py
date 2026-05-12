#!/usr/bin/env python3
"""Curate raw v7 corpus: dedupe, length filter, edge cases, stratified split."""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("/home/electron/ailiance/data/router-v7-multimodel-raw")
OUT = Path("/home/electron/ailiance/data/router-v7-multimodel")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Hand-crafted edge cases for the most ambiguous / critical labels.
EDGE_CASES = {
    "reasoning": [
        "Think step by step about why the sky is blue.",
        "Raisonne logiquement: si A>B et B>C, que peut-on dire de A et C ?",
        "Solve this puzzle step by step: 3 cannibals 3 missionaries crossing a river.",
        "Décompose le raisonnement étape par étape pour résoudre cette énigme.",
        "Walk me through the logical deduction.",
        "Explique le pourquoi en plusieurs étapes.",
    ],
    "python": [
        "Write a Python fibonacci function.",
        "Ecris une fonction Python pour reverse une string.",
        "Pandas dataframe filter rows where col > 5",
        "fastapi endpoint POST with pydantic model",
        "asyncio gather example python",
        "Python: comment lire un CSV avec pandas ?",
    ],
    "chat-fr": [
        "Bonjour, comment ça va aujourd'hui ?",
        "Salut, tu peux m'aider ?",
        "Merci beaucoup pour ton aide !",
        "Comment tu t'appelles ?",
        "À quoi sers-tu exactement ?",
        "Quel temps fait-il à Paris ?",
    ],
    "quick": [
        "What is the capital of France?",
        "Date du jour ?",
        "Define entropy",
        "Qu'est-ce que TCP ?",
        "Currency of Japan",
    ],
    "tldr": [
        "TL;DR this article",
        "Give me the TLDR",
        "Résume en une phrase",
        "tl;dr please",
    ],
    "kicad": [
        "How do I add a new footprint in KiCad?",
        "Comment créer une netlist dans KiCad ?",
        "KiCad ERC error: pin not connected",
    ],
    "spice": [
        "ngspice transient analysis example",
        "How do I model a MOSFET in SPICE?",
        "Netlist SPICE pour amplificateur classe A",
    ],
    "stm32": [
        "STM32 HAL UART receive interrupt",
        "Configure ADC on STM32F4 with DMA",
        "CubeMX clock tree for STM32H7",
    ],
    "emc": [
        "How to reduce common-mode emissions on a switching converter",
        "FCC Part 15 radiated emissions limits",
        "Pre-compliance EMC test plan for medical device",
    ],
    "general": [
        "What should I cook for dinner?",
        "Quel film me conseilles-tu ?",
        "Tell me a joke",
        "Donne-moi une idée de cadeau",
    ],
    "traduction-tech": [
        "Translate 'low-dropout regulator' to French",
        "Comment dit-on 'thermal pad' en français ?",
        "Translate this datasheet excerpt to English",
    ],
}


def main():
    all_by_label: dict[str, list[dict]] = defaultdict(list)
    model_counter: Counter = Counter()
    raw_total = 0

    for f in sorted(RAW.glob("*.jsonl")):
        label = f.stem
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            words = len(text.split())
            if words < 3 or words > 400:
                continue
            # Drop prompts that contain JSONL meta-instructions (model leaked the template back)
            if re.search(r"\b(JSONL|JSON Lines|\"label\"|generate.{0,20}prompts)\b", text, re.I):
                continue
            all_by_label[label].append({
                "prompt": text,
                "domain": label,
                "_model": ex.get("_model", "unknown"),
            })
            model_counter[ex.get("_model", "unknown")] += 1
            raw_total += 1

    print(f"Raw kept: {raw_total} across {len(all_by_label)} labels")

    # inject hand edge cases
    for label, prompts in EDGE_CASES.items():
        for p in prompts:
            all_by_label[label].append({"prompt": p, "domain": label, "_model": "manual"})

    # Dedupe per-label (case-insensitive)
    for label, lst in all_by_label.items():
        seen = set()
        unique = []
        for ex in lst:
            k = ex["prompt"].lower().strip()
            if k in seen:
                continue
            seen.add(k)
            unique.append(ex)
        all_by_label[label] = unique

    # Stratified split 80/10/10
    train, val, test = [], [], []
    per_label_stats = {}
    for label, lst in sorted(all_by_label.items()):
        random.shuffle(lst)
        n = len(lst)
        n_test = max(5, n // 10)
        n_val = max(5, n // 10)
        n_train = n - n_test - n_val
        test.extend(lst[:n_test])
        val.extend(lst[n_test:n_test + n_val])
        train.extend(lst[n_test + n_val:])
        per_label_stats[label] = {"total": n, "train": n_train, "val": n_val, "test": n_test}

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    def write(name: str, data: list[dict]):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for ex in data:
                f.write(json.dumps({"prompt": ex["prompt"], "domain": ex["domain"]}, ensure_ascii=False) + "\n")
        # also write the file train_router.py expects (`valid.jsonl`)
        if name == "val":
            with open(OUT / "valid.jsonl", "w") as f:
                for ex in data:
                    f.write(json.dumps({"prompt": ex["prompt"], "domain": ex["domain"]}, ensure_ascii=False) + "\n")

    write("train", train)
    write("val", val)
    write("test", test)

    summary = {
        "total": raw_total + sum(len(v) for v in EDGE_CASES.values()),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "num_labels": len(all_by_label),
        "per_label": per_label_stats,
        "model_contribution": dict(model_counter.most_common()),
    }
    (OUT / "stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"train={len(train)} val={len(val)} test={len(test)} labels={len(all_by_label)}")
    print("Model contribution top:")
    for m, c in model_counter.most_common(15):
        print(f"  {m}: {c}")
    print(f"Wrote {OUT}/")


if __name__ == "__main__":
    main()
