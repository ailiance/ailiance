#!/usr/bin/env python3
"""Build HF datasets for niche domains missing from the main pipeline.

Covers: docker, web-backend, llm-ops, embedded, kicad, music-audio,
        traduction-tech, misra-c (via CertiCoder configs)

Usage:
    cd ~/ailiance && uv run python scripts/build_hf_niche_datasets.py
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

SEED = 42
MAX = 3000
VALID_RATIO = 0.05
OUT = Path("data/hf-traced")
MANIFEST_PATH = OUT / "MANIFEST_niche.json"
manifest_entries = []


def make_msg(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]}


def save(domain: str, records: list[dict]) -> tuple[int, int]:
    rng = random.Random(SEED)
    rng.shuffle(records)
    n_val = max(1, round(len(records) * VALID_RATIO))
    train, valid = records[n_val:], records[:n_val]
    d = OUT / domain
    d.mkdir(parents=True, exist_ok=True)
    for name, data in [("train.jsonl", train), ("valid.jsonl", valid)]:
        with open(d / name, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → {domain}: {len(train)} train / {len(valid)} valid")
    return len(train), len(valid)


def cap(records, n=MAX):
    if len(records) <= n:
        return records
    return random.Random(SEED).sample(records, n)


def record(domain, hf_id, license_, n_src, n_used, n_train, n_valid, notes=""):
    manifest_entries.append({
        "domain": domain, "hf_id": hf_id, "license": license_,
        "n_source": n_src, "n_used": n_used, "n_train": n_train, "n_valid": n_valid,
        "access_date": datetime.now(timezone.utc).isoformat(), "notes": notes,
    })


# ─────────────────────────────────────────────────────────────
# 1. Docker/DevOps — from StackOverflow Kubernetes Q&A
# ─────────────────────────────────────────────────────────────
def build_docker_devops():
    HF = "mcipriano/stackoverflow-kubernetes-questions"
    print(f"\n[docker-devops] Loading {HF}...")
    try:
        ds = load_dataset(HF, split="train")
        records = []
        for row in ds:
            q = str(row.get("title", "") or row.get("question", ""))
            a = str(row.get("body", "") or row.get("answer", ""))
            if q and a and len(a) > 50:
                records.append(make_msg(q, a[:3000]))
        records = cap(records)
        n_t, n_v = save("docker-devops", records)
        record("docker-devops", HF, "CC-BY-SA-4.0", len(ds), len(records), n_t, n_v,
               "StackOverflow Kubernetes Q&A")
    except Exception as e:
        print(f"  SKIP: {e}")


# ─────────────────────────────────────────────────────────────
# 2. LLM-ops / MLOps — from ZenML LLMOps database
# ─────────────────────────────────────────────────────────────
def build_llmops():
    HF = "zenml/llmops-database"
    print(f"\n[llm-ops] Loading {HF}...")
    try:
        ds = load_dataset(HF, split="train")
        records = []
        for row in ds:
            # Try various column patterns
            for q_col in ["question", "prompt", "instruction", "input"]:
                if row.get(q_col):
                    q = str(row[q_col])
                    break
            else:
                q = str(row.get(list(row.keys())[0], ""))
            for a_col in ["answer", "response", "output", "completion"]:
                if row.get(a_col):
                    a = str(row[a_col])
                    break
            else:
                a = str(row.get(list(row.keys())[-1], ""))
            if q and a and len(a) > 30:
                records.append(make_msg(q, a[:3000]))
        records = cap(records)
        n_t, n_v = save("llm-ops", records)
        record("llm-ops", HF, "Open", len(ds), len(records), n_t, n_v,
               "ZenML LLMOps real-world implementations")
    except Exception as e:
        print(f"  SKIP: {e}")


# ─────────────────────────────────────────────────────────────
# 3. MLOps infrastructure
# ─────────────────────────────────────────────────────────────
def build_mlops_infra():
    HF = "AYI-NEDJIMI/mlops-infrastructure-en"
    print(f"\n[ml-training] Loading {HF}...")
    try:
        ds = load_dataset(HF, split="train")
        records = []
        for row in ds:
            q = str(row.get("instruction", "") or row.get("question", "") or row.get("prompt", ""))
            a = str(row.get("output", "") or row.get("response", "") or row.get("answer", ""))
            if not q:
                q = str(row.get(list(row.keys())[0], ""))
            if not a:
                a = str(row.get(list(row.keys())[-1], ""))
            if q and a and len(a) > 30:
                records.append(make_msg(q, a[:3000]))
        records = cap(records)
        n_t, n_v = save("ml-training", records)
        record("ml-training", HF, "Open", len(ds), len(records), n_t, n_v,
               "MLOps infrastructure instruction pairs")
    except Exception as e:
        print(f"  SKIP: {e}")


# ─────────────────────────────────────────────────────────────
# 4. KiCad / Electronics schematics
# ─────────────────────────────────────────────────────────────
def build_kicad():
    HF = "bshada/open-schematics"
    print(f"\n[kicad] Loading {HF}...")
    try:
        ds = load_dataset(HF, split="train")
        records = []
        for row in ds:
            name = str(row.get("name", "") or row.get("project", ""))
            desc = str(row.get("description", "") or row.get("text", ""))
            components = str(row.get("components", ""))
            if name and (desc or components):
                content = desc
                if components:
                    content += f"\n\nComponents: {components}"
                records.append(make_msg(
                    f"Describe the electronic schematic: {name}",
                    content[:3000],
                ))
        records = cap(records)
        n_t, n_v = save("kicad", records)
        record("kicad", HF, "CC-BY-4.0", len(ds), len(records), n_t, n_v,
               "Open hardware schematics with component lists")
    except Exception as e:
        print(f"  SKIP: {e}")


# ─────────────────────────────────────────────────────────────
# 5. Music/Audio instruction
# ─────────────────────────────────────────────────────────────
def build_music_audio():
    HF = "m-a-p/Music-Instruct"
    print(f"\n[music-audio] Loading {HF}...")
    try:
        ds = load_dataset(HF, split="train")
        records = []
        for row in ds:
            q = str(row.get("instruction", "") or row.get("question", "") or row.get("input", ""))
            a = str(row.get("output", "") or row.get("response", "") or row.get("answer", ""))
            if q and a and len(a) > 30:
                records.append(make_msg(q, a[:3000]))
        records = cap(records)
        n_t, n_v = save("music-audio", records)
        record("music-audio", HF, "Open", len(ds), len(records), n_t, n_v,
               "Music instruction/understanding pairs")
    except Exception as e:
        print(f"  SKIP: {e}")


# ─────────────────────────────────────────────────────────────
# 6. Translation EU — Europarl FR-EN
# ─────────────────────────────────────────────────────────────
def build_traduction_tech():
    HF = "FrancophonIA/europarl-v7_fr-en"
    print(f"\n[traduction-tech] Loading {HF}...")
    try:
        ds = load_dataset(HF, split="train", streaming=True)
        records = []
        for row in ds:
            fr = str(row.get("fr", "") or row.get("translation", {}).get("fr", ""))
            en = str(row.get("en", "") or row.get("translation", {}).get("en", ""))
            if not fr or not en:
                # Try nested structure
                if "translation" in row:
                    t = row["translation"]
                    fr = str(t.get("fr", ""))
                    en = str(t.get("en", ""))
            if fr and en and len(fr) > 20 and len(en) > 20:
                records.append(make_msg(
                    f"Translate the following French text to English:\n{fr}",
                    en,
                ))
            if len(records) >= MAX * 2:
                break
        records = cap(records)
        n_t, n_v = save("traduction-tech", records)
        record("traduction-tech", HF, "Open", -1, len(records), n_t, n_v,
               "Europarl FR-EN parallel corpus")
    except Exception as e:
        print(f"  SKIP: {e}")


# ─────────────────────────────────────────────────────────────
# 7. Embedded — from Arduino docs + OSHWA scraped data
# ─────────────────────────────────────────────────────────────
def build_embedded():
    print(f"\n[embedded] Merging scraped OSHWA + Arduino docs...")
    records = []

    # Load OSHWA data if available
    oshwa_path = Path("data/scraped/oshwa/train.jsonl")
    if oshwa_path.exists():
        with open(oshwa_path) as f:
            for line in f:
                r = json.loads(line)
                # Strip provenance for training format
                records.append({"messages": r["messages"]})
        print(f"  OSHWA: {len(records)} projects")

    # Load Arduino docs
    HF = "gavmac00/arduino-docs"
    try:
        ds = load_dataset(HF, split="train")
        for row in ds:
            q = str(row.get("instruction", "") or row.get("question", "") or row.get("input", ""))
            a = str(row.get("output", "") or row.get("response", "") or row.get("text", ""))
            if q and a and len(a) > 30:
                records.append(make_msg(q, a[:3000]))
        print(f"  Arduino docs: {len(records)} total after merge")
    except Exception as e:
        print(f"  Arduino docs skip: {e}")

    records = cap(records)
    n_t, n_v = save("embedded", records)
    record("embedded", "OSHWA+gavmac00/arduino-docs", "Open+CC-BY-SA", -1,
           len(records), n_t, n_v, "Merged OSHWA certified HW + Arduino docs")


# ─────────────────────────────────────────────────────────────
# 8. EMC/DSP/Power — from scraped arXiv + Wikipedia
# ─────────────────────────────────────────────────────────────
def build_emc_dsp():
    print(f"\n[emc-dsp-power] Merging scraped arXiv eess + Wikipedia electronics...")
    records = []

    for src in ["data/scraped/arxiv-eess/train.jsonl", "data/scraped/wikipedia-electronics/train.jsonl"]:
        p = Path(src)
        if p.exists():
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    records.append({"messages": r["messages"]})
            print(f"  {p.stem}: {len(records)} total")

    records = cap(records)
    n_t, n_v = save("emc-dsp-power", records)
    record("emc-dsp-power", "arXiv-eess+Wikipedia", "arXiv+CC-BY-SA", -1,
           len(records), n_t, n_v, "Merged arXiv eess papers + Wikipedia electronics")


# ─────────────────────────────────────────────────────────────
# 9. CertiCoder MISRA — try all configs
# ─────────────────────────────────────────────────────────────
def build_misra():
    HF = "wuog/CertiCoder"
    print(f"\n[misra-c] Loading {HF}...")
    for config in ["rule_tuning", "cold_start", "preference", "default", None]:
        try:
            if config:
                ds = load_dataset(HF, config, split="train")
            else:
                ds = load_dataset(HF, split="train")
            print(f"  Config '{config}': {len(ds)} rows, cols={ds.column_names}")
            records = []
            for row in ds:
                # Try all possible column combos
                q = str(row.get("instruction", "") or row.get("prompt", "") or row.get("input", ""))
                a = str(row.get("output", "") or row.get("response", "") or row.get("chosen", ""))
                if q and a and len(a) > 20:
                    records.append(make_msg(q, a[:3000]))
            if records:
                records = cap(records)
                n_t, n_v = save("misra-c", records)
                record("misra-c", HF, "Research", len(ds), len(records), n_t, n_v,
                       f"CertiCoder config={config}, MISRA C:2012 aware")
                return
        except Exception as e:
            print(f"  Config '{config}': {e}")
            continue
    print("  SKIP: no loadable config found")


def main():
    print("=" * 60)
    print("ailiance — Niche domain HF dataset builder")
    print("=" * 60)

    build_docker_devops()
    build_llmops()
    build_mlops_infra()
    build_kicad()
    build_music_audio()
    build_traduction_tech()
    build_embedded()
    build_emc_dsp()
    build_misra()

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_entries, f, indent=2, ensure_ascii=False)
    print(f"\nManifest: {MANIFEST_PATH}")

    print("\n" + "=" * 60)
    print("Summary:")
    for e in manifest_entries:
        print(f"  {e['domain']:<20s} {e['n_used']:>5d} used  ({e['hf_id']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
