# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

EU-KIKI is a 100% EU-sovereign multi-model LLM serving pipeline. It routes requests to 3 European models via a Jina v3 domain classifier, each with LoRA adapters.

## Architecture

Gateway (:9200) dispatches to 3 workers:
- Apertus-70B (:9201) — reasoning, hardware, EU normative (20 LoRA domains)
- Devstral Small 2 (:9202) — code generation (16 LoRA domains)
- EuroLLM-22B (:9203) — multilingual EU (4 LoRA domains)

Router: Jina Embeddings v3 (Berlin) + MLP classifier (40 domains)

## Commands

    # Setup
    uv venv && uv pip install -e ".[dev,router]"

    # Tests
    uv run python -m pytest
    uv run python -m pytest tests/test_xielu.py -v     # single file
    uv run python -m pytest -k "test_name"              # single test

    # Launch all services
    bash scripts/start.sh

    # Train router
    uv run python scripts/build_router_data.py
    uv run python scripts/train_router.py

    # Logs
    tail -f /tmp/eu-kiki/gateway.log
    tail -f /tmp/eu-kiki/apertus.log

## Key Design Decisions

- BF16 for all models (512GB unified memory allows it)
- Multi-process workers (1 model per process, shared memory pool)
- Sigmoid routing (domains overlap, not mutually exclusive)
- LoRA on attention projections only (q/k/v/o_proj)
- xielu activation custom-implemented for Apertus MLX support
