# Router — Mascarade Tower override (2026-05-11)

Status: **shipped** (PR #49, commits 3d02560 + b11cf7f), merged to main
and deployed on `electron-server` at 2026-05-11 10:05 CEST.

## What changed

Two coupled corrections to `src/router/domain_map.py`:

### 1. 10 hardware domains now route to Tower-Ollama Mascarade LoRA

The Jina v3 classifier already labels prompts with high confidence
(top-1 typically >0.99) for the 10 mascarade specialties below. Before
this patch they all routed to `APERTUS_PORT` (Studio Mistral-Medium
128B Q8, ~3 tok/s). Now they route to `MASCARADE_PORT = 8004` —
`autossh` tunnel to Tower `:11434` Ollama exposing 11 fine-tuned
Qwen3 4B Q4_K_M LoRA adapters.

| Domain | Top-1 score (typical) | New worker | LoRA adapter |
|---|---|---|---|
| `kicad` (alias `kicad-pcb`) | 0.996 | Tower :8004 | `mascarade-kicad:latest` |
| `stm32` | 0.999 | Tower :8004 | `mascarade-stm32:latest` |
| `spice` | 0.997 | Tower :8004 | `mascarade-spice:latest` |
| `emc` | — | Tower :8004 | `mascarade-emc:latest` |
| `embedded` | 0.994 | Tower :8004 | `mascarade-embedded:latest` |
| `platformio` | — | Tower :8004 | `mascarade-platformio:latest` |
| `freecad` | — | Tower :8004 | `mascarade-freecad:latest` |
| `dsp` | — | Tower :8004 | `mascarade-dsp:latest` |
| `iot` | — | Tower :8004 | `mascarade-iot:latest` |
| `power` | 0.951 | Tower :8004 | `mascarade-power:latest` |

**Throughput**: ~80 tok/s on Tower Q4 (NVIDIA Quadro P2000 5 GB) vs
~3 tok/s on Studio MLX 128B Q8 → **~20× speedup** for these prompts.

**Quality**: domain-fine-tuned LoRA on Qwen3 4B beats Mistral-Medium
128B generalist on the targeted EDA/embedded tasks (per the
mascarade eval set the LoRA were trained against).

### 2. EuroLLM `:9303` fallback to Gemma `:9304`

The Studio EuroLLM 22B plist refuses bootstrap via SSH (`Domain does
not support specified action` — requires a GUI session). The bare
prompt "Bonjour" classifies to `chat-fr` (score 0.9999) and used to
hit a dead backend → 60s timeout / 502.

The four `EUROLLM_DOMAINS` (`chat-fr`, `traduction-tech`,
`redaction-multilingue`, `localisation-doc`) now temporarily route
to `GEMMA_PORT = 9304` (Tower llama.cpp Gemma 3 4B IT GGUF Q4_K_M).

Gated on the constant `EUROLLM_LIVE = False` in `domain_map.py`.
Flip to `True` once the Studio plist is back up; the test
`test_eurollm_fallback_to_gemma_while_down` will fail loudly and
remind you to invert its assertion.

## Routing logic

```
DOMAIN_TO_WORKER construction (last-write-wins):
  1. APERTUS_DOMAINS  -> APERTUS_PORT  (19 labels)
  2. DEVSTRAL_DOMAINS -> DEVSTRAL_PORT (16 labels)
  3. EUROLLM_DOMAINS  -> EUROLLM_LIVE ? EUROLLM_PORT : GEMMA_PORT
  4. GEMMA_DOMAINS    -> GEMMA_PORT   (5 labels)
  5. QWEN_DOMAINS     -> QWEN_PORT    (1 label)
  6. MASCARADE_DOMAINS -> MASCARADE_PORT  <- overrides (1) for 10 labels
```

Invariant: `MASCARADE_DOMAINS ⊆ APERTUS_DOMAINS` (asserted at module
load). The classifier head still emits 40 logits → 45 routable
`ALL_DOMAINS` (with the 5 GEMMA fallback labels), unchanged.

## Confidence gating (opt-in, not yet wired)

New helper:

```python
def get_worker_for_domain_with_confidence(
    domain: str | None,
    score: float,
    *,
    mascarade_min_score: float = 0.85,
) -> int | None:
```

When `domain ∈ MASCARADE_DOMAINS` AND `score < mascarade_min_score`,
returns `APERTUS_PORT` instead. Protects against false-positive
specialist routing on ambiguous prompts. **`server.py` not yet
updated** — current behaviour is unconditional override regardless
of score. Follow-up to thread `top-1 score` from `router.route()`
result through the forward path.

## Live verification

```console
$ curl -s -X POST http://localhost:9300/v1/route \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Comment router un PCB 4 couches sous KiCad ?"}'
{
  "router_loaded": true,
  "selections": [{"domain":"kicad-pcb","score":0.9956}],
  "chosen_domain": "kicad-pcb",
  "chosen_port": 8004
}

$ curl -s -X POST http://localhost:9300/v1/route \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Bonjour"}' | jq '.chosen_port'
9304
```

E2E latency on direct alias path (bypassing DELIBERATE chain):

| Prompt | Alias | Port | Latency |
|---|---|---|---|
| "Comment router un PCB 4 couches en KiCad ?" | `ailiance-kicad` | 8004 | 5.25s |
| "Bonjour" | `ailiance` | 9304 (auto) | 1.10s |

The bare `ailiance` model field on hardware prompts engages the
DELIBERATE chain orchestrator (now backed by the real
`IactBenchValidator` — submodule `vendored/iact-bench` initialised
2026-05-11). Validation + retry passes through the worker multiple
times, so wall-clock can exceed 60s on long technical prompts. Use
the explicit `ailiance-kicad` alias (or `extra_body.chain_policy =
"direct"`) to bypass the chain.

## Test coverage

| Test | Asserts |
|---|---|
| `test_mascarade_overrides_apertus` | 10 labels route to 8004, residual Apertus labels still 9301 |
| `test_kicad_pcb_alias_routes_to_mascarade` | `kicad-pcb` (router output) canonicalizes + lands on 8004 |
| `test_confidence_gating_falls_back_to_apertus` | High score → mascarade, low score → Apertus, edge cases |
| `test_eurollm_fallback_to_gemma_while_down` | 4 EUROLLM_DOMAINS → 9304, with revert instructions |

Run with: `PATH=$HOME/.local/bin:$PATH uv run --with pytest pytest tests/test_router.py -v`

19/19 PASSED in CI-equivalent local run (test_router + tests/router/ + test_gateway_auto_router).

## Revert path

```python
# In src/router/domain_map.py, comment out the MASCARADE override block:
# for d in MASCARADE_DOMAINS:
#     DOMAIN_TO_WORKER[d] = MASCARADE_PORT

# And flip EUROLLM_LIVE = True once Studio :9303 is back:
EUROLLM_LIVE = True
```

Then `sudo systemctl restart eu-kiki-gateway.service`.

## Follow-ups

- Thread `top-1 score` into `server.py` forward path so
  `get_worker_for_domain_with_confidence()` actually gates routing
  on confidence (today the override is unconditional)
- Add active healthcheck on `:8004` (Tower-Ollama tunnel) so
  `_healthy_ports` reflects tower unreachability and triggers the
  Gemma fallback automatically (today the override blindly trusts
  Tower availability)
- Train `mascarade-misra-c`, `mascarade-autosar-cert`,
  `mascarade-doc-technique-ce` LoRA → extend `MASCARADE_DOMAINS`
- Author an ADR documenting the Mascarade > Apertus priority
  semantics (subset override) for posterity
