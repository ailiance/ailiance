# ailiance gateway router v0.3 — agentic tool-chain orchestration

**Status**: planning, target after iact-bench v0.2.0 stable run.
**Inspiration**: [RecursiveMAS](https://recursivemas.github.io/) recursive-LLM
patterns (Hypneum Lab framework). We borrow the 4 patterns but execute
them in **text-space** (HTTP / OpenAI-compatible) rather than the
latent-space `RecursiveLink` of the upstream design, because our worker
fleet is heterogeneous (Mistral MLX, Qwen GGUF, Gemma llama-server, …)
and shares no embedding manifold.

## Why upgrade the router

Today (`src/router/classifier.py`):
```
prompt → MiniLM L6 v2 embed → MLP head → domain → 1 worker URL
                                                  → 1 chat completion
```

Limits:
- **No verification**: the model's first answer ships, even if a
  `kicad-cli pcb drc` would reject the `.kicad_pcb` it emitted.
- **No cross-judging on synth or ambiguous domains**: cross-judge logic
  is iact-bench-only; the live gateway can't replicate it.
- **No retry-with-feedback**: a failed compile is a dead-end for the
  caller; the system can't say "try again, here's the linker error".
- **No tool suggestion**: a caller asking "validate this PCB" gets a
  raw LLM answer, not an actual `kicad-cli pcb drc` invocation.

Target:
```
prompt → router → {worker, tools[], chain_policy}
       → orchestrator runs the chain
       → returns final answer + audit trace
```

## 4 RecursiveMAS patterns mapped to gateway

| RecursiveMAS pattern | Gateway implementation | Use case |
|---|---|---|
| **Sequential** (Planner→Critic→Solver) | LLM A plans steps → LLM B critiques plan → LLM C executes | Complex multi-domain prompts ("design + sim + BOM") |
| **Mixture** (parallel specialists + aggregator) | Fan-out to N workers, judge LLM aggregates | High-stakes domains, cross-judge bench |
| **Distillation** (expert→learner) | Not gateway-time; training-time iact-bench feedback loop | Out-of-scope router v0.3 |
| **Deliberation** (tool-caller + reflector) | LLM gen → validator sandbox → if fail, reflector reads stderr → re-prompt | **Primary case**: KiCad/SPICE/MCAD code generation |

**Primary v0.3 target = Deliberation pattern** because it directly
unlocks the audit-grade benchmark signal: a model that fails first try
but succeeds with feedback is qualitatively different from one that
fails twice. iact-bench can grade both.

## Architecture

### Router output (new schema)

```python
@dataclass
class RouteDecision:
    primary_worker: str             # alias, e.g. "ailiance-mistral"
    fallback_workers: list[str]     # ordered by router confidence
    recommended_tools: list[str]    # validator names from iact-bench registry
    chain_policy: ChainPolicy       # see below
    domain: str                     # classifier output, for audit
    confidence: float               # router softmax score
```

### ChainPolicy enum

```python
class ChainPolicy(StrEnum):
    DIRECT = "direct"               # current behaviour, no chain
    VALIDATE = "validate"           # run validator on output; surface result
    DELIBERATE = "deliberate"       # validate; if fail, retry with stderr feedback
    MIXTURE = "mixture"             # fan-out + judge aggregator
    SEQUENTIAL = "sequential"       # multi-step planner+solver
```

### Domain → ChainPolicy map (configs/chain_policies.yaml)

```yaml
policies:
  # Verified by sandbox tool → Deliberate
  kicad-pcb:       { policy: deliberate, tool: kicad-drc,        max_retries: 2 }
  kicad-dsl:       { policy: deliberate, tool: atopile-build,    max_retries: 2 }
  cpp:             { policy: deliberate, tool: compile-cpp,      max_retries: 2 }
  embedded:        { policy: deliberate, tool: compile-arm-gcc,  max_retries: 2 }
  python:          { policy: deliberate, tool: pass-rate-execute, max_retries: 1 }
  spice-sim:       { policy: deliberate, tool: ngspice-converge, max_retries: 2 }
  freecad:         { policy: deliberate, tool: freecad-script,   max_retries: 1 }
  shell:           { policy: deliberate, tool: compile-shell,    max_retries: 1 }
  typescript:      { policy: deliberate, tool: compile-typescript, max_retries: 1 }
  sql:             { policy: validate,   tool: parse-sql }       # no retry, info only
  yaml-json:       { policy: validate,   tool: compile-yaml-json }
  html-css:        { policy: validate,   tool: parse-html-css }

  # No deterministic sandbox tool → Mixture (judge-only)
  chat-fr:         { policy: mixture, workers: [ailiance-mistral, ailiance-gemma4],
                     judge: ailiance-mistral }
  docker-devops:   { policy: direct }
  emc-dsp-power:   { policy: direct }
  iot:             { policy: direct }
  llm-ops:         { policy: direct }
  llm-orch:        { policy: direct }
  ml-training:     { policy: direct }
  music-audio:     { policy: direct }
  security-fenrir: { policy: mixture, workers: [ailiance-mistral, ailiance-qwen],
                     judge: ailiance-mistral }

  # Math: exact-match grading, no need for chain
  math-gsm8k:      { policy: direct }
  math-reasoning:  { policy: direct }

  # Translation: bleu metric, no chain
  multilingual-eu: { policy: direct }
  traduction-tech: { policy: direct }
```

## Deliberation loop (canonical case)

```
1. LLM call:    model emits raw output (e.g. .kicad_pcb S-expr)
2. Tool call:   run validator sandbox (kicad-cli pcb drc)
3. Branch:
   - exit_code == 0 → return output + audit trace.
   - exit_code != 0 AND retries_left > 0:
       Reflector prompt = original_prompt
                       + "\n\nPrevious attempt:\n" + previous_output
                       + "\n\nValidator stderr:\n" + stderr_head
                       + "\n\nFix the issues and try again."
       Loop to step 1 with reflector prompt, decrement retries.
   - exit_code != 0 AND retries_left == 0 → return last output +
       full retry trace + status="exhausted".
```

### Sandbox guarantees (unchanged from iact-bench v0.2)

The validator dispatch reuses the iact-bench `validators/runner.py`
exactly: `--network=none --read-only --tmpfs --user 1000:1000
--cap-drop=ALL --security-opt no-new-privileges`. Pinned image digest
per call. Same `audit/runs/<chain_id>/cells.ndjson` format for
auditability — gateway runs and bench runs share the artefact schema.

## Shared registry with iact-bench

`configs/domain_validators.yaml` becomes the **single source of truth**.
Both iact-bench `bench_runner` and ailiance gateway `chain_orchestrator`
import the same registry. A v0.3.0 validator scaffolded in iact-bench
becomes automatically available to the gateway after digest pin.

Path: copy `iact-bench/configs/domain_validators.yaml` into
`ailiance/configs/` at build time, or git-submodule, or HTTP fetch on
boot. Decision: **git submodule** for explicit pin + auditability.

## API surface (OpenAI-compat preserved)

The orchestrator stays behind `/v1/chat/completions`. Per-request
opt-in via OpenAI `extra_body` field:

```json
{
  "model": "ailiance",
  "messages": [...],
  "extra_body": {
    "chain_policy": "deliberate",
    "max_retries": 2,
    "include_audit": true
  }
}
```

Default policy comes from the domain map; `extra_body.chain_policy`
overrides per request. `include_audit: true` returns the per-step
trace inline as a `tool_calls` array (OpenAI standard).

## Mixture pattern (cross-judge live)

For domains with no deterministic validator, route to N workers in
parallel, then a judge LLM picks the best answer or aggregates:

```
prompt → fan-out → [worker_a, worker_b, worker_c]
                 → judge_llm sees all 3 + original prompt
                 → returns ranked: [winner, alternates[]]
```

Same trade-off as iact-bench cross-judge: when no sandbox tool exists,
LLM-as-judge is the best signal. Live latency cost: ~3× single call,
mitigated by streaming the winner once judge decides.

## Sequential pattern (multi-step)

For prompts like "design a 4-channel battery monitor PCB + boîtier
IP54 + CAM G-code", the router decomposes into a plan:

```
1. PlanLLM emits structured plan:
   { schematic: skidl, layout: kicad, enclosure: cadquery, cam: freecad }
2. Solver loops: for each step, call Deliberation chain on the right
   domain.
3. Aggregator concatenates artefacts into a single response.
```

This is **v0.4 territory** — out of v0.3 scope. Document for roadmap.

## Implementation phases

### v0.3.0 — Deliberation (primary)

- [ ] `RouteDecision` dataclass + `ChainPolicy` enum
- [ ] `configs/chain_policies.yaml` with 31 domain mappings
- [ ] `chain_orchestrator.py` implementing the Deliberation loop
- [ ] Reuse `validators/runner.py` from iact-bench via submodule
- [ ] `extra_body.chain_policy` opt-in API
- [ ] Per-chain audit NDJSON at `audit/chains/<chain_id>/`
- [ ] Unit tests (mocked LLM + mocked validator) + smoke
- [ ] Bench impact: iact-bench v0.3 adds `chain_score` column

### v0.3.1 — Mixture (cross-judge live)

- [ ] Fan-out worker calls (asyncio.gather)
- [ ] Judge prompt template + parsing
- [ ] Streaming the winner once judge picks
- [ ] Latency budget guards (max 30s for non-streaming, abort if exceeded)

### v0.4 — Sequential (multi-step planner)

- [ ] PlanLLM prompt template
- [ ] Plan validation (only known domain keys)
- [ ] Step-by-step orchestrator with explicit dependency graph
- [ ] Aggregator response synthesis
- [ ] Cross-axis mechatronic stack test (SKiDL → KiCad → CadQuery → G-code)

## What this does **not** do (vs. RecursiveMAS)

- **No latent-space transfer**: our workers don't share an embedding
  space, so we stay in text-space. Cost: vocabulary projection per
  hop. Benefit: heterogeneous worker pool, OpenAI-API compat.
- **No backprop through the chain**: gradient-based credit assignment
  needs `RecursiveLink` modules, requires a shared backbone, requires
  training-time orchestration. Out of scope for a serving gateway.
- **No "round-based refinement queues"**: each request is one-shot
  (with retry loop), not a streaming refinement queue. Could be a v0.5
  feature.

## Open questions for brainstorm

1. **Submodule vs. HTTP fetch** for `domain_validators.yaml` sharing?
2. **Reflector prompt template** — exact wording for the retry?
   Variants per domain (KiCad vs. SPICE vs. shell)?
3. **`extra_body` API** for the chain trace — what format gives
   auditors the most readable evidence? `tool_calls[]` is OpenAI
   standard but lossy for stderr blobs.
4. **Mixture aggregator policy** — best-of-N pick, vote, or weighted
   ensemble? Pick simplifies UI; weighted ensemble would need
   workers to return logprobs (Mistral MLX doesn't expose those).
5. **Streaming with retries** — should we stream the winning attempt
   only, or stream each attempt's prefix with a marker? Latter is
   richer for the UI but bloats the OpenAI stream protocol.

## AI Act / audit considerations

The orchestrator's per-chain NDJSON gives an auditor:
- Reproducibility of multi-step decisions (which worker, which tool,
  which retry).
- A trace they can replay step-by-step.
- A way to attribute failures: "the LLM was wrong here, the validator
  caught it, the reflector fixed it on retry 2".

This is **stronger evidence than iact-bench alone** because it
exercises the production path the customer actually uses, not just an
offline grade.

## Roadmap link

- iact-bench v0.2.0 (shipped 2026-05-11) — validators registry
- iact-bench v0.3.0 (planned) — 4-dialect kicad-dsl + 3-engine spice-sim
- iact-bench v0.4.0 (planned) — MCAD trilogy
- **ailiance router v0.3.0** (this doc) — Deliberation in gateway
- **ailiance router v0.3.1** — Mixture
- **ailiance router v0.4.0** — Sequential
- iact-bench v0.5.0 — cross-axis mechatronic stack bench, exercises
  the Sequential pattern end-to-end
