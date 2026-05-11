# Router v0.3 — Deliberation chain (preview)

Status: shipped 2026-05-11 as opt-in. Default behaviour is unchanged.
Plan reference: [`docs/plans/2026-05-11-router-agentic-v0.3.md`][plan].

[plan]: ./plans/2026-05-11-router-agentic-v0.3.md

## What it is

The gateway now wraps `/v1/chat/completions` with an optional chain
orchestrator implementing the **Deliberation** pattern from
RecursiveMAS:

```
prompt -> LLM -> validator -> exit_code
                            -> 0       => return ok
                            -> non-0   => reflector(stderr) -> retry
```

Mixture (v0.3.1) and Sequential (v0.4) are scaffolded but degrade
silently to DIRECT in v0.3.0.

## Auto-router (default for `model: "ailiance"`)

When `req.model == "ailiance"` (the bare auto-router alias, no
`MODEL_FORCE_MAP` entry), the gateway:

1. Classifies the prompt → domain.
2. Looks up `chain_policies.yaml` for that domain.
3. If the policy is non-`direct`, **engages the chain orchestrator
   automatically** — no `extra_body` opt-in required.

Forced aliases (`ailiance-mistral`, `ailiance-qwen`, …) bypass the
classifier and stay on the legacy 1-shot proxy unless the caller
sends `extra_body.chain_policy` explicitly. Rule of thumb: naming
a specific worker = "I know what I want, no chain"; using the
generic alias = "router, do the right thing".

Streaming requests with `model: "ailiance"` that auto-classify to a
non-direct domain **silently degrade to direct** so the SSE stream
keeps flowing. Only an explicit `extra_body.chain_policy + stream`
combination returns 400 (the caller asked for the impossible).

The response envelope exposes `ailiance_chain.auto_engaged` (bool)
so observability can distinguish auto-engaged chains from explicit
opt-in chains.

## How to call it

The orchestrator is **opt-in per request** via the OpenAI-compatible
`extra_body` field. Clients that don't pass `extra_body` keep the
legacy 1-shot proxy path verbatim.

```jsonc
POST /v1/chat/completions
{
  "model": "ailiance-mistral",
  "messages": [
    {"role": "user", "content": "Emit a .kicad_pcb for a 4-channel I2C hub"}
  ],
  "extra_body": {
    "chain_policy": "deliberate",   // direct | validate | deliberate
    "max_retries": 2,                // optional; default from policy YAML
    "include_audit": true            // attach per-step trace inline
  }
}
```

### Response shape (deliberate)

```jsonc
{
  "id": "chatcmpl-<chain_id>",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "ailiance_chain": {
    "chain_id": "abc123…",
    "policy": "deliberate",
    "status": "ok | exhausted | direct",
    "domain": "kicad-pcb"
  },
  "audit_trace": [          // only when include_audit=true
    {"kind": "llm",       "attempt": 1, "success": true,  "exit_code": null, "duration_s": 0.42},
    {"kind": "validator", "attempt": 1, "success": false, "exit_code": 1,    "duration_s": 0.18},
    {"kind": "reflector", "attempt": 2, "success": true,  "exit_code": null, "duration_s": 0.39},
    {"kind": "validator", "attempt": 2, "success": true,  "exit_code": 0,    "duration_s": 0.20}
  ]
}
```

### Streaming

`stream=true` combined with a non-direct `chain_policy` returns a
**400** in v0.3.0. We expose only the winning attempt today; richer
multi-attempt streaming is deferred to post-v0.3.0 UX work.

## Where audit trails live

Every chain run writes one NDJSON file per request:

```
audit/
  chains/
    <chain_id>/
      cells.ndjson    # one ChainStep per line
```

Set `AILIANCE_AUDIT_DIR=/var/lib/ailiance/audit` to relocate. The
schema matches `iact-bench cells.ndjson` byte-for-byte so an auditor
can replay bench and gateway runs with the same tooling.

## Domain → policy map

`configs/chain_policies.yaml` is the single source of truth. Add a
new domain:

```yaml
policies:
  my-new-domain: { policy: deliberate, tool: my-validator, max_retries: 1 }
```

Reload requires a gateway restart (cached at orchestrator construction).
Wildcard `_default: { policy: direct }` covers any classifier output
not explicitly listed.

## Reflector prompts

`configs/reflector_prompts.yaml` keys per domain. Each template uses
two placeholders:

- `{stderr}` — head of validator stderr (truncated to 4 KB).
- `{previous_output}` — the LLM's last attempt.

Authoring rule: each template should hint at the *class of fix* (DRC
violation vs. parse error vs. linker), not just dump the stderr.
Falls back to `_default` when a domain has no entry.

## Validators (production)

iact-bench v0.2.0 is vendored at `vendored/iact-bench`, pinned to
SHA `0027a593106215e7143f2cf99402a1908d540921`. The gateway selects
its validator at startup via `AILIANCE_VALIDATOR`:

- `auto` (default) — try `IactBenchValidator()`; on any import or
  registry-load failure, fall back to `StubValidator()` and log a
  warning. The orchestrator still degrades to `direct` for any tool
  not present in the registry, so a missing validator never 500s.
- `iact_bench` — force the real adapter; raises at startup if the
  submodule is not initialised.
- `stub` — local dev / CI without docker.

### Updating the validator pin

```bash
cd vendored/iact-bench
git fetch
git checkout <new-sha>
cd ../..
git add vendored/iact-bench
git commit -m "chore: bump iact-bench to <short-sha>"
```

### Local dev without docker

```bash
export AILIANCE_VALIDATOR=stub
uv run uvicorn src.gateway.server:app
```

### Performance budget

`chain_policy=deliberate` requests now spawn 1-N validator
containers (~1-5 s each, sandboxed `--network=none --read-only`).
Hardware validators are the slowest tier (5-15 s per container)
because they invoke real toolchains: `kicad-drc`, `freecad-script`,
`atopile-build`, `compile-arm-gcc`. Keep `max_retries` low for hot
paths and treat `deliberate` as opt-in (it already is — only
requests that set `extra_body.chain_policy` enter the chain
orchestrator). Domains where no validator exists in iact-bench
v0.2.0 (e.g. `python`) are mapped to `direct` in
`configs/chain_policies.yaml`.

### Production rollout — first client

The first production consumer of `extra_body.chain_policy=deliberate`
is **electron-rare** (Ailiance hardware/PCB consulting).
Their workload concentrates on hardware verification domains, all
covered by iact-bench v0.2.0:

| Client domain      | Validator        | Container budget |
| ------------------ | ---------------- | ---------------- |
| `kicad-pcb`        | `kicad-drc`      | 8-15 s           |
| `kicad-dsl`        | `atopile-build`  | 5-10 s           |
| `spice-sim`        | `ngspice-converge` | 2-5 s          |
| `freecad`          | `freecad-script` | 5-12 s           |
| `cpp`              | `compile-cpp`    | 1-3 s            |
| `embedded`         | `compile-arm-gcc` | 2-4 s           |

Recommended initial config for electron-rare workload:
- Set `max_retries: 1` for `kicad-pcb` (the slowest validator) for
  the first weeks of production use to bound p95 latency.
- Monitor `ailiance_gw_route_seconds` p95 via `/metrics` — flip to
  `max_retries: 2` once the histogram stabilises.
- The Python `pass-rate-execute` gap is acceptable for this client
  cohort (Python is not on their production hot path).

## What this does **not** do (v0.3.0 scope)

- **No Mixture (cross-judge live)** — v0.3.1.
- **No Sequential (planner+solver)** — v0.4.
- **No streaming with retries** — deferred.
- **No validator hot-reload** — restart gateway after editing YAML.

See the plan doc for the full roadmap and design rationale
(RecursiveMAS lineage, registry decision, mixture aggregator design).
