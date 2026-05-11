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

## Validators not yet wired

The `IactBenchValidator` shim exists (`src/orchestrator/validators.py`)
but the iact-bench submodule is not vendored in this repo. Steps to
wire production validators:

1. Add submodule:
   `git submodule add ../iact-bench vendored/iact-bench`
2. Pin to a known-good iact-bench commit and check in `.gitmodules`.
3. Set `AILIANCE_IACT_BENCH_PATH=vendored/iact-bench/...` if the
   default path doesn't fit your layout.
4. In `make_gateway_app`, swap `app.state.orchestrator_validator =
   StubValidator()` for `IactBenchValidator()`.

Until that's done, the orchestrator silently degrades any
`deliberate` request to `direct` and logs a warning, so a missing
submodule never 500s a caller.

## What this does **not** do (v0.3.0 scope)

- **No Mixture (cross-judge live)** — v0.3.1.
- **No Sequential (planner+solver)** — v0.4.
- **No streaming with retries** — deferred.
- **No validator hot-reload** — restart gateway after editing YAML.

See the plan doc for the full roadmap and design rationale
(RecursiveMAS lineage, registry decision, mixture aggregator design).
