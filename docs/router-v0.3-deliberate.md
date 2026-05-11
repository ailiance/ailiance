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
Keep `max_retries` low for hot paths and treat `deliberate` as
opt-in (it already is — only requests that set
`extra_body.chain_policy` enter the chain orchestrator). Domains
where no validator exists in iact-bench v0.2.0 (e.g. `python`) are
mapped to `direct` in `configs/chain_policies.yaml`.

## What this does **not** do (v0.3.0 scope)

- **No Mixture (cross-judge live)** — v0.3.1.
- **No Sequential (planner+solver)** — v0.4.
- **No streaming with retries** — deferred.
- **No validator hot-reload** — restart gateway after editing YAML.

See the plan doc for the full roadmap and design rationale
(RecursiveMAS lineage, registry decision, mixture aggregator design).
