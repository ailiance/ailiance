# 2026-05-19 — Phase 1 RAM demotion (on-demand model loading)

Part of the on-demand model loading plan
(`docs/superpowers/plans/2026-05-19-on-demand-model-loading.md`).

## Action

Stopped 5 low-traffic resident MLX servers on the Mac Studio to free RAM
for the Phase 2 swap server. Studio free RAM: 94 GB -> 197 GB.

| Server | Port (real) | Reclaimed |
|--------|-------------|-----------|
| qwen2.5-7B | :8501 | ~4 GB |
| qwen3-4b | :9341 | ~8 GB |
| mistral-small | :9326 | ~13 GB |
| qwen36 | :8500 (plan said :9305 — wrong port) | ~18 GB |
| eurollm | :9303 | ~43 GB |

## Demoted aliases — pending swap-pool routing (Phase 2)

Until the Phase 2 swap server (:9350) is deployed, these aliases fall
back to Gemma:

- `ailiance-qwen36` — newly demoted (server :8500 stopped).
- `ailiance-mistral-small` — newly demoted (server :9326 stopped).
- EuroLLM auto-router domains — newly demoted (worker :9303 stopped).
  No public `ailiance-eurollm` alias exists; the degradation is on the
  EUROLLM_DOMAINS auto-router path.
- `ailiance-llama`, `ailiance-mixtral`, `ailiance-mixtral-8x22b`,
  `ailiance-qwen-235b`, `ailiance-flagship`, `ailiance-devstral-base` —
  already degraded before Phase 1 (dedicated servers :9324/:9329/:9328/
  :9316 down since the 2026-05-12 reboot).

All 8 aliases are routed to the swap pool by PR #103 (branch
`feat/swap-pool-routing`), which must not be deployed until the Phase 2
swap server is up.
