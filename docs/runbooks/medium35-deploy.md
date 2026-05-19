# Runbook — deploying the medium35 training feature

Deploys the gateway-orchestrated training feature (PR #106) to the production
gateway on **electron-server** (`ailiance-gateway.service`, FastAPI :9300).

## 0. Branch state — no reconciliation needed

The production checkout `/home/electron/ailiance` sits on the branch
`hotfix/rollback-mascarade-9340-to-8004`. Verified 2026-05-19: that branch is
**0 commits ahead of `origin/main`** — the rollback is already merged into
`main`, and the hotfix branch name is now just a stale label on the same
content. So no reconciliation is required: merge PR #106 into `main`, then
deploy `main` (step 3). The prod checkout also has some untracked files
(`audit/`, local `scripts/router_v7_*.py`) — harmless, a `git checkout main`
leaves them in place.

## 1. Choose the admin token

The training admin endpoints require a shared secret. Generate one:

    openssl rand -hex 32

Keep it — it is the `X-Admin-Token` header value for every
`/admin/training/*` call. Without it set, those endpoints return 503.

## 2. systemd drop-in for the new env vars

Following the existing `fifo.conf` drop-in pattern, create
`/etc/systemd/system/ailiance-gateway.service.d/training.conf` (needs sudo):

    sudo install -d /etc/systemd/system/ailiance-gateway.service.d
    sudo tee /etc/systemd/system/ailiance-gateway.service.d/training.conf >/dev/null <<'EOF'
    [Service]
    Environment="AILIANCE_ADMIN_TOKEN=<PASTE-THE-TOKEN-FROM-STEP-1>"
    Environment="AILIANCE_CAMPAIGN_STATE=/home/electron/ailiance/campaign_state.json"
    EOF

`AILIANCE_STUDIO_SSH` defaults to `clems@100.116.92.12` and electron-server
already reaches Studio directly — no override needed unless that changes.

## 3. Pull the code

    ssh electron-server
    cd /home/electron/ailiance
    git fetch origin
    git checkout main          # off the stale hotfix label (see step 0)
    git pull --ff-only origin main

The medium35 feature lands under `src/gateway/training/` and `scripts/studio/`.
No new Python dependencies (stdlib + existing FastAPI/httpx).

## 4. Restart the service

    sudo systemctl daemon-reload
    sudo systemctl restart ailiance-gateway.service
    systemctl status ailiance-gateway.service --no-pager

## 5. Verify

    # gateway up
    curl -sf http://localhost:9300/health

    # admin endpoint reachable and token-gated
    curl -s -o /dev/null -w '%{http_code}\n' http://localhost:9300/admin/training/status        # expect 401
    curl -s -H "X-Admin-Token: <TOKEN>" http://localhost:9300/admin/training/status              # expect {"status":"IDLE",...}

A 503 from the second-to-last call means `AILIANCE_ADMIN_TOKEN` is not set —
re-check the drop-in and `daemon-reload`.

## 6. Rollback

If the gateway misbehaves after deploy:

    cd /home/electron/ailiance
    git checkout hotfix/rollback-mascarade-9340-to-8004
    sudo systemctl restart ailiance-gateway.service

The `training.conf` drop-in is inert when the training code is absent, but it
can be removed too: `sudo rm /etc/systemd/system/ailiance-gateway.service.d/training.conf`.

## Next

Once deployed and verified, the campaign is launched per
`medium35-campaign.md` (pre-flight, then `POST /admin/training/start`).
