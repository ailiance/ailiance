# Runbook — deploying the medium35 training feature

Deploys the gateway-orchestrated training feature (PR #106) to the production
gateway on **electron-server** (`ailiance-gateway.service`, FastAPI :9300).

## 0. Prerequisite — branch reconciliation (DECISION REQUIRED)

The production checkout `/home/electron/ailiance` currently runs the branch
`hotfix/rollback-mascarade-9340-to-8004`, **not** `main`. PR #106 targets
`main`. Before deploying, decide how to reconcile:

- **Option A** — merge the hotfix into `main` as well (or confirm it is
  already there), merge PR #106, then deploy `main`. Production ends on
  `main` with both the rollback and the training feature.
- **Option B** — keep production on the hotfix branch and merge PR #106 into
  that branch instead of `main`.

Option A is preferred (production back on `main`). Do not switch the prod
branch blind — the hotfix is deployed for a reason (the :9340 bf16 incident).

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
    git checkout main          # per Option A
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
