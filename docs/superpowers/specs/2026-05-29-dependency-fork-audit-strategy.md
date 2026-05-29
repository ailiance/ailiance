# Dependency Fork / Vendoring / Audit Strategy — ailiance stack

**Date:** 2026-05-29
**Author:** Clément Saillant (org `ailiance`)
**Status:** DRAFT — for human review before any upgrade or fork action
**Scope:** READ-ONLY inventory + strategy. No forks created, no deps installed
or upgraded, no pushes. The only artefact produced is this document.

## 1. Objective

Establish a **frozen, audited supply-chain baseline** for the ailiance serving
stack (gateway + omlx serving node + iact-bench) so that:

- every dependency is traced to an upstream (PyPI / GitHub) and a pinned version;
- a compromised upstream update cannot silently flow into production;
- upgrades go through a **human-in-the-loop (HITL)** diff review against the
  frozen baseline before any bump.

This document is the EU-AI-Act-style audit artefact for the dependency posture.

## 2. Sources inspected (local only)

| File | What it gave us |
|------|-----------------|
| `ailiance-gateway/pyproject.toml` | direct deps, `requires-python>=3.13`, Apache-2.0 |
| `ailiance-gateway/requirements-ci.txt` | mlx-free CI subset (Linux runner) |
| `ailiance-gateway/uv.lock` | full resolved graph — **106 locked packages** |
| `ailiance-gateway/.venv/.../site-packages` | **79 installed dist-info** (actual pinned versions on GrosMac, py3.14) |
| `ailiance-gateway/.gitmodules` | submodule `vendored/iact-bench` -> `github.com/ailiance/iact-bench` |
| `iact-bench/pyproject.toml` + `uv.lock` | bench deps, `requires-python>=3.14`, **62 locked packages** |
| `ailiance-omlx-node/` (repo) | how omlx is deployed; omlx version + upstream |
| `ailiance-omlx-node/src/omlx_deploy/steps.py` | actual omlx install command |

> Note: `omlx` itself is NOT installed in any local venv — it lives on MacStudio
> in `~/omlx-venv` (Apple-Silicon only). Versions for the MLX/omlx runtime below
> are taken from the omlx-node manifest and the gateway venv (which mirrors the
> Linux side); the **MacStudio omlx-venv pins are UNVERIFIED locally** and must be
> captured by `pip freeze` on the Studio node.

## 3. omlx identification

| Field | Value | Source / confidence |
|-------|-------|---------------------|
| Distribution name | `omlx` (PyPI) | `steps.py`: `pip install omlx==<ver>` — **verified local** |
| Pinned version | `0.3.9` | omlx-node manifest default + design/plan docs — **verified local** |
| Upstream repo | `https://github.com/jundot/omlx` | omlx-node `README.md` + design doc — **verified local (string), repo existence UNVERIFIED — confirm with user**|
| Install method | PyPI `omlx==0.3.9` into `~/omlx-venv` | `steps.py` uses PyPI; design doc *also* mentions `git+https://github.com/jundot/omlx` — **minor discrepancy, confirm canonical source with user** |
| Pulls (pinned git deps) | `mlx`, `mlx-lm`, `mlx-vlm`, `dflash-mlx` | omlx-node design doc — **partially verified; exact pins UNVERIFIED** |
| License | UNKNOWN | not present in any local file — **confirm with user / inspect upstream LICENSE** |
| Platform | macOS arm64 only | `steps.py::preflight` — verified |
| Local artefact | `~/Documents/Projets/_tools/omlx-tahoe.dmg` (616 MB, 2026-05-05) | possibly a packaged build / unrelated installer — **confirm relevance** |

**Criticality: TOP.** omlx is the single consolidated serving backend (`:8500`)
on MacStudio. A compromised omlx update directly executes in the inference path
on a 512 GB host. This is the #1 fork candidate.

## 4. Dependency inventory — gateway (direct)

`current` = version actually installed in `ailiance-gateway/.venv` (py3.14).
Licenses read from local dist-info METADATA where available.

| name | current | dir/trans | upstream | license | criticality |
|------|---------|-----------|----------|---------|-------------|
| **omlx** | 0.3.9 | runtime (ext) | github.com/jundot/omlx (PyPI) | UNKNOWN | **serving-critical** |
| **mlx** | 0.31.2 | direct | github.com/ml-explore/mlx | MIT | **serving-critical** |
| **mlx-lm** | 0.31.3 | direct | github.com/ml-explore/mlx-lm | MIT | **serving-critical** |
| mlx-metal | 0.31.2 | transitive (mlx) | ml-explore | MIT | serving-critical |
| **fastapi** | 0.136.1 | direct | github.com/fastapi/fastapi | MIT | **serving-critical** |
| **uvicorn[standard]** | 0.46.0 | direct | github.com/encode/uvicorn | BSD-3-Clause | **serving-critical** |
| **transformers** | 5.6.2 | trans (mlx-lm) | github.com/huggingface/transformers | Apache-2.0 | **serving-critical** |
| **pydantic** | 2.13.3 | direct | github.com/pydantic/pydantic | MIT | serving-critical |
| pydantic-core | 2.46.3 | trans (pydantic) | pydantic | MIT | serving-critical |
| **httpx** | 0.28.1 | direct | github.com/encode/httpx | BSD-3-Clause | serving-critical (worker proxy) |
| **safetensors** | 0.7.0 | direct | github.com/huggingface/safetensors | Apache-2.0 | serving-critical (weight load) |
| sentencepiece | 0.2.1 | trans (mlx-lm) | github.com/google/sentencepiece | Apache-2.0 | serving-critical (tokenizer) |
| numpy | 2.4.4 | direct (ci) / trans | github.com/numpy/numpy | BSD-3-Clause+ | serving-critical |
| pyyaml | 6.x | direct | github.com/yaml/pyyaml | MIT | config |
| prometheus-client | 0.21+ | direct | github.com/prometheus/client_python | Apache-2.0 | observability |
| structlog | 24+ | direct | github.com/hynek/structlog | MIT OR Apache-2.0 | observability |
| langfuse | 2.60.10 | direct (pinned `<3`) | github.com/langfuse/langfuse-python | MIT | observability (tracing) |
| networkx | 3.6.1+ | direct | github.com/networkx/networkx | BSD-3-Clause | peripheral |
| scipy | 1.10+ | direct | github.com/scipy/scipy | BSD-3-Clause | peripheral (audio resample) |
| soundfile | 0.12+ | direct | github.com/bastibe/python-soundfile | BSD-3-Clause | peripheral (realtime) |
| msgpack | 1.0+ | direct | github.com/msgpack/msgpack-python | Apache-2.0 | peripheral (realtime) |
| websockets | 12+ | direct | github.com/python-websockets/websockets | BSD-3-Clause | peripheral (realtime) |
| python-multipart | 0.0.9+ | direct | github.com/Kludex/python-multipart | Apache-2.0 | peripheral (uploads) |
| pdfminer.six | 20231228+ | direct | github.com/pdfminer/pdfminer.six | MIT | peripheral (file extract) |
| python-docx | 1.1+ | direct | github.com/python-openxml/python-docx | MIT | peripheral |
| openpyxl | 3.1+ | direct | foss.heptapod.net/openpyxl | MIT | peripheral |
| python-pptx | 1.0+ | direct | github.com/scanny/python-pptx | MIT | peripheral |
| beautifulsoup4 | 4.12+ | direct | crummy.com/software/BeautifulSoup | MIT | peripheral |
| pytesseract | 0.3.10+ | direct | github.com/madmaze/pytesseract | Apache-2.0 | peripheral (OCR) |
| pillow | 10.0+ | direct | github.com/python-pillow/Pillow | MIT-CMU | peripheral |
| pdf2image | 1.17+ | direct | github.com/Belval/pdf2image | MIT | peripheral |
| jwcrypto | 1.5.7 | direct | github.com/latchset/jwcrypto | LGPL-3.0 ⚠️ | peripheral (Gaia-X VC) |
| pyld | 3.0.0 | direct | github.com/digitalbazaar/pyld | BSD-3-Clause | peripheral (Gaia-X JSON-LD) |
| cryptography | 48.0.0 | direct | github.com/pyca/cryptography | Apache-2.0 OR BSD | serving-critical (signing keys) |
| sentence-transformers | (extra `router`) | optional | github.com/UKPLab/sentence-transformers | Apache-2.0 | **serving-critical (router)** |
| torch | (extra `router`, >=2.6) | optional | github.com/pytorch/pytorch | BSD-3-Clause | **serving-critical (router)** |
| datasets | (extra `data`) | optional | github.com/huggingface/datasets | Apache-2.0 | peripheral (training) |
| huggingface-hub | (extra `data`) | optional | github.com/huggingface/huggingface_hub | Apache-2.0 | serving-critical (weight pull) |

**Gateway totals:** ~38 direct deps (incl. optional extras), **106 locked / 79
installed transitive** packages.

⚠️ **License flag:** `jwcrypto` is **LGPL-3.0** — the only copyleft lib in the
direct set; the gateway is Apache-2.0. LGPL dynamic-linking is fine for an
unmodified import, but flag for legal review if jwcrypto is ever forked/modified.

## 5. Dependency inventory — iact-bench (direct)

`requires-python>=3.14`. **62 locked packages.** No mlx in the default set
(inference is an opt-in extra; bench talks to remote workers via HTTP).

| name | pinned (`pyproject`) | dir/trans | upstream | license | criticality |
|------|----------------------|-----------|----------|---------|-------------|
| huggingface-hub | >=0.27 | direct | huggingface/huggingface_hub | Apache-2.0 | data pull |
| datasets | >=3.0 | direct | huggingface/datasets | Apache-2.0 | data |
| requests | >=2.32 | direct | psf/requests | Apache-2.0 | bench HTTP |
| pyyaml | >=6.0 | direct | yaml/pyyaml | MIT | config |
| tenacity | >=9.0 | direct | jd/tenacity | Apache-2.0 | retry logic |
| mlx-lm | >=0.20 (extra `inference`) | optional | ml-explore/mlx-lm | MIT | serving (opt-in) |
| pytest / pytest-mock / ruff | dev | dev | — | MIT | dev-only |

The bench is itself **vendored into the gateway** as a git submodule
(`vendored/iact-bench` -> `github.com/ailiance/iact-bench`) — already on the
ailiance org, already commit-pinned by submodule SHA. Good baseline pattern.

## 6. Criticality tiers (drives the fork decision)

- **Tier 0 — serving-critical, fork candidates:** `omlx`, `mlx`, `mlx-lm`,
  `mlx-vlm`, `dflash-mlx`, `transformers`, `safetensors`, `sentencepiece`,
  `torch` + `sentence-transformers` (router), `fastapi`, `uvicorn`. These run
  arbitrary model code / weight loading / the request path. A poisoned update =
  remote code execution on a serving host.
- **Tier 1 — hash-pin + SBOM, no fork:** `httpx`, `pydantic`, `cryptography`,
  `huggingface-hub`, `numpy`, `langfuse`, `prometheus-client`, `structlog`.
  High-trust, widely-audited; freeze by hash, watch advisories.
- **Tier 2 — hash-pin only (peripheral):** file-extraction libs, realtime/audio
  libs, Gaia-X libs, networkx, pyyaml, tenacity, requests.

## 7. Strategy options

### Option A — GitHub forks into the `ailiance` org
Fork each Tier-0 repo to `ailiance/<lib>`, pin installs to the fork's commit.
- **Pros:** full control of the exact tree; can carry security patches; PRs
  back upstream; survives upstream deletion/takeover.
- **Cons:** heaviest. N forks to keep in sync; risk of drift / staleness; CI to
  rebuild MLX/torch wheels is non-trivial (native + Metal).

### Option B — Vendored snapshots pinned by commit/SHA
Git submodules (like the existing `vendored/iact-bench`) or copied `vendored/`
trees, each pinned to an exact commit.
- **Pros:** strongest reproducibility; classic supply-chain freeze; the tree you
  audited is byte-for-byte the tree you ship; no PyPI dependency at deploy time.
- **Cons:** updates are manual; large native libs (torch, mlx) are impractical to
  vendor as source (multi-GB, native build). Best for pure-Python libs and for
  small high-risk libs.

### Option C — Pinned lockfile + hash pinning + SBOM (lightest)
Keep PyPI installs but enforce `uv.lock` / `requirements.txt` with
`--require-hashes`, generate a CycloneDX SBOM per release, and gate upgrades on
advisory scans (osv-scanner / pip-audit).
- **Pros:** lightest; no fork maintenance; hashes make a swapped artefact fail
  the install; SBOM is the AI-Act-grade audit record.
- **Cons:** still depends on PyPI availability; a *signed-but-malicious* upstream
  release would pass hash check (hashes verify integrity, not intent — the HITL
  diff review covers intent).

## 8. Recommendation — tiered hybrid

1. **Tier 0 (serving-critical) -> Option A fork + Option C hashes.**
   - **`omlx`** (and its `jundot/*` siblings if any): **full fork to
     `ailiance/omlx`**, pinned to the audited commit of `0.3.9`. Highest priority
     — it is the least-known, lowest-reputation, highest-privilege dep. Audit its
     source before the first bump; record its LICENSE (currently unknown).
   - **`mlx`, `mlx-lm`, `mlx-vlm`, `dflash-mlx`**: fork to `ailiance/` for
     traceability, but **install from the fork by commit SHA** rather than
     re-publishing wheels (build from the fork tag). These are high-reputation
     (ml-explore = Apple) but they are on the serving path and omlx pins them.
   - **`torch`, `transformers`, `fastapi`, `uvicorn`, `safetensors`,
     `sentencepiece`, `sentence-transformers`**: do **not** fork (too large /
     too high-trust); hash-pin (Option C) + advisory watch. Fork only on a
     concrete trigger (CVE you must patch ahead of upstream, or upstream
     compromise).
2. **Tier 1 & 2 -> Option C.** `uv.lock` with `--require-hashes`, CycloneDX SBOM
   committed per release, `osv-scanner` / `pip-audit` in CI.
3. **iact-bench**: already a commit-pinned submodule on `ailiance/` — keep as-is;
   add its own hash-pinned lock + SBOM.

This forks **only what is genuinely low-trust and high-privilege** (omlx, and the
MLX family it pins) while leaving the well-audited big libs on a hash-pinned
lockfile.

## 9. HITL upgrade workflow (frozen baseline -> approved bump)

```
[frozen baseline]                          audit trail location
   uv.lock (hashes) + SBOM (CycloneDX)  -> docs/audit/sbom/<date>.cdx.json
   fork SHAs for Tier-0                 -> docs/audit/baseline.lock.md
        |
        v
1. upstream change detected (Dependabot / Renovate / osv-scanner alert)
        |
        v
2. produce DIFF vs frozen baseline:
   - PyPI lib:  pip download <new> ; diff source tree vs audited version
   - forked lib: git fetch upstream ; git log/diff baseline_sha..upstream
        |
        v
3. HUMAN review of the diff (focus: install hooks, network calls, eval/exec,
   new transitive deps, license changes). Record verdict.
        |
        v
4. on approval: bump pin, regenerate uv.lock + SBOM, move fork SHA forward,
   commit "chore(deps): bump <lib> X->Y (reviewed <reviewer> <date>)".
        |
        v
5. CI re-runs --require-hashes install + osv-scanner; deploy.
```

- **Audit trail lives in-repo** under `docs/audit/` (baseline lock + per-release
  SBOMs + review verdicts), mirroring the iact-bench NDJSON audit-trail pattern.
  This is the artefact an EU-AI-Act auditor inspects.
- **No bump without a recorded human verdict.** Renovate/Dependabot may *open*
  PRs but must not auto-merge serving-critical libs.

## 10. Unknowns to confirm with the user

1. **omlx upstream** — confirm `github.com/jundot/omlx` is canonical, public, and
   the real source of `0.3.9`. Resolve the PyPI-vs-`git+` discrepancy
   (`steps.py` uses PyPI; design doc mentions git). Capture omlx's **LICENSE**.
2. **omlx-venv pins (MacStudio)** — run `~/omlx-venv/bin/pip freeze` on Studio to
   get the *actual* runtime pins (mlx/mlx-lm/mlx-vlm/dflash-mlx + transitives);
   the local venv is the Linux mirror, not the Studio truth.
3. **Fork target org** — confirm all forks go to `ailiance` (vs
   `electron-rare` / `L-electron-Rare`). Apache-2.0 house license assumed.
4. **Scope** — direct-only audit, or full transitive (106 gateway / 62 bench)?
   Recommendation above is direct + Tier-0 transitives; confirm depth.
5. **License constraints** — `jwcrypto` is **LGPL-3.0** in an Apache-2.0 product;
   confirm dynamic-link-only use is acceptable, or replace.
6. **`omlx-tahoe.dmg`** (616 MB in `_tools/`) — is this a packaged omlx build to
   include in the baseline, or unrelated? Confirm.
7. **Wheel-build capacity** — forking mlx/torch implies rebuilding native+Metal
   wheels. Confirm there is build infra (or keep those hash-pinned only).
8. **Tooling choice** — Renovate vs Dependabot; osv-scanner vs pip-audit;
   CycloneDX generator (`cyclonedx-py`) — confirm preferences.
```

## 11. Decisions (2026-05-29, user)

- **Strategy = Pin-only + SBOM** as the baseline for *every* ecosystem
  (lockfiles + hash-pinning `--require-hashes`/`@sha256:`/pinned formulae +
  CycloneDX SBOM). **No mass forking.**
- **Vendoring is the escalation, not the default** — reserved for the single
  Tier-0 untrusted case: **omlx** (license unknown, high-privilege, runs in
  the inference path). If omlx is vendored, it goes to a **dedicated vendoring
  org** (kept separate from the `ailiance` product repos), pinned by SHA.
- **jwcrypto = KEPT** — LGPL-3.0 used via dynamic linking only (no copyleft
  contamination of the Apache-2.0 product); record it in `NOTICE` + SBOM.

### omlx resolved (2026-05-29, MacStudio bastion probe)
- **omlx 0.3.9 = Apache-2.0** (dist-info METADATA: `License: Apache-2.0`,
  classifier OSI Apache, `LICENSE` file present). Upstream confirmed
  `github.com/jundot/omlx`. **License-unknown blocker is CLEARED.** The
  vendoring case now rests on *reputation + privilege* (single-maintainer
  repo running in the inference path), not licensing.
- **omlx pulls 5 deps via `git+…@<commit>` (NOT PyPI) — the real supply-chain
  surface:** `mlx-lm @ ml-explore@ed1fca4`, `mlx-embeddings @ Blaizzy@32981fa`,
  `mlx-vlm @ Blaizzy@f96138e`, `dflash-mlx @ bstnxbt@1ba6713`, `mlx-audio @
  Blaizzy@5175326` (audio extra). Commit-pinned (good) but they point at
  individual-maintainer forks — risk if a repo is deleted/force-pushed.
  **Action:** mirror those 5 commits to the dedicated vendoring org (this is
  the strongest argument for vendoring omlx + its git deps together).
- `mlx>=0.31.2`, `transformers>=5.0.0`, `fastapi>=0.108.0`, etc. are `>=`
  ranges → pin exact installed versions in the baseline lock.
- The uv-style `~/omlx-venv` has **no `pip`**; use `~/omlx-venv/bin/python`
  or read `*.dist-info/METADATA` directly (done here).

### Installed MLX-family pins on Studio (`~/omlx-venv`, 2026-05-29)
| Package | Installed version |
|---------|-------------------|
| `mlx` | 0.31.2 |
| `mlx-metal` | 0.31.2 |
| `mlx-lm` | 0.31.3 (git `ml-explore@ed1fca4`) |
| `mlx-vlm` | 0.5.0 (git `Blaizzy@f96138e`) |
| `mlx-embeddings` | 0.1.0 (git `Blaizzy@32981fa`) |
| `mlx-audio` | 0.4.3 (git `Blaizzy@5175326`) |
| `dflash-mlx` | 0.1.7 (git `bstnxbt@1ba6713`) |

These 7 + omlx 0.3.9 = the **Tier-0 vendoring set** (Apache-2.0 omlx + its
git-pinned MLX forks). Mirror by commit SHA to the dedicated vendoring org.

### brew baseline (partial, 2026-05-29)
- **GrosMac**: 80 `brew leaves`; critical: `tailscale 1.98.3`, `autossh 1.4g`,
  `ffmpeg 8.1.1`, `git 2.54.0`. `cloudflared` is NOT brew-managed (installed
  via Cloudflare's own channel — track separately).
- **MacStudio**: 23 `brew leaves`; `tailscale 1.96.4`, `ffmpeg 8.1.1`,
  `git 2.53.0`. NB: tailscale older than GrosMac (1.98.3) — flag for alignment.
- **macM1**: pending — electron-server lacks macM1's SSH host key
  (`Host key verification failed`); accept the host key on electron-server,
  then re-run `/opt/homebrew/bin/brew leaves`.

### Progress 2026-05-29
- ✅ **Docker FROM digest pins applied & merged** — 25 Dockerfiles on
  iact-bench `master` (`4d14f53`) + qet's on `feat/qet-validator` (`dcc5f44`).
  26 Dockerfiles total (the doc earlier said "27" — actual is 26).
- ✅ brew baseline: GrosMac + MacStudio captured (macM1 pending host key).
- ✅ SBOM + pip-audit generated (`sbom/cyclonedx.json` + `sbom/pip-audit.json`
  in both repos; gateway commit `a87b735`, iact-bench `3c1954c`). Gateway = 80
  components, iact-bench = 45. Method: `uv export` → `uvx cyclonedx-py` ;
  `uvx pip-audit --path .venv/...site-packages -s osv`.
- ✅ Tier-0 vendoring script committed (gateway `26a001b`,
  `scripts/vendor-tier0.sh`, 6 SHAs incl. omlx v0.3.9 `8cad1212`).

### Vulnerabilities found (pip-audit, 2026-05-29) — HITL upgrade candidates
| Repo | Package | Version | Advisory | Fix |
|------|---------|---------|----------|-----|
| gateway | **starlette** | 1.0.0 | CVE-2026-48710 / PYSEC-2026-161 | **1.0.1** (ASGI under FastAPI — prod, prioritise) |
| gateway | urllib3 | 2.6.3 | CVE-2026-44431 + CVE-2026-44432 | 2.7.0 |
| gateway + bench | idna | 3.13 | CVE-2026-45409 | 3.15 |

### ⚠️ Defect found: stale `uv.lock` (gateway)
The committed `uv.lock` is **out of sync with `pyproject.toml`** — missing
~13 declared deps (beautifulsoup4, cryptography, jwcrypto, msgpack, openpyxl,
pdf2image, pdfminer-six, pillow, pyld, pytesseract, python-docx,
python-multipart, python-pptx — the Gaia-X + doc-processing work). The SBOM
agent's `uv export` resynced it (+485 lines); that incidental change was
**reverted** (no blind lockfile mutation). Fix deliberately: `uv lock` +
review the regenerated set, then regenerate the SBOM from the synced lock.

### Open items still needed before execution
- **Deliberate `uv lock`** on gateway (stale-lock defect above) → re-SBOM.
- Vuln upgrades above (starlette first) via HITL review.
- macM1 brew (host-key accept on electron-server).
- Tier-0 vendoring run → **needs the dedicated org created first** (GitHub
  orgs cannot be created via API/`gh` — manual web step); then
  `ORG=<org> ./scripts/vendor-tier0.sh`.
- Full `brew bundle dump` (vs `leaves`) once host set is final.
- npm/brew/cargo/apt/Docker extension inventory — NOT yet written (the
  extension agent hit a session limit 2026-05-29). Re-run after reset.
- Tooling proposal: `uv lock` + `pip-audit`/`osv-scanner` + `cyclonedx-py`;
  Renovate for PR-only (no auto-merge of serving-critical).

## 12. Ecosystem extension: npm / brew / cargo / apt / Docker (2026-05-29)

Inventory done inline (the dedicated agent hit a session limit). Local-only;
host-resident package sets (brew on the Macs) marked UNVERIFIED.

| Ecosystem | Artifact found | Pinning state | Criticality | Freeze action |
|-----------|----------------|---------------|-------------|---------------|
| **npm** | `ailiance-agent` (`/Users/electron/code/ailiance-agent`: root + `webview-ui` have `package-lock.json`; `cli/`, `standalone/`, `ailiance-demo/*` also present) | Lockfiles present; **no `postinstall` scripts** (lower risk) | Medium (dev tooling, not in inference path) | `npm ci` against committed lockfile + `npm audit` snapshot in baseline; verify `cli/`+`standalone/` also have locks |
| **Docker (output)** | 27 `ghcr.io/electron-rare/iact-bench-*` validator images | **`@sha256:` digest-pinned** in YAML config (good) | High (sandbox boundary) | Keep digest pins; record digests in SBOM |
| **Docker (base/input)** | 27 Dockerfile `FROM` lines: `debian:13-slim`, `node:22-bookworm-slim`, `python:3.13-slim(-bookworm)`, `rust:1-slim-bookworm`, `ubuntu:24.04`, `espressif/idf:v5.5`, `mcr.microsoft.com/dotnet/sdk:9.0` | **0/27 digest-pinned — all mutable tags** ⚠️ | High | **Biggest gap.** Pin every `FROM` to `@sha256:`; a rebuild currently can pull a changed base under a stable tag |
| **brew** | No Brewfile in workspace | UNVERIFIED | Medium (host tooling: tailscale, cloudflared, autossh) | `brew bundle dump` per Mac host → commit Brewfile + `brew pin` critical formulae |
| **cargo** | None found locally | n/a | — | Rust deps live inside the `iact-bench-rust*` Docker images (`Cargo.lock` if/when projects added) |
| **apt** | Inside Dockerfiles (`apt-get install`) | Unpinned versions | Medium | Pin apt package versions in Dockerfiles OR rely on the base-image digest pin (above) for reproducibility |

**Tiered mapping:** none of these warrant a Tier-0 fork — all fit **pin-only +
SBOM** (the chosen baseline). The single highest-leverage action is pinning
the 27 Dockerfile `FROM` bases by digest.

**HITL upgrade gates per ecosystem:**
- npm: bump via lockfile diff + `npm audit` review → human verdict → `npm ci`.
- Docker: change a `@sha256:` only after reviewing the new base's diff.
- brew: `brew bundle` from a committed, reviewed Brewfile; no ad-hoc upgrades.

**New unknowns:**
1. `cli/` and `standalone/` npm lockfile coverage — confirm all have locks.
2. brew formulae per host (needs `brew bundle dump` on GrosMac/MacStudio/macM1).
3. Acceptable to pin Dockerfile `FROM` by digest given multi-arch (amd64 host
   vs the Macs)? Digest is arch-specific — may need per-arch pins or a manifest
   list digest.
