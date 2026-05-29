#!/usr/bin/env bash
# Tier-0 supply-chain vendoring — mirror omlx + its git-pinned MLX forks by
# SHA into a dedicated vendoring org (frozen, audited baseline).
#
# Why: pin-only + SBOM is the baseline for everything (see
# docs/superpowers/specs/2026-05-29-dependency-fork-audit-strategy.md).
# omlx is the one Tier-0 escalation: Apache-2.0 but single-maintainer, runs in
# the inference path, and pulls 5 deps via git+@<commit> from personal forks
# that could be deleted or force-pushed. Mirroring those commits by SHA makes
# the serving stack reproducible independent of upstream availability.
#
# The per-repo `git checkout <sha>` fails LOUD if the pinned commit is gone —
# that is the force-push / delete detector, by design.
#
# PREREQUISITE: the dedicated vendoring org must ALREADY EXIST. GitHub orgs
# cannot be created via API/gh — create it on the web first. `gh auth status`
# must show repo-create rights in that org.
#
# Usage:  ORG=<vendoring-org> ./scripts/vendor-tier0.sh
set -euo pipefail
: "${ORG:?set ORG to the dedicated vendoring org (must already exist on GitHub)}"
command -v gh  >/dev/null || { echo "gh not found"; exit 1; }
command -v git >/dev/null || { echo "git not found"; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# name            upstream_url                                pinned_sha
# (omlx 0.3.9 = tag v0.3.9; the 5 forks = the git+@<commit> pins from omlx's
#  Requires-Dist, read off MacStudio ~/omlx-venv 2026-05-29.)
read -r -d '' MANIFEST <<'EOF' || true
omlx           https://github.com/jundot/omlx            8cad1212729b4efaeae7e56e131b5b80f8d44f85
mlx-lm         https://github.com/ml-explore/mlx-lm       ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd
mlx-embeddings https://github.com/Blaizzy/mlx-embeddings  32981fa4e8064ed664b52071789dd18271fe4206
mlx-vlm        https://github.com/Blaizzy/mlx-vlm         f96138eef1f5ce7fb5d97f8dd41a664a195b5659
dflash-mlx     https://github.com/bstnxbt/dflash-mlx      1ba671372b289c025b435c1a13aabb4bfb80b183
mlx-audio      https://github.com/Blaizzy/mlx-audio       51753266e0a4f766fd5e6fbc46652224efc23981
EOF

printf '%s\n' "$MANIFEST" | while read -r name url sha; do
  [ -z "$name" ] && continue
  dest="$WORK/$name"
  echo ">> $name @ $sha"
  git clone --quiet "$url" "$dest"
  # fetch the exact commit in case it is not on a default branch
  git -C "$dest" fetch --quiet origin "$sha" 2>/dev/null || true
  # FAILS LOUD if the pinned SHA is gone (force-push / delete detector):
  git -C "$dest" checkout --quiet "$sha"
  # idempotent repo create (ignore "already exists")
  gh repo create "$ORG/${name}-vendored" --private \
     --description "Vendored $name pinned @ $sha" >/dev/null 2>&1 || true
  git -C "$dest" remote add vendor "https://github.com/$ORG/${name}-vendored.git"
  git -C "$dest" push --quiet vendor "$sha:refs/heads/vendored-$sha"
  git -C "$dest" tag -f "vendored-$sha" "$sha" >/dev/null 2>&1 || true
  git -C "$dest" push --quiet --force vendor "vendored-$sha" 2>/dev/null || true
  echo "   mirrored -> $ORG/${name}-vendored  (branch+tag vendored-$sha)"
done

cat <<'NEXT'

Done. Next (HITL):
  - Record the 6 SHAs in the CycloneDX SBOM.
  - Repoint the gateway deps to the vendored remotes (uv source overrides),
    e.g. mlx-lm @ git+https://github.com/<ORG>/mlx-lm-vendored@<sha>.
  - Any future bump = review upstream diff vs the frozen SHA, then re-run.
NEXT
