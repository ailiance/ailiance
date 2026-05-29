#!/usr/bin/env bash
# Vendor Tier-0 deps into ONE private repo as frozen source snapshots (subdirs),
# pinned by upstream SHA. The per-dep `git checkout <sha>` fails loud if a commit
# is gone (force-push/delete detector).
set -euo pipefail
REPO="${REPO:-ailiance/vendored}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

gh repo view "$REPO" >/dev/null 2>&1 || \
  gh repo create "$REPO" --private \
    -d "Tier-0 vendored deps frozen by SHA (supply-chain HITL baseline)"

if git clone -q "https://github.com/$REPO.git" "$WORK/repo" 2>/dev/null && \
   [ -d "$WORK/repo/.git" ]; then
  cd "$WORK/repo"
else
  mkdir -p "$WORK/repo"; cd "$WORK/repo"; git init -q
  git remote add origin "https://github.com/$REPO.git"
fi

mkdir -p vendored
{
  echo "# Tier-0 vendored dependencies"
  echo
  echo "Frozen source snapshots pinned by upstream SHA (no \`.git\`)."
  echo "Strategy: ailiance-gateway docs/superpowers/specs/2026-05-29-dependency-fork-audit-strategy.md"
  echo
  echo "| dep | upstream | pinned sha |"
  echo "|-----|----------|------------|"
} > MANIFEST.md

while read -r name url sha; do
  [ -z "$name" ] && continue
  echo ">> $name @ $sha"
  t="$(mktemp -d)"
  git clone -q "$url" "$t"
  git -C "$t" fetch -q origin "$sha" 2>/dev/null || true
  git -C "$t" checkout -q "$sha"
  rm -rf "$t/.git"
  rm -rf "vendored/$name"; mkdir -p "vendored/$name"
  cp -R "$t/." "vendored/$name/"
  echo "| $name | $url | \`$sha\` |" >> MANIFEST.md
  rm -rf "$t"
done <<'PAIRS'
omlx https://github.com/jundot/omlx 8cad1212729b4efaeae7e56e131b5b80f8d44f85
mlx-lm https://github.com/ml-explore/mlx-lm ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd
mlx-embeddings https://github.com/Blaizzy/mlx-embeddings 32981fa4e8064ed664b52071789dd18271fe4206
mlx-vlm https://github.com/Blaizzy/mlx-vlm f96138eef1f5ce7fb5d97f8dd41a664a195b5659
dflash-mlx https://github.com/bstnxbt/dflash-mlx 1ba671372b289c025b435c1a13aabb4bfb80b183
mlx-audio https://github.com/Blaizzy/mlx-audio 51753266e0a4f766fd5e6fbc46652224efc23981
PAIRS

git add -A
git -c user.email=clement@saillant.cc -c user.name=electron-rare \
    commit -q -m "vendor: freeze tier-0 deps by SHA" || { echo "nothing to commit"; exit 0; }
git branch -M main
git push -u origin main
echo "VENDOR_DONE: $REPO populated ($(ls vendored | wc -l | tr -d ' ') deps)"
