# Model Provenance Records

EU AI Act Annex IV §1(c) requires that every deployed model includes a
supply-chain record covering: source, licence, lineage, modifications.

## One file per deployed model

Each `*.json` in this directory documents a single served model:

| Field | Meaning |
|---|---|
| `deployment_id` | Stable id used by the gateway (`ailiance/<alias>`) |
| `deployed_at_utc` | Timestamp of first-bring-up |
| `deployed_on_host` | Tailscale hostname where the weights live |
| `served_via` | Runtime stack (llama.cpp, MLX, vLLM…) and notable flags |
| `source.huggingface_repo` | Provider's repo |
| `source.repo_commit_sha` | Pinned commit — protects against silent upstream changes |
| `source.license_spdx` | SPDX id, or `unknown` |
| `weights_file.sha256_expected` | Tied to the LFS pointer; verify after download |
| `architecture` | Total / active params, expert count, attention type |
| `quantization.method` | `Q4_K_M`, `MLX-4bit`, `BF16`, etc. |
| `quantization.produced_by` | Who produced the quant — Qwen team, unsloth, bartowski, internal |
| `modifications_post_download` | List of any further tuning, merging, distilling we apply |
| `intended_use` | Plain-text statement of what the model is exposed for |
| `out_of_scope` | Use cases explicitly excluded |

## Workflow

1. Before deploying a new model, capture a record from the HF API:

   ```bash
   curl -fsS "https://huggingface.co/api/models/$REPO?blobs=true" \
     | jq '{repo:.id, sha:.sha, license:.cardData.license, lastModified:.lastModified}'
   ```

2. Pin the commit and the file SHA256 (from the LFS metadata).

3. Drop a JSON file here, commit it together with whatever cockpit / gateway
   change exposes the model.

4. After downloading, run `sha256sum` against the file and verify it matches
   `weights_file.sha256_expected` — log the discrepancy if not.

5. Any later fine-tune or merge becomes a NEW provenance file (not an edit
   of the original) — append it to `modifications_post_download` of the
   parent record.
