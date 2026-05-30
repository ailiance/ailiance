# omlx: Mistral-Small-3.1 (mistral3) text corrupted with byte-level surface forms

**Status:** open upstream (jundot/omlx). Worked around gateway-side
(`_repair_byte_level` in `server.py`, PR #141). Ready-to-file issue below
(sanitized of internal infra details).

---

## Title
Mistral-Small-3.1 (mistral3) text output corrupted with byte-level surface forms (Ġ/Ã©/Ċ)

## Body

`Mistral-Small-3.1-24B-Instruct` (`model_type: mistral3`,
`Mistral3ForConditionalGeneration`) served by `omlx serve` returns text where
the GPT-2 byte-level surface forms are never byte-decoded: spaces appear as
`Ġ`, newlines as `Ċ`, accents as latin-1 mojibake (`Ã©`→`é`, `Ã§`→`ç`). The
token ids are correct — only the final byte-decode is missing.

```
prompt:  "Réponds en français: cite trois villes."
omlx:    "PourĠciterĠtroisĠvillesĠenĠfranÃ§aisĠ:Ċ..."
expected:"Pour citer trois villes en français :\n..."
```

**Environment:** omlx 0.3.9, mlx-lm 0.31.3, mlx-vlm 0.5.0, transformers 5.9.0,
macOS Apple Silicon, Python 3.12, `Mistral-Small-3.1-24B-Instruct-MLX-4bit`.

**Scope — only this model.** On the same server, Qwen3-Coder-30B,
DeepSeek-R1-Distill-Qwen-32B, EuroLLM-22B and gemma-4-E4B all serve clean
UTF-8. Only the `mistral3` model leaks (the only served VLM with a
`tekken.json` + Mistral tool-call chat template).

**Standalone is clean.** `mlx_lm.load(path)` picks `BPEStreamingDetokenizer`
and `mlx_lm.generate(...)` is correct. The model, tokenizer and mlx-lm's
detokenizer are fine — the corruption is in omlx's serving/streaming detok.

**Ruled out:**
- Not stale — full restart reproduces deterministically.
- Not the engine. `mistral3` routes to the VLM engine (`vision_config`), but
  mlx-vlm 0.5.0 can't load the quant
  (`Received 6 parameters not in model: multi_modal_projector.linear_1.scales/biases, ...`)
  so it falls back. Removing `vision_config` forces the **batched** engine and
  the corruption persists there too.
- Not `tekken.json` — disabling it changes nothing.
- `_get_detokenizer` returns `tokenizer.detokenizer` = `BPEStreamingDetokenizer`
  (byte-decoding) standalone, yet served output is the raw vocab surface
  (the `SPMStreamingDetokenizer`-on-byte-level-BPE signature).

**Repro:**
```bash
omlx serve --model-dir <dir> --port 8500 --no-cache
curl -s :8500/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Mistral-Small-3.1-24B-Instruct-MLX-4bit","messages":[{"role":"user","content":"Réponds en français: cite trois villes."}],"max_tokens":40}'
# -> content contains literal Ġ / Ã© / Ċ
```

The corruption is deterministically reversible via the GPT-2 byte decoder
(client-side workaround), confirming the token stream is correct and only
server-side detokenization is wrong. Likely a fast/slow-tokenizer or
byte-level-vocab mismatch in the scheduler's streaming detok for Mistral
tokenizers. Happy to test a patch.
