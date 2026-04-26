"""Apertus MLX model implementation.

Apertus-70B from the Swiss AI Initiative. Modified Llama architecture:
- xIELU activation in MLP (direct, no gating — up_proj + down_proj only)
- QK-norm: RMSNorm on Q and K projections before RoPE
- GQA (Grouped Query Attention)
- Llama3-style RoPE scaling (theta=12M, factor=8)

Weight names match HF checkpoint swiss-ai/Apertus-70B-Instruct-2509:
  model.layers.N.attention_layernorm.weight
  model.layers.N.feedforward_layernorm.weight
  model.layers.N.mlp.act_fn.{alpha_p,alpha_n,beta,eps}
  model.layers.N.mlp.{up_proj,down_proj}.weight
  model.layers.N.self_attn.{q,k,v,o}_proj.weight
  model.layers.N.self_attn.{q,k}_norm.weight
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from src.mlx_models.xielu import XIELU


@dataclass
class ApertusConfig:
    vocab_size: int = 131072
    hidden_size: int = 8192
    intermediate_size: int = 43008
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-5
    rope_theta: float = 12_000_000.0
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = False
    qk_norm: bool = True


class ApertusRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class ApertusMLP(nn.Module):
    """Apertus MLP: up_proj → xIELU → down_proj (no gate_proj)."""

    def __init__(self, cfg: ApertusConfig):
        super().__init__()
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.act_fn = XIELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class ApertusAttention(nn.Module):
    def __init__(self, cfg: ApertusConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = ApertusRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = ApertusRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if mask is None and L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(queries.dtype)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.head_dim ** -0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class ApertusDecoderLayer(nn.Module):
    def __init__(self, cfg: ApertusConfig):
        super().__init__()
        self.self_attn = ApertusAttention(cfg)
        self.mlp = ApertusMLP(cfg)
        self.attention_layernorm = ApertusRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.feedforward_layernorm = ApertusRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        r = self.self_attn(self.attention_layernorm(x), mask=mask, cache=cache)
        x = x + r
        r = self.mlp(self.feedforward_layernorm(x))
        return x + r


class ApertusModel(nn.Module):
    def __init__(self, cfg: ApertusConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [ApertusDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = ApertusRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, tokens: mx.array, cache=None) -> mx.array:
        x = self.embed_tokens(tokens)

        _, L, _ = x.shape
        mask = None
        if L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(x.dtype)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, mask=mask, cache=layer_cache)

        return self.lm_head(self.norm(x))
