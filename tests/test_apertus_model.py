import mlx.core as mx


def test_apertus_model_forward_shape():
    """Tiny Apertus model produces correct output shape."""
    from src.mlx_models.apertus import ApertusModel, ApertusConfig

    cfg = ApertusConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = ApertusModel(cfg)
    tokens = mx.array([[1, 2, 3, 4]])  # (1, 4)
    out = model(tokens)
    mx.eval(out)
    assert out.shape == (1, 4, 256), f"Expected (1, 4, 256), got {out.shape}"


def test_apertus_uses_xielu():
    """MLP layers use xIELU, not SiLU."""
    from src.mlx_models.apertus import ApertusModel, ApertusConfig
    from src.mlx_models.xielu import XIELU

    cfg = ApertusConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )
    model = ApertusModel(cfg)
    mlp = model.layers[0].mlp
    assert isinstance(mlp.act, XIELU), f"Expected XIELU, got {type(mlp.act)}"


def test_apertus_qk_norm():
    """Attention layers apply QK-norm (RMSNorm on Q and K)."""
    from src.mlx_models.apertus import ApertusModel, ApertusConfig

    cfg = ApertusConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )
    model = ApertusModel(cfg)
    attn = model.layers[0].self_attn
    assert hasattr(attn, "q_norm"), "Missing q_norm (QK-norm)"
    assert hasattr(attn, "k_norm"), "Missing k_norm (QK-norm)"
