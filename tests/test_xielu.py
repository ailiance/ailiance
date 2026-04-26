import math

import mlx.core as mx
import mlx.nn as nn


def test_xielu_positive_quadratic():
    """For x > 0: f(x) = alpha_p * x^2 + beta * x"""
    from src.mlx_models.xielu import XIELU

    act = XIELU(alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5)
    x = mx.array([1.0, 2.0, 3.0])
    y = act(x)
    mx.eval(y)

    alpha_p = float(nn.softplus(mx.array(math.log(math.exp(0.8) - 1))))
    expected = [alpha_p * v * v + 0.5 * v for v in [1.0, 2.0, 3.0]]
    for got, exp in zip(y.tolist(), expected):
        assert abs(got - exp) < 1e-4, f"{got} != {exp}"


def test_xielu_negative_exponential():
    """For x <= 0: involves exp(x) term"""
    from src.mlx_models.xielu import XIELU

    act = XIELU(alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5)
    x = mx.array([-1.0, -2.0, -0.5])
    y = act(x)
    mx.eval(y)

    for v in y.tolist():
        assert math.isfinite(v), f"non-finite output: {v}"


def test_xielu_zero_continuous():
    """f(0) should be 0 (continuous at origin)"""
    from src.mlx_models.xielu import XIELU

    act = XIELU()
    x = mx.array([0.0])
    y = act(x)
    mx.eval(y)
    assert abs(y.item()) < 1e-5


def test_xielu_learnable_params():
    """alpha_p and alpha_n should be nn.Module parameters"""
    from src.mlx_models.xielu import XIELU

    act = XIELU()
    params = act.parameters()
    flat = nn.utils.tree_flatten(params)
    param_names = [name for name, _ in flat]
    assert "log_alpha_p" in param_names
    assert "log_alpha_n" in param_names
    assert len([n for n, _ in flat]) == 2
