"""xIELU activation function for MLX.

Piecewise trainable activation from arXiv:2411.13010.
Used by Apertus-70B (Swiss AI Initiative).

  x > 0:  f(x) = alpha_p * x^2 + beta * x
  x <= 0: f(x) = alpha_n * (exp(min(x, eps)) - 1 - x) + beta * x

Apertus checkpoints store alpha_p, alpha_n, beta, eps as direct
values (not softplus-reparameterized).
"""

import mlx.core as mx
import mlx.nn as nn


class XIELU(nn.Module):
    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
    ):
        super().__init__()
        self.alpha_p = mx.array(alpha_p_init)
        self.alpha_n = mx.array(alpha_n_init)
        self.beta = mx.array(beta)
        self.eps = mx.array(eps)

    def __call__(self, x: mx.array) -> mx.array:
        pos = self.alpha_p * x * x + self.beta * x
        neg = self.alpha_n * (mx.exp(mx.minimum(x, self.eps)) - 1.0 - x) + self.beta * x
        return mx.where(x > 0, pos, neg)
