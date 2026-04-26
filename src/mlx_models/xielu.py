"""xIELU activation function for MLX.

Piecewise trainable activation from arXiv:2411.13010.
Used by Apertus-70B (Swiss AI Initiative).

  x > 0:  f(x) = alpha_p * x^2 + beta * x
  x <= 0: f(x) = alpha_n * (exp(min(x, eps)) - 1 - x) + beta * x

alpha_p and alpha_n are per-layer learnable parameters stored
via softplus reparameterization.
"""

import math

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
        self.log_alpha_p = mx.array(math.log(math.exp(alpha_p_init) - 1))
        self.log_alpha_n = mx.array(math.log(math.exp(alpha_n_init - beta) - 1))
        self._beta = beta
        self._eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        alpha_p = nn.softplus(self.log_alpha_p)
        alpha_n = self._beta + nn.softplus(self.log_alpha_n)
        pos = alpha_p * x * x + self._beta * x
        neg = alpha_n * (mx.exp(mx.minimum(x, self._eps)) - 1.0 - x) + self._beta * x
        return mx.where(x > 0, pos, neg)
