# cram/modules/retention.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

class RetentionMechanism(nn.Module):
    hidden_size: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        B, N, D = x.shape
        
        def compute_retention(carry, x_t):
            xi_prev = carry
            xi_t = jax.nn.sigmoid(x_t)
            xi_new = 0.5 * xi_t + 0.5 * xi_prev
            return xi_new, xi_new
        
        xi_init = jnp.zeros((B, D))
        _, xi_sequence = jax.lax.scan(compute_retention, xi_init, jnp.transpose(x, (1, 0, 2)))
        
        return jnp.transpose(xi_sequence, (1, 0, 2))
