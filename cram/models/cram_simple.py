import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

class ResidualNN(nn.Module):
    n_in: int
    n_hidden: int
    n_out: int
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Input projection if dimensions don't match
        residual = nn.Dense(features=self.n_hidden)(x)
        
        # Hidden layer
        x = nn.Dense(features=self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_hidden)(x)
        x = nn.relu(x)
        
        # Add residual connection
        x = x + residual
        
        # Output layer
        x = nn.Dense(features=self.n_out)(x)
        return x

def create_model(n_in: int, n_hidden: int, n_out: int, key: Any) -> tuple[ResidualNN, Any]:
    """Create and initialize the model.
    
    Args:
        n_in: Input dimension
        n_hidden: Hidden layer size
        n_out: Output dimension
        key: JAX random key
    
    Returns:
        Tuple of (model, initialized parameters)
    """
    model = ResidualNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    params = model.init(key, jnp.ones((1, n_in)))
    return model, params

@jax.jit
def forward_pass(params, model, batch):
    """Perform a forward pass through the model.
    
    Args:
        params: Model parameters
        model: ResidualNN instance
        batch: Input batch of shape (batch_size, n_in)
    
    Returns:
        Model predictions
    """
    return model.apply(params, batch)