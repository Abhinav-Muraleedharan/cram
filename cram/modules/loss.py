import optax
import jax.numpy as jnp

def mse_loss(params, model, batch, targets):
    """Calculate Mean Squared Error loss."""
    predictions = model.apply(params, batch, training=True)
    return jnp.mean((predictions - targets) ** 2)

def cross_entropy_loss(params,model,batch,targets):
    predictions = model.apply(params,batch,training=True)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions, targets))