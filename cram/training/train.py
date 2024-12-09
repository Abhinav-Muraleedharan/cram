
import time
from cram.models.cram_simple import ResidualNN, create_model, forward_pass
import jax
import jax.numpy as jnp
import optax
from typing import Any, Optional, Callable
from functools import partial


def mse_loss(params, model, batch, targets):
    """Calculate Mean Squared Error loss."""
    predictions = model.apply(params, batch, training=True)
    return jnp.mean((predictions - targets) ** 2)

def train_step(params, opt_state, model, batch, targets, tx):
    """Perform a single training step.
    
    Args:
        params: Model parameters
        opt_state: Optimizer state
        model: Flax model instance
        batch: Input batch
        targets: Target values
        tx: Optax optimizer
    
    Returns:
        Tuple of (updated params, updated optimizer state, loss)
    """
    loss_fn = lambda p: mse_loss(p, model, batch, targets)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Create a JIT-compiled version of the training step
@partial(jax.jit, static_argnums=(2,5))
def train_step_jit(params, opt_state, model, batch, targets, tx):
    return train_step(params, opt_state, model, batch, targets, tx)

def train_model(model, params, train_data, train_targets, 
                n_epochs: int = 100, batch_size: int = 128, 
                learning_rate: float = 1e-3):
    """Train the model.
    
    Args:
        model: ResidualNN instance
        params: Initial model parameters
        train_data: Training data of shape (n_samples, n_in)
        train_targets: Training targets of shape (n_samples, n_out)
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        Final model parameters
    """
    # Initialize optimizer
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)
    
    # Training loop
    n_batches = len(train_data) // batch_size
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        # Shuffle data
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, len(train_data))
        train_data_shuffled = train_data[perm]
        train_targets_shuffled = train_targets[perm]
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            
            batch = train_data_shuffled[batch_start:batch_end]
            targets = train_targets_shuffled[batch_start:batch_end]
            
            # Train step
            params, opt_state, loss = train_step_jit(
                params, opt_state, model, batch, targets, tx)
            total_loss += loss
            
        avg_loss = total_loss / n_batches
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return params

if __name__ == "__main__":
    start_time = time.time()
    # Generate some dummy data
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    n_samples = 1000
    n_in = 20
    n_hidden = 32
    n_out = 20
    
    # Random input data
    X = jax.random.normal(key, (n_samples, n_in))
    # Random target data
    Y = jax.random.normal(subkey, (n_samples, n_out))
    
    # Create and train model
    model, params = create_model(n_in, n_hidden, n_out, key)
    
    # Train the model
    final_params = train_model(model, params, X, Y)
    
    # Make predictions
    test_input = jax.random.normal(key, (1, n_in))
    prediction = forward_pass(final_params, model, test_input)
    end_time = time.time()
    print("Test prediction shape:", prediction.shape)
    print("Total Execution time:", end_time-start_time)